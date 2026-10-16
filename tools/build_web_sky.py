#!/usr/bin/env python3
"""Build the licensed School dome and GGX reflection assets for the browser.

Run using the native viewer environment, for example:
  PYTHONPATH=python .venv/bin/python tools/build_web_sky.py --source /path/to/DS360_Autumn_Pack_01_03a_Ref.hdr

The photograph is separately licensed; this program is Apache-2.0.
PNG RGB stores sqrt(linear radiance / scale), alpha is always opaque. Each
vertical atlas contains every mip, largest first. Decode after sampling by
squaring and multiplying by scale. Mips are computed in LINEAR radiance.
The CPU GGX convolution follows shaders/environment_filter.wgsl.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from PIL import Image

from encino_waves.sky import load_sky, ocean_dome
from encino_waves.lighting import analyze_sky
from encino_waves.foam_detail import detail_mips


def mip_chain(base):
    levels = [base]
    while max(levels[-1].shape[:2]) > 1:
        old = levels[-1]
        h, w = old.shape[:2]
        if h == 1:
            new = old.reshape(1, w // 2, 2, 3).mean(axis=2)
        elif w == 1:
            new = old.reshape(h // 2, 2, 1, 3).mean(axis=1)
        else:
            new = old.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
        levels.append(new.astype(np.float32))
    return levels


def sample_level(pixels, uv):
    h, w = pixels.shape[:2]
    x = uv[..., 0] * w - .5
    y = np.clip(uv[..., 1] * h - .5, 0, h - 1)
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    wx = (x - ix)[..., None]
    wy = (y - iy)[..., None]
    lo = pixels[iy, ix % w] * (1-wx) + pixels[iy, (ix+1) % w] * wx
    hi = pixels[np.minimum(iy+1, h-1), ix % w] * (1-wx) + pixels[np.minimum(iy+1, h-1), (ix+1) % w] * wx
    return lo * (1-wy) + hi * wy


def sample_mips(levels, direction, lod):
    uv = np.stack((np.arctan2(direction[..., 1], direction[..., 0])/(2*np.pi)+.5,
                   np.arccos(np.clip(direction[..., 2], -1, 1))/np.pi), axis=-1)
    lower = np.floor(lod).astype(np.int32)
    fraction = lod-lower
    result = np.zeros((*lod.shape, 3), np.float64)
    for i in range(len(levels)):
        weight = np.where(lower == i, 1-fraction, 0) + np.where(lower+1 == i, fraction, 0)
        mask = weight > 0
        if mask.any():
            result[mask] += sample_level(levels[i], uv[mask]) * weight[mask, None]
    return result


def reverse_bits(number):
    return int(f'{number:032b}'[::-1], 2)


def ggx_mips(source, width=512, samples=128):
    levels = int(math.log2(width))+1
    output = []
    sh, sw = source[0].shape[:2]
    for level in range(levels):
        w, h = max(1, width >> level), max(1, (width//2) >> level)
        roughness = level/(levels-1)
        yy, xx = np.mgrid[:h, :w]
        uv = np.stack(((xx+.5)/w, (yy+.5)/h), -1)
        if level == 0:
            output.append(sample_level(source[int(math.log2(sw/width))], uv).astype(np.float32))
            continue
        print(f'GGX level {level}: {w}x{h}, roughness {roughness:.3f}', flush=True)
        phi, theta = (uv[..., 0]-.5)*2*np.pi, uv[..., 1]*np.pi
        normal = np.stack((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)), -1)
        tangent = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), -1)
        bitangent = np.cross(normal, tangent)
        alpha2 = roughness**4
        radiance, total = np.zeros((h,w,3)), np.zeros((h,w))
        for i in range(samples):
            xi0, xi1 = (i+.5)/samples, reverse_bits(i)*2.3283064365386963e-10
            azimuth = 2*np.pi*xi0
            ct = math.sqrt((1-xi1)/max(1+(alpha2-1)*xi1, 1e-7))
            st = math.sqrt(max(0,1-ct*ct))
            half = tangent*(math.cos(azimuth)*st)+bitangent*(math.sin(azimuth)*st)+normal*ct
            light = 2*np.sum(normal*half, axis=-1)[...,None]*half-normal
            light /= np.linalg.norm(light, axis=-1)[...,None]
            nl = np.maximum(np.sum(normal*light, axis=-1),0)
            nh = np.maximum(np.sum(normal*half, axis=-1),0)
            distribution = alpha2/(np.pi*(nh*nh*(alpha2-1)+1)**2)
            pdf = np.maximum(distribution*.25,1e-9)
            angle = np.arccos(np.clip(light[...,2],-1,1))
            row = np.minimum(np.floor(angle*sh/np.pi),sh-1)
            area = np.maximum((4*np.pi/sw)*np.sin((row+.5)*np.pi/sh)*math.sin(.5*np.pi/sh),1e-12)
            lod = np.clip(.5*np.log2(1/(samples*pdf*area)),0,len(source)-1)
            radiance += sample_mips(source,light,lod)*nl[...,None]
            total += nl
        output.append((radiance/np.maximum(total[...,None],1e-7)).astype(np.float32))
    return output


def write_atlas(path, levels, scale):
    atlas = np.zeros((sum(p.shape[0] for p in levels), levels[0].shape[1], 3),np.uint8)
    offset, dimensions = 0, []
    for pixels in levels:
        h,w = pixels.shape[:2]
        atlas[offset:offset+h,:w] = np.round(np.sqrt(np.clip(pixels/scale,0,1))*255).astype(np.uint8)
        dimensions.append({'width':w,'height':h,'y':offset})
        offset += h
    Image.fromarray(atlas).save(path, optimize=True)
    return {'url':path.name,'scale':float(scale),'levels':dimensions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'web/assets')
    parser.add_argument('--width',type=int,default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    pixels, _ = load_sky(args.source,max_width=args.width)
    dome = ocean_dome(pixels,10.0)[...,:3]
    lighting = analyze_sky(dome)
    levels = mip_chain(dome)
    reflection = ggx_mips(levels)
    scale = float(np.max(dome))*1.001
    metadata = {
        'encoding':'sqrt-radiance-rgb8',
        'attribution':'Dutch Skies 360, Bob Groothuis — Autumn Pack 01, 03a. Used with permission.',
        'horizonTrimDegrees':10,
        'sky':write_atlas(args.output/'school-sky.png',levels,scale),
        'reflection':write_atlas(args.output/'school-reflection.png',reflection,scale),
        'diffuseSH':lighting.diffuse_sh.tolist(),
        'ambientSH':lighting.ambient_sh.tolist(),
        'sunDirection':lighting.dominant_direction.tolist(),
        'sunIrradiance':lighting.direct_irradiance.tolist(),
        'meanRadiance':lighting.mean_radiance.tolist(),
    }
    detail = detail_mips(256)
    with (args.output/'foam-detail.bin').open('wb') as out:
        for level in detail:
            out.write(level.astype('<f2').tobytes())
    metadata['foamDetail']={'url':'foam-detail.bin','resolution':256,'levels':len(detail),'period':4.0}
    (args.output/'sky.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Wrote assets to {args.output}',flush=True)


if __name__ == '__main__':
    main()
