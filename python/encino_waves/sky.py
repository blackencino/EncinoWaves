# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Linear HDR environment loading; original commercial assets stay local."""
from pathlib import Path
import math
import os
import numpy as np
import OpenImageIO as oiio


def find_sky():
    explicit = os.environ.get("ENCINO_WAVES_SKY")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    roots = (Path("assets/local"), Path.home()/"Pictures"/"Dutch Skies",
             Path.home()/"HDR", Path.home()/"HDRI")
    for root in roots:
        if root.is_dir():
            files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".hdr", ".exr"))
            if files:
                return files[0]
    return None


def load_sky(path=None, max_width=4096):
    path = Path(path).expanduser() if path else find_sky()
    if path is None:
        return procedural_sky(), "Procedural sky (select a Dutch Skies HDR)"
    source = oiio.ImageBuf(str(path))
    if not source.read():
        raise RuntimeError(f"Cannot read sky {path}: {source.geterror()}")
    spec = source.spec()
    if spec.width > max_width:
        height = round(spec.height*max_width/spec.width)
        source = oiio.ImageBufAlgo.resize(source, roi=oiio.ROI(0,max_width,0,height,0,1,0,spec.nchannels))
    rgb = np.asarray(source.get_pixels(oiio.FLOAT), np.float32)[...,:3]
    if rgb.shape[-1] != 3 or rgb.shape[1] < rgb.shape[0]:
        raise ValueError("Sky must be an RGB latitude/longitude panorama")
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        rgb = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    if not np.isfinite(rgb).all():
        raise ValueError("Sky contains non-finite pixels")
    rgba = np.ones((*rgb.shape[:2], 4), np.float32)
    rgba[...,:3] = np.maximum(rgb, 0)
    return rgba, str(path.resolve())


def procedural_sky(width=2048):
    """A usable offline fallback, clearly named in the UI; no external assets."""
    height = width//2
    y,x = np.mgrid[0:height,0:width]
    latitude = math.pi/2 - (y+.5)/height*math.pi
    longitude = (x+.5)/width*2*math.pi - math.pi
    z = np.sin(latitude)
    horizon = np.exp(-np.abs(z)*5)
    rgb = np.zeros((height,width,3),np.float32)
    rgb[:] = [.11,.22,.39]
    rgb += horizon[...,None]*np.array([.65,.58,.46])
    # Deterministic periodic cloud structure, used only for fallback lighting.
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(321)
    clouds = np.zeros((height,width))
    for sigma,weight in ((85,1),(34,.45),(13,.20),(5,.08)):
        noise = gaussian_filter(rng.normal(size=(height,width)),sigma,mode="wrap")
        clouds += weight*noise/(np.std(noise)+1e-12)
    coverage = np.clip((clouds+.5)*.32,0,.88)*(z>0)
    rgb = rgb*(1-coverage[...,None]) + np.array([.68,.68,.64])*coverage[...,None]
    dot_sun = np.cos(latitude)*np.cos(longitude-.65)*math.cos(.22) + z*math.sin(.22)
    glow = np.exp((dot_sun-1)*90)*1.3 + np.exp((dot_sun-1)*90000)*1.5
    rgb += glow[...,None]*np.array([1,.76,.48])
    rgb[z<0] = [.13,.15,.17]
    return np.concatenate((rgb,np.ones((height,width,1),np.float32)),axis=-1)
