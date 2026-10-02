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
    project_assets = Path(__file__).resolve().parents[2]/"assets"/"local"
    roots = (Path("assets/local"), project_assets, Path.home()/"Pictures"/"Dutch Skies",
             Path.home()/"HDR", Path.home()/"HDRI")
    for root in roots:
        if root.is_dir():
            # sIBL packs pair a detailed _Ref panorama with a tiny, blurred _Env
            # map. The original water shader needs the detailed reflection sky.
            files = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".hdr", ".exr")),
                           key=lambda p: (not p.stem.lower().endswith("_ref"), str(p).casefold()))
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


def ocean_dome(rgba, horizon_trim_degrees=8.0):
    """Return a linear HDR ocean dome without the photographed ground.

    The input is a north-to-south latitude/longitude image with pixel-centered
    samples. The visible horizon samples ``horizon_trim_degrees`` above the
    source horizon, tapering this angular offset to zero at the zenith. No
    horizontal resampling, exposure adjustment, or color conversion is applied.

    Below the horizon, each longitude extends its cleaned horizon color and
    smoothly darkens to 20 percent at the nadir. Alpha is interpolated without
    this attenuation. The source is never modified; the result is float32 RGBA.
    """
    source = np.asarray(rgba, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 4 or source.shape[0] < 2 or source.shape[1] < 1:
        raise ValueError("Sky dome requires an RGBA latitude/longitude image with at least two rows")
    trim = float(horizon_trim_degrees)
    if not math.isfinite(trim) or not 0 <= trim < 90:
        raise ValueError("Horizon trim must be finite and in [0, 90) degrees")
    if not np.isfinite(source).all():
        raise ValueError("Sky dome source must contain only finite pixels")

    height = source.shape[0]
    latitude = math.pi/2 - (np.arange(height)+.5)*math.pi/height
    trim = math.radians(trim)
    source_latitude = trim + np.maximum(latitude, 0)*(1-trim/(math.pi/2))
    source_y = (.5-source_latitude/math.pi)*height-.5
    # Limit both interpolation taps to strictly positive source elevations.
    # This also prevents ground leaking through a zero-trim horizon or tiny map.
    last_sky_row = height//2-1
    source_y = np.clip(source_y, 0, last_sky_row)
    lower = np.floor(source_y).astype(np.intp)
    upper = np.minimum(lower+1, last_sky_row)
    weight = (source_y-lower).astype(np.float32)[:,None,None]
    result = source[lower]*(1-weight) + source[upper]*weight

    depth = np.maximum(-latitude/(math.pi/2), 0)
    fade = (1-.8*depth*depth*(3-2*depth)).astype(np.float32)
    result[...,:3] *= fade[:,None,None]
    np.maximum(result, 0, out=result)
    return result
