# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Static, metre-scaled foam relief; independent of wave shape and foam coverage.

RGBA stores slope-x, slope-y, a mean-one micro reflectance factor, and squared
slope length. Linear mips preserve reflectance and unresolved slope variance.
The default 4 m tile contains overlapping 6 cm, 16 cm and 42 cm cellular scales;
it has no preferred flow/wind direction and never scrolls over the ocean.

The material pattern is a procedural appearance model, not a bubble solver.
Crest's OceanFoam.hlsl similarly derives a separate foam normal from detail;
the slope-moment filtering follows the geometry-to-BRDF/LEAN approach.
"""
from pathlib import Path
import math

import numpy as np
from scipy.spatial import cKDTree
import wgpu


FOAM_DETAIL_WGSL = (Path(__file__).parent / "shaders" / "foam_detail.wgsl").read_text()


def _periodic_noise(resolution, cells, rng):
    lattice = rng.uniform(-1, 1, (cells, cells))
    coordinates = (np.arange(resolution)+.5)*cells/resolution
    lower = np.floor(coordinates).astype(np.int32) % cells
    upper = (lower+1) % cells
    f = coordinates-np.floor(coordinates)
    w = f*f*f*(f*(f*6-15)+10)
    row0 = lattice[lower[:, None], lower[None, :]]*(1-w) + lattice[lower[:, None], upper[None, :]]*w
    row1 = lattice[upper[:, None], lower[None, :]]*(1-w) + lattice[upper[:, None], upper[None, :]]*w
    return row0*(1-w[:, None])+row1*w[:, None]


def detail_mips(resolution=512, period=4.0, seed=0xF0A6):
    """Build deterministic float32 slope/material moments once, on the CPU.

    The periodic cell lookup and derivatives wrap both axes. Returns a tuple of
    read-only RGBA arrays; no frame-dependent calculation or wave data is used.
    """
    if not isinstance(resolution, int) or resolution < 32 or resolution > 2048 or resolution & (resolution-1):
        raise ValueError("Foam detail resolution must be a power of two from 32 to 2048")
    if not math.isfinite(period) or not 1 <= period <= 16:
        raise ValueError("Foam detail tile period must be between 1 and 16 metres")
    rng = np.random.Generator(np.random.PCG64(seed))
    coordinate = (np.arange(resolution)+.5)*period/resolution
    x, y = np.meshgrid(coordinate, coordinate)
    warp_cells = max(2, round(period/.8))
    x = (x + .045*_periodic_noise(resolution, warp_cells, rng)) % period
    y = (y + .045*_periodic_noise(resolution, warp_cells, rng)) % period
    query = np.stack((x.ravel(), y.ravel()), axis=1)
    height = np.zeros((resolution, resolution), np.float64)
    micro_reflectance = np.zeros_like(height)
    for spacing, relief, weight in ((.06, .0018, .45), (.16, .0030, .35), (.42, .0048, .20)):
        count = max(8, round((period/spacing)**2))
        points = rng.uniform(0, period, (count, 2))
        distances, _ = cKDTree(points, boxsize=period).query(query, k=2)
        first = distances[:, 0].reshape(height.shape)
        gap = (distances[:, 1]-distances[:, 0]).reshape(height.shape)
        caps = np.exp(-np.square(first/(.45*spacing)))
        walls = np.exp(-np.square(gap/(.12*spacing)))
        # Rounded cells plus narrow connecting films, warped continuously above.
        # Relief amplitudes are millimetres, never ocean displacement.
        pattern = .75*caps + .25*walls
        height += relief*pattern
        micro_reflectance += weight*pattern
    step = period/resolution
    slope_x = (np.roll(height, -1, axis=1)-np.roll(height, 1, axis=1))/(2*step)
    slope_y = (np.roll(height, -1, axis=0)-np.roll(height, 1, axis=0))/(2*step)
    contrast = micro_reflectance-micro_reflectance.mean()
    contrast *= .08/max(float(np.abs(contrast).max()), 1e-12)
    base = np.stack((slope_x, slope_y, 1+contrast, slope_x*slope_x+slope_y*slope_y), axis=-1).astype(np.float32)
    levels = [base]
    while levels[-1].shape[0] > 1:
        previous = levels[-1]
        n = previous.shape[0]//2
        levels.append(previous.reshape(n, 2, n, 2, 4).mean(axis=(1, 3)))
    for level in levels:
        level.flags.writeable = False
    return tuple(levels)


class Foam_detail:
    """Upload a reusable static texture with all its prefiltered mip levels.

    Integrate one sample at ``ocean_material_metres / detail.period`` using a
    repeating trilinear sampler. Prepend ``FOAM_DETAIL_WGSL`` to use the helper
    normal and unresolved-variance functions. No update method is needed.
    """
    def __init__(self, device, resolution=512, period=4.0, seed=0xF0A6):
        levels = detail_mips(resolution, period, seed)
        self.period = float(period)
        self.texture = device.create_texture(
            label="Static foam relief and slope moments", size=(resolution, resolution, 1),
            format="rgba16float", mip_level_count=len(levels),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
                  | wgpu.TextureUsage.COPY_SRC)
        self.view = self.texture.create_view()
        for level, pixels in enumerate(levels):
            n = pixels.shape[0]
            device.queue.write_texture({"texture": self.texture, "mip_level": level},
                pixels.astype(np.float16), {"bytes_per_row": n*8, "rows_per_image": n}, (n, n, 1))

    def destroy(self):
        self.texture.destroy()
