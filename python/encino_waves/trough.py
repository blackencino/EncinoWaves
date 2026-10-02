# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Quiet resolved small ripples in low water with a spatial dry/wet blend.

The Gaussian sigma is measured in world metres on the undisplaced grid. Only
height changes: horizontal displacement and the original crest source stay
intact, and the caller computes normals from the final displaced surface.
No additional FFTs or spectral state changes are involved.
"""
from functools import lru_cache
import math
import torch
import torch.nn.functional as functional


_MAX_RADIUS = 16


def _smoothstep(value):
    t = value.clamp(0, 1)
    return t*t*(3-2*t)


def crest_from_derivatives(dxx, dyy, dxy, pinch):
    jxx, jyy, jxy = 1-pinch*dxx, 1-pinch*dyy, -pinch*dxy
    return -.5*(jxx+jyy-torch.sqrt((jxx-jyy).square()+4*jxy.square()))


def trough_retention(height, amount):
    # Statistics count each periodic texel once, excluding duplicated borders.
    std, mean = torch.std_mean(height, correction=0)
    depth = (mean-height)/(math.sqrt(2)*std).clamp_min(1e-12)
    retention = 1-amount*_smoothstep(depth)
    # A flat guide field contains no meaningful trough/crest distinction.
    return torch.where(std > 1e-8, retention, torch.ones_like(retention))


@lru_cache(maxsize=64)
def _gaussian_kernel(sigma_pixels, dtype, device):
    # Truncate at three sigma. Very fine grids cap sigma at 16/3 pixels rather
    # than extending a large blur across the sea: at most 33 taps per axis.
    # Thus the effective world sigma is min(requested_sigma, 16*spacing/3).
    sigma = min(sigma_pixels, _MAX_RADIUS/3)
    radius = min(_MAX_RADIUS, max(1, math.ceil(3*sigma)))
    weights = [math.exp(-.5*(i/sigma)**2) for i in range(-radius, radius+1)]
    total = sum(weights)
    return torch.tensor([weight/total for weight in weights], dtype=dtype, device=device)


def smooth_height(height, spacing, sigma):
    """Periodic separable Gaussian; sigma and grid spacing are world metres.

    Support is three sigma, bounded to 16 samples per side. For exceptionally
    fine grids this reduces the effective sigma instead of widening the blur.
    The normal viewer domains/resolutions use the requested 0.1 m sigma in full.
    """
    kernel = _gaussian_kernel(sigma/spacing, height.dtype, str(height.device))
    radius = len(kernel)//2
    source = height[None, None]
    # Circular padding supports one wrap. The minimum wave resolution is 16,
    # matching the maximum radius, so every valid ocean grid is supported.
    horizontal = functional.conv2d(functional.pad(source, (radius, radius, 0, 0), mode="circular"),
                                   kernel.reshape(1, 1, 1, -1))
    return functional.conv2d(functional.pad(horizontal, (0, 0, radius, radius), mode="circular"),
                             kernel.reshape(1, 1, -1, 1))[0, 0]


@torch.inference_mode()
def damp_troughs(p, fields):
    spacing = p.domain/p.resolution
    # A Gaussian's half-amplitude wavelength is about 5.34 sigma. Do nothing
    # unless the grid can represent the requested ~5-sigma ripple scale, even
    # on its shortest diagonal mode. In particular 512 m / 1024 is unchanged.
    if p.trough_damping == 0 or 5*p.trough_smoothing_length <= math.sqrt(2)*spacing:
        return fields[:3]
    height = fields[0]
    smoothed = smooth_height(height, spacing, p.trough_smoothing_length)
    retention = trough_retention(smoothed, p.trough_damping)
    # This form leaves the entire crest side bit-identical when retention=1.
    blended = height+(1-retention)*(smoothed-height)
    return torch.stack((blended, fields[1], fields[2]))
