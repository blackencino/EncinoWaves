# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Functional port of the spatial trough blend in the 2015 Propagation.h.

This appearance filter follows spectral synthesis; it is not a spectrum model.
The original crest source is retained, and normals follow the displaced surface.
"""
import math
import torch


def _smoothstep(value):
    t = value.clamp(0, 1)
    return t*t*(3-2*t)


def crest_from_derivatives(dxx, dyy, dxy, pinch):
    jxx, jyy, jxy = 1-pinch*dxx, 1-pinch*dyy, -pinch*dxy
    return -.5*(jxx+jyy-torch.sqrt((jxx-jyy).square()+4*jxy.square()))


def trough_retention(crest, amount):
    # Statistics count each periodic texel once, excluding duplicated borders.
    std, mean = torch.std_mean(crest, correction=0)
    t = _smoothstep((crest-mean)/(2.2*std).clamp_min(1e-12))
    retention = 1-amount+amount*t
    # A flat guide field contains no meaningful trough/crest distinction.
    return torch.where(std > 1e-8, retention, torch.ones_like(retention))


@torch.inference_mode()
def damp_troughs(p, spectral_height, multipliers, fields):
    if p.trough_damping == 0:
        return fields[:3]
    # dxx + dyy multipliers equal |k|. Invert the old smooth wavelength band
    # to synthesize the field that remains when its short-wave detail is removed.
    k = (multipliers[3]+multipliers[4]).real
    wavelength = 2*math.pi/k.clamp_min(1e-12)
    band = (_smoothstep(wavelength/p.trough_small_wavelength) -
            _smoothstep((wavelength-p.trough_big_wavelength)/p.trough_soft_width))
    filtered = torch.fft.irfft2(multipliers*(spectral_height*(1-band))[None],
                                s=(p.resolution, p.resolution), norm="forward")
    # The original guide uses a fixed positive pinch, so damping still works
    # when the artist turns horizontal displacement off. Standardization removes
    # its scale. Positive values mark crests, not troughs.
    guide = crest_from_derivatives(*filtered[3:].unbind(), 1.25)
    retention = trough_retention(guide, p.trough_damping)
    return filtered[:3]+retention*(fields[:3]-filtered[:3])
