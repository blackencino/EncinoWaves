# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Solid-angle lighting analysis for a linear latitude/longitude HDR sky.

The nine real spherical harmonics use the order
``[1, y, z, x, xy, yz, 3*z*z-1, xz, x*x-y*y]`` and standard orthonormal
constants. Coefficients already include Lambert's cosine convolution and
``1/pi``: a shader only evaluates ``sum(diffuse_sh[i] * Y_i(normal))``.
Normals must be rotated into the environment's coordinate system first.

The SH basis is integrated analytically over each latitude/longitude texel,
including texels crossed by the horizon. The lower hemisphere contributes no
illumination. We never alter the supplied image or normalize its exposure.

References: Ramamoorthi and Hanrahan, SIGGRAPH 2001,
https://graphics.stanford.edu/papers/envmap/ ; Filament, image-based lights,
https://google.github.io/filament/main/filament.html#lighting/imagebasedlights .
The compact bright-source detector is a documented practical approximation,
not a claim to recover an obscured solar disk from an arbitrary photograph.
"""

from dataclasses import dataclass
import math

import numpy as np


_SH_CONSTANTS = np.array([
    .28209479177387814,
    .4886025119029199, .4886025119029199, .4886025119029199,
    1.0925484305920792, 1.0925484305920792, .31539156525252005,
    1.0925484305920792, .5462742152960396,
])
_LAMBERT_OVER_PI = np.array([1., 2/3, 2/3, 2/3, .25, .25, .25, .25, .25])
_LUMINANCE = np.array([.2126, .7152, .0722])


def _readonly(value):
    result = np.array(value, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Sky_lighting:
    """Exposure-independent radiometric quantities in the input sky's units.

    ``diffuse_sh`` contains the entire upper sky, suitable for foam illumination.
    ``ambient_sh`` excludes an extracted direct source; use it when adding that
    source separately. Both return Lambertian reflected radiance for albedo one.

    ``direct_irradiance`` is the integral of residual source radiance projected
    onto a plane perpendicular to ``dominant_direction``. It has irradiance
    units, unlike an HDR pixel. ``direct_angular_radius`` is the radius of a
    uniform cap with the same solid angle as residual source flux divided by
    peak residual luminance. It is an energy-equivalent angular size in radians.
    The direction is the source's radiance-weighted first moment, or the upper
    sky's first moment when no compact source is present. A black sky uses +Z.

    ``mean_radiance`` is the solid-angle mean of the complete upper sky.
    Arrays are copied and made read-only, as well as the value being frozen.
    """

    diffuse_sh: np.ndarray
    ambient_sh: np.ndarray
    dominant_direction: np.ndarray
    direct_irradiance: np.ndarray
    direct_angular_radius: float
    mean_radiance: np.ndarray

    def __post_init__(self):
        for name in ("diffuse_sh", "ambient_sh", "dominant_direction",
                     "direct_irradiance", "mean_radiance"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def has_direct(self):
        return bool(np.any(self.direct_irradiance > 0))


def sh_basis(directions):
    """Evaluate normalized real SH9 in the documented order at unit directions."""
    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape[-1:] != (3,) or not np.isfinite(directions).all():
        raise ValueError("SH directions must be finite vectors with three components")
    length = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(length <= 1e-12):
        raise ValueError("SH directions must have nonzero length")
    x, y, z = np.moveaxis(directions / length, -1, 0)
    return np.stack((np.ones_like(x), y, z, x, x*y, y*z, 3*z*z-1,
                     x*z, x*x-y*y), axis=-1) * _SH_CONSTANTS


def evaluate_diffuse(coefficients, directions):
    """Evaluate irradiance/pi, clamping SH truncation's negative ringing to zero."""
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (9, 3):
        raise ValueError("Diffuse SH coefficients must have shape (9, 3)")
    return np.maximum(sh_basis(directions) @ coefficients, 0)


def _texel_integrals(height, width):
    """Separable exact integrals of SH polynomials over upper-sky texels."""
    rows = (height + 1) // 2
    theta0 = np.arange(rows) * math.pi / height
    theta1 = np.minimum((np.arange(rows) + 1) * math.pi / height, math.pi / 2)
    phi0 = np.arange(width) * 2 * math.pi / width - math.pi
    phi1 = (np.arange(width) + 1) * 2 * math.pi / width - math.pi
    c0, c1 = np.cos(theta0), np.cos(theta1)
    dphi = np.full(width, 2 * math.pi / width)
    sintheta = c0 - c1
    sin2theta = (theta1-theta0)/2 - (np.sin(2*theta1)-np.sin(2*theta0))/4
    sin3theta = c0-c1 + (c1**3-c0**3)/3
    sin2costheta = (np.sin(theta1)**3-np.sin(theta0)**3)/3
    sinphi = np.cos(phi0)-np.cos(phi1)
    cosphi = np.sin(phi1)-np.sin(phi0)
    integrals = (
        (sintheta, dphi),
        (sin2theta, sinphi),
        ((c0*c0-c1*c1)/2, dphi),
        (sin2theta, cosphi),
        (sin3theta, (np.cos(2*phi0)-np.cos(2*phi1))/4),
        (sin2costheta, sinphi),
        (c0**3-c0-c1**3+c1, dphi),
        (sin2costheta, cosphi),
        (sin3theta, (np.sin(2*phi1)-np.sin(2*phi0))/2),
    )
    return integrals, (theta0+theta1)/2, (phi0+phi1)/2


def _project(rgb, integrals):
    # Integrating longitude first avoids allocating nine full-resolution maps.
    return np.array([
        np.einsum("h,w,hwc->c", latitude, longitude, rgb, optimize=True) * constant
        for (latitude, longitude), constant in zip(integrals, _SH_CONSTANTS)
    ])


def _first_moment(coefficients):
    # SH order y,z,x -> Cartesian x,y,z, with RGB columns.
    return coefficients[[3, 1, 2]] / _SH_CONSTANTS[1]


def _unit_direction(moment):
    direction = moment @ _LUMINANCE
    length = np.linalg.norm(direction)
    return direction/length if length > 1e-12 else np.array([0., 0., 1.])


def analyze_sky(pixels, *, max_cap_radius=math.radians(5), min_peak_contrast=8.):
    """Project an RGB/RGBA equirectangular image and optionally extract a source.

    A direct source requires a luminance peak at least ``min_peak_contrast``
    times its local 5--10 degree annulus background (for the default cap size).
    This prevents a broad bright cloud or constant dome becoming a fake sun.
    The source is the positive RGB residual over that background inside the
    cap. Its solid angle and perpendicular irradiance are integrated, never
    approximated by multiplying the brightest pixel by a hand-tuned gain.

    The cap selection uses texel centers; integration of selected pixels uses
    their exact spherical area and first moment. Very small or heavily blurred
    sources therefore retain the energy present in the image, but their angular
    extent is only as reliable as the image resolution. A dim/obscured sun may
    intentionally remain part of the diffuse sky instead of becoming a light.
    """
    pixels = np.asarray(pixels)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or min(pixels.shape[:2]) < 2:
        raise ValueError("Sky must be an RGB or RGBA latitude/longitude image")
    if not np.isfinite(pixels[..., :3]).all() or np.any(pixels[..., :3] < 0):
        raise ValueError("Sky radiance must be finite and nonnegative")
    if not math.isfinite(max_cap_radius) or not 0 < max_cap_radius <= math.pi/6:
        raise ValueError("Maximum source cap radius must be in (0, pi/6]")
    if not math.isfinite(min_peak_contrast) or min_peak_contrast <= 1:
        raise ValueError("Source peak contrast must exceed one")

    height, width = pixels.shape[:2]
    integrals, theta, phi = _texel_integrals(height, width)
    rgb = np.asarray(pixels[:len(theta), :, :3], dtype=np.float64)
    coefficients = _project(rgb, integrals)
    flux = coefficients[0] / _SH_CONSTANTS[0]
    diffuse_sh = coefficients * _LAMBERT_OVER_PI[:, None]
    ambient_sh = diffuse_sh.copy()
    dominant = _unit_direction(_first_moment(coefficients))
    direct = np.zeros(3)
    radius = 0.

    luminance = rgb @ _LUMINANCE
    peak_row, peak_column = np.unravel_index(np.argmax(luminance), luminance.shape)
    peak = luminance[peak_row, peak_column]
    cosine = (np.sin(theta)[:, None]
              * np.cos(phi[None, :]-phi[peak_column]) * math.sin(theta[peak_row])
              + np.cos(theta)[:, None] * math.cos(theta[peak_row]))
    cap = cosine >= math.cos(max_cap_radius)
    annulus = ((cosine < math.cos(max_cap_radius))
               & (cosine >= math.cos(2*max_cap_radius)))
    solid_angle = integrals[0][0][:, None] * integrals[0][1][None, :]
    annulus_area = np.sum(solid_angle * annulus)
    if annulus_area > 0:
        background = np.sum(rgb * (solid_angle*annulus)[..., None], axis=(0, 1)) / annulus_area
    else:
        # At exceptionally low resolutions a local ring may contain no texels.
        background = flux / (2*math.pi)
    background_luminance = float(background @ _LUMINANCE)

    if peak > 1e-12 and peak >= min_peak_contrast * max(background_luminance, 1e-12):
        residual = np.maximum(rgb-background, 0) * cap[..., None]
        source_sh = _project(residual, integrals)
        source_flux = source_sh[0] / _SH_CONSTANTS[0]
        source_moment = _first_moment(source_sh)
        dominant = _unit_direction(source_moment)
        direct = np.maximum(dominant @ source_moment, 0)
        ambient_sh -= source_sh * _LAMBERT_OVER_PI[:, None]
        residual_peak = float(np.max(residual @ _LUMINANCE))
        effective_solid_angle = float(source_flux @ _LUMINANCE) / max(residual_peak, 1e-12)
        radius = math.acos(np.clip(1-effective_solid_angle/(2*math.pi), -1, 1))

    return Sky_lighting(diffuse_sh, ambient_sh, dominant, direct, radius, flux/(2*math.pi))
