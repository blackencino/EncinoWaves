# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Post-seed spectral evaluation and phase-continuous parameter editing.

The lattice and random variates are values, reused for all edits on that grid.
Every spectrum evaluation runs on the Torch device, including normalization of
directional spreading. Radial quantities are evaluated once per distinct |k|.
No interpolated spectrum tables, amplitude crossfades, or extra wave layers.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
import torch
from .model import (Wave_parameters, Initial_state, Phase_step, select_device,
                    random_variates, peak_omega, phase_at)


def same_wave_basis(a, b):
    return (a.resolution, a.domain, a.seed) == (b.resolution, b.domain, b.seed)


@dataclass(frozen=True)
class Wave_basis:
    parameters: Wave_parameters
    radial_k: torch.Tensor
    radial_index: torch.Tensor
    angle: torch.Tensor
    noise_positive: torch.Tensor
    noise_negative: torch.Tensor
    multipliers: torch.Tensor
    quadrature: torch.Tensor  # positive half of the 64-point Gauss rule

    @property
    def device(self):
        return self.radial_k.device


@torch.inference_mode()
def make_wave_basis(parameters=Wave_parameters(), device="auto"):
    p = parameters
    device = select_device(str(device))
    n, width = p.resolution, p.resolution//2+1
    ix = np.arange(width, dtype=np.int64)[None, :]
    j = np.arange(n, dtype=np.int64)
    iy = np.where(j <= n//2, j, j-n)[:, None]
    squared, index = np.unique(ix*ix+iy*iy, return_inverse=True)
    dk = 2*math.pi/p.domain
    kx, ky = ix*dk, iy*dk
    k = np.hypot(kx, ky)
    safe_k = np.maximum(k, 1e-12)
    amp_pos, amp_neg, phase_pos, phase_neg = random_variates(kx, ky, p.seed)
    positive = amp_pos*np.exp(-1j*phase_pos)
    negative = amp_neg*np.exp(-1j*phase_neg)
    positive[0, 0] = negative[0, 0] = 0
    multipliers = np.empty((6, n, width), np.complex64)
    multipliers[0] = 1
    multipliers[1] = -1j*kx/safe_k
    multipliers[2] = -1j*ky/safe_k
    multipliers[3] = kx*kx/safe_k
    multipliers[4] = ky*ky/safe_k
    multipliers[5] = kx*ky/safe_k
    multipliers[:, 0, 0] = 0
    angles, weights = np.polynomial.legendre.leggauss(64)
    def tensor(array, dtype):
        return torch.from_numpy(np.asarray(array, dtype=dtype)).to(device)
    return Wave_basis(p, tensor(np.sqrt(squared)*dk, np.float32),
        tensor(index.reshape(n, width), np.int64),
        tensor(np.arctan2(-ky, kx), np.float32),
        tensor(positive, np.complex64), tensor(negative, np.complex64),
        tensor(multipliers, np.complex64),
        tensor(np.stack((angles[32:], weights[32:]*2)), np.float32))


def _dispersion(p, k):
    if p.dispersion == "deep":
        omega = torch.sqrt(p.gravity*k)
        derivative = p.gravity/(2*omega.clamp_min(1e-12))
    else:
        hk = p.depth*k
        th = torch.tanh(hk)
        k2s = k*k*(p.surface_tension/p.density if p.dispersion == "capillary" else 0)
        omega = torch.sqrt(k*(p.gravity+k2s)*th)
        numerator = (p.gravity+3*k2s)*th + hk*(p.gravity+k2s)*(1-th*th)
        derivative = numerator/(2*omega.clamp_min(1e-12))
    return omega, torch.where(k > 0, derivative, 0)


def _spectrum(p, omega):
    w = omega.clamp_min(1e-12)
    # Log form avoids inf * 0 at the DC mode and at very small frequencies.
    if p.spectrum == "pm":
        wm = .87*p.gravity/p.wind_speed
        log_s = math.log(.0081*p.gravity**2)-5*torch.log(w)-1.291*(wm/w)**4
    else:
        wm = peak_omega(p)
        x = p.gravity*p.fetch_km*1000/p.wind_speed**2
        sigma = torch.where(w <= wm, .07, .09)
        peak = math.log(p.gamma)*torch.exp(-.5*((w-wm)/(sigma*wm))**2)
        log_s = math.log(.076*x**-.22*p.gravity**2)-5*torch.log(w)-1.25*(wm/w)**4+peak
    result = torch.exp(log_s)
    if p.spectrum == "tma":
        # Same depth factor: sigmoid(2*x) == .5 + .5*tanh(x). The direct
        # sigmoid avoids cancellation in the small-factor end of the curve.
        result = result*torch.sigmoid(3.6*(w*math.sqrt(p.depth/p.gravity)-1.125))
    return torch.where(omega > 0, result, 0)


def _product(p, angle, beta, bias):
    if p.spreading == "donelan_banner":
        # sech^2 without overflowing cosh, even in irrelevant low-energy bins.
        e = torch.exp(-2*torch.abs(beta*angle))
        base = 4*e/(1+e)**2
    else:
        # Exact zero at the endpoints avoids a spurious reverse-going wave from
        # float32 cos(pi/2), which is about -4e-8 rather than zero.
        base = torch.where(torch.abs(angle) < math.pi/2, torch.cos(angle)**2, 0)
    half_cos = torch.abs(torch.sin((math.pi-torch.abs(angle))/2))
    return base*half_cos**(2*bias)


def _product_normalization(p, beta, bias, quadrature):
    if p.convention == "paper" and p.swell <= 0 and p.spreading == "donelan_banner":
        return 2*torch.tanh(beta*math.pi)/beta
    if p.convention == "paper" and p.swell == 0 and p.spreading == "cosine_squared":
        return torch.full_like(beta, math.pi/2)
    if p.convention == "paper":
        extent = math.pi/2 if p.spreading == "cosine_squared" else math.pi
        angles, weights = quadrature*extent
    else:
        # Original 36-segment trapezoid, using even symmetry.
        angles = torch.arange(19, device=beta.device)*math.pi/36
        weights = torch.full_like(angles, math.pi/18)
        weights[0] = weights[-1] = math.pi/36
    pieces = []
    # Bound scratch memory at 32 * 65536 floats, also at 4096 squared.
    for start in range(0, beta.numel(), 65536):
        section = slice(start, start+65536)
        values = _product(p, angles[:, None], beta[section][None], bias[section][None])
        pieces.append(torch.sum(weights[:, None]*values, dim=0))
    return torch.cat(pieces)


@torch.inference_mode()
def state_from_basis(basis, parameters):
    """Evaluate the paper's per-wavenumber equations after the fixed seed stage."""
    p = parameters
    if not same_wave_basis(basis.parameters, p):
        raise ValueError("domain, resolution, or seed changed; create a new wave basis")
    k, index = basis.radial_k, basis.radial_index
    omega, derivative = _dispersion(p, k)
    energy = _spectrum(p, omega)*(2*math.pi/p.domain)**2*derivative/k.clamp_min(1e-12)
    ratio = omega.clamp_min(1e-12)/peak_omega(p, directional=True)
    coefficient = 16 if p.convention == "paper" else 16.1
    bias = coefficient*torch.tanh(1/ratio)*max(0.0, p.swell)**2
    # Wind direction places the mesh; the seeded ocean always has +X wind.
    theta = basis.angle
    opposite = torch.where(theta < 0, theta+math.pi, theta-math.pi)
    if p.spreading in ("hasselmann", "mitsuyasu"):
        u_over_c = p.wind_speed*peak_omega(p, directional=True)/p.gravity
        if p.spreading == "hasselmann":
            shape = torch.where(ratio <= 1, 6.97*ratio**4.06,
                                9.77*ratio**(-2.33-1.45*(u_over_c-1.17)))
        else:
            shape = 11.5*u_over_c**-2.5*torch.where(ratio <= 1, ratio**5, ratio**-2.5)
        shape = shape+bias
        # Duplication identity avoids cancellation between large lgamma terms.
        log_q = torch.lgamma(shape+1)-torch.lgamma(shape+.5)-math.log(2*math.sqrt(math.pi))
        def directional(angle):
            half_cos = torch.abs(torch.sin((math.pi-torch.abs(angle))/2))
            return torch.exp(log_q[index])*half_cos**(2*shape[index])
    else:
        beta = torch.where(ratio < .95, 2.61*ratio**1.3,
               torch.where(ratio < 1.6, 2.28*ratio**-1.3,
                   10**(-.4+.8393*torch.exp(-.567*torch.log(ratio.clamp_min(1.6)**2)))))
        denominator = _product_normalization(p, beta, bias, basis.quadrature).clamp_min(1e-30)
        def directional(angle):
            return _product(p, angle, beta[index], bias[index])/denominator[index]
    def coefficients(angle, noise):
        direction = directional(angle)
        if p.swell < 0:
            direction = (1+p.swell)*direction-p.swell/(2*math.pi)
        return noise*torch.sqrt((2*energy[index]*direction).clamp_min(0))
    return Initial_state(p, coefficients(theta, basis.noise_positive),
                         coefficients(opposite, basis.noise_negative), omega[index], basis.multipliers)


def _same_dispersion(a, b):
    def key(p):
        return (p.gravity, None if p.dispersion == "deep" else p.depth,
                p.surface_tension/p.density if p.dispersion == "capillary" else 0)
    return key(a) == key(b)


def edit_state(basis, previous, parameters, time):
    """Retain the wave tensors for mesh rotation; retune other edits in phase."""
    spatial = ("wind_direction", "pinch", "amplitude_gain", "trough_damping",
               "trough_smoothing_length", "trough_filter_revision")
    if replace(previous.parameters, **{name:getattr(parameters, name) for name in spatial}) == parameters:
        return replace(previous, parameters=parameters)
    return preserve_phase(previous, state_from_basis(basis, parameters), time)


@torch.inference_mode()
def preserve_phase(previous, target, time):
    """Retarget a spectrum without restarting the travelling waves.

    The explicit phase segments record the edits, not hidden evaluation history.
    The returned state can be evaluated at any time in any order.
    """
    if not same_wave_basis(previous.parameters, target.parameters) or previous.device != target.device:
        raise ValueError("phase preservation requires the same wave basis and device")
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    if _same_dispersion(previous.parameters, target.parameters):
        return replace(target, phase=previous.phase, phase_steps=previous.phase_steps)
    if previous.phase_steps and time == previous.phase_steps[-1].time:
        # Editing a paused ocean does not accumulate propagation segments.
        return replace(target, phase=previous.phase, phase_steps=previous.phase_steps)
    step = Phase_step.from_parameters(previous.parameters, time)
    phase = torch.remainder(phase_at(previous, time), 2*math.pi)
    return replace(target, phase=phase, phase_steps=previous.phase_steps+(step,))


@torch.inference_mode()
def restore_phase(basis, state, steps):
    """Replay only dispersion on the radial grid; works at export resolutions too."""
    if not same_wave_basis(basis.parameters, state.parameters) or basis.device != state.device:
        raise ValueError("phase replay requires the state's wave basis and device")
    steps = tuple(steps)
    if not steps:
        return replace(state, phase=None, phase_steps=())
    phase, previous_time = None, 0.0
    for step in steps:
        omega, _ = _dispersion(step, basis.radial_k)
        advance = omega*(step.time-previous_time)
        phase = torch.remainder(advance if phase is None else phase+advance, 2*math.pi)
        previous_time = step.time
    return replace(state, phase=phase[basis.radial_index], phase_steps=steps)


def make_edited_state(parameters, device="auto", phase_steps=()):
    """Recreate a saved edited sea, including at a different FFT resolution."""
    basis = make_wave_basis(parameters, device)
    return restore_phase(basis, state_from_basis(basis, parameters), phase_steps)


def follow_parameters(current, target, elapsed, response=.10):
    """Smooth control motion; evaluate actual parameters, never blended spectra."""
    if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(response) or response <= 0:
        raise ValueError("elapsed must be nonnegative and response must be positive")
    if not same_wave_basis(current, target):
        raise ValueError("domain, resolution, or seed changed; create a new wave basis")
    amount = -math.expm1(-elapsed/response)
    values = {}
    for name in ("wind_speed", "fetch_km", "depth", "swell", "wind_direction", "pinch", "amplitude_gain",
                 "trough_damping", "trough_smoothing_length"):
        a, b = getattr(current, name), getattr(target, name)
        if name in ("wind_speed", "fetch_km", "depth"):
            distance = math.log(b/a)
            value = a*math.exp(amount*distance)
        else:
            distance = b-a
            if name == "wind_direction": distance = (distance+180)%360-180
            value = a+amount*distance
        values[name] = b if abs(distance) < 1e-4 else value
    return replace(target, **values)
