# Copyright 2015-2026 Christopher Jon Horvath. Apache-2.0.
"""Functional spectral ocean model; equation numbers refer to Horvath (2015).

Setup is evaluated in float64 on the CPU, once per parameter change. All
per-frame work, including six batched inverse FFTs, runs on the selected GPU.
No autograd, simulation history, amplitude normalization, or hidden wave layers.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.special import gammaln
import torch


@dataclass(frozen=True)
class Wave_parameters:
    resolution: int = 1024
    domain: float = 512.0
    wind_speed: float = 17.0
    fetch_km: float = 300.0
    depth: float = 100.0
    swell: float = 0.35
    spreading: str = "donelan_banner"
    spectrum: str = "tma"
    dispersion: str = "capillary"
    wind_direction: float = 0.0  # degrees
    gravity: float = 9.81
    surface_tension: float = 0.074
    density: float = 1000.0
    pinch: float = 0.75
    amplitude_gain: float = 1.0
    seed: int = 54321
    gamma: float = 3.3
    convention: str = "paper"

    def __post_init__(self):
        n = self.resolution
        if not isinstance(n, int) or n < 16 or n > 4096 or n & (n - 1):
            raise ValueError("resolution must be a power of two from 16 to 4096")
        for name in ("domain", "wind_speed", "fetch_km", "depth", "gravity", "density", "gamma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("swell", "wind_direction", "surface_tension", "pinch", "amplitude_gain"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not -1 <= self.swell <= 2:
            raise ValueError("swell must be in [-1, 2]")
        if self.surface_tension < 0 or self.amplitude_gain < 0:
            raise ValueError("surface_tension and amplitude_gain cannot be negative")
        if self.spreading not in ("cosine_squared", "mitsuyasu", "hasselmann", "donelan_banner"):
            raise ValueError("unknown directional spreading model")
        if self.spectrum not in ("pm", "jonswap", "tma"):
            raise ValueError("unknown spectrum")
        if self.dispersion not in ("deep", "finite", "capillary"):
            raise ValueError("unknown dispersion")
        if self.convention not in ("paper", "legacy_2015", "houndstooth"):
            raise ValueError("unknown equation convention")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")

    def tessendorf(self):
        # The requested comparison: PM + positive cosine squared, no swell.
        # Keep dispersion, seed, gain, camera and shading matched.
        return replace(self, spectrum="pm", spreading="cosine_squared", swell=0.0)


def select_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("No GPU found. Use --device cpu explicitly for reference work.")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal is unavailable. Run outside the sandbox on Apple silicon.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch wheel.")
    if device.type not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda[:index]")
    return device


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_omega(p, *, directional=False):
    # The original master branch omitted kilometres -> metres for D only.
    fetch = p.fetch_km * (1.0 if directional and p.convention == "legacy_2015" else 1000.0)
    x = p.gravity * fetch / p.wind_speed**2
    return 2 * math.pi * 3.5 * p.gravity / p.wind_speed * x**-0.33


def dispersion_at(p, k):
    k = np.asarray(k, dtype=np.float64)
    if p.dispersion == "deep":
        omega = np.sqrt(p.gravity * k)
        derivative = np.divide(p.gravity, 2 * omega, out=np.zeros_like(k), where=omega > 0)
    else:
        hk = p.depth * k
        tanh_hk = np.tanh(hk)
        k2s = k * k * (p.surface_tension / p.density if p.dispersion == "capillary" else 0)
        omega = np.sqrt(k * (p.gravity + k2s) * tanh_hk)
        # sech^2 = 1 - tanh^2 avoids overflow for deep water / short waves.
        numerator = (p.gravity + 3*k2s)*tanh_hk + hk*(p.gravity+k2s)*(1-tanh_hk*tanh_hk)
        derivative = np.divide(numerator, 2*omega, out=np.zeros_like(k), where=omega > 0)
    return omega, derivative


def spectrum_at(p, omega):
    w = np.maximum(np.asarray(omega, dtype=np.float64), 1e-12)
    if p.spectrum == "pm":
        wm = 0.87 * p.gravity / p.wind_speed
        result = .0081 * p.gravity**2 / w**5 * np.exp(-1.291*(wm/w)**4)
    else:
        wm = peak_omega(p)
        x = p.gravity * p.fetch_km * 1000 / p.wind_speed**2
        sigma = np.where(w <= wm, .07, .09)
        sharpen = p.gamma ** np.exp(-.5*((w-wm)/(sigma*wm))**2)
        result = .076*x**-.22 * p.gravity**2 / w**5 * np.exp(-1.25*(wm/w)**4) * sharpen
        if p.spectrum == "tma":
            # Smooth depth approximation explicitly allowed in section 5.1.5,
            # and used by both original implementations.
            result *= .5 + .5*np.tanh(1.8*(w*math.sqrt(p.depth/p.gravity)-1.125))
    return np.where(np.asarray(omega) > 0, result, 0.0)


def spreading_at(p, omega, theta):
    w = np.maximum(np.asarray(omega, dtype=np.float64), 1e-12)
    theta = np.asarray(theta, dtype=np.float64)
    ratio = w / peak_omega(p, directional=True)
    coefficient = 16.0 if p.convention == "paper" else 16.1
    bias = coefficient * np.tanh(1/ratio) * max(0.0, p.swell)**2
    if p.spreading in ("hasselmann", "mitsuyasu"):
        u_over_c = p.wind_speed * peak_omega(p, directional=True) / p.gravity
        if p.spreading == "hasselmann":
            shape = np.where(ratio <= 1, 6.97*ratio**4.06,
                             9.77*ratio**(-2.33-1.45*(u_over_c-1.17)))
        else:
            shape = 11.5*u_over_c**-2.5 * np.where(ratio <= 1, ratio**5, ratio**-2.5)
        shape = shape + bias
        # Log form of eq. 34: no overflowing gamma or power-of-two factors.
        log_q = (2*shape-1)*math.log(2) - math.log(math.pi) + 2*gammaln(shape+1)-gammaln(2*shape+1)
        result = np.exp(log_q + 2*shape*np.log(np.maximum(np.abs(np.cos(theta/2)), 1e-30)))
    else:
        if p.spreading == "donelan_banner":
            beta = np.where(ratio < .95, 2.61*ratio**1.3,
                    np.where(ratio < 1.6, 2.28*ratio**-1.3,
                             10**(-.4+.8393*np.exp(-.567*np.log(np.maximum(ratio,1.6)**2)))))
            def base(angle):
                return 1 / np.cosh(np.clip(beta*angle, -350, 350))**2
        else:
            def base(angle):
                return np.where(np.abs(angle) <= math.pi/2, np.cos(angle)**2, 0.0)
        def product(angle):
            return base(angle) * np.abs(np.cos(angle/2))**(2*bias)
        if p.convention != "paper":
            # Preserve the published code's 36-segment half-circle integral.
            angles = np.linspace(-math.pi/2, math.pi/2, 37)
            denom = .5*(product(angles[0]) + product(angles[-1]))
            for angle in angles[1:-1]:
                denom = denom + product(angle)
            denom *= math.pi/36
        elif p.swell <= 0 and p.spreading == "donelan_banner":
            denom = 2*np.tanh(beta*math.pi)/beta
        elif p.swell == 0 and p.spreading == "cosine_squared":
            denom = math.pi/2
        else:
            # Gaussian quadrature for eq. 43 over the *whole* angular domain.
            extent = math.pi/2 if p.spreading == "cosine_squared" else math.pi
            angles, weights = np.polynomial.legendre.leggauss(64)
            denom = np.zeros_like(w)
            for angle, weight in zip(angles*extent, weights*extent):
                denom += weight * product(angle)
        result = product(theta) / np.maximum(denom, 1e-30)
    if p.swell < 0:
        result = (1+p.swell)*result - p.swell/(2*math.pi)  # eq. 40
    return result


def _hash_state(state):
    return (state*np.uint64(747796405) + np.uint64(2891336453)) & np.uint64(0xffffffff)


def _word(state):
    word = ((state >> ((state >> np.uint64(28)) + np.uint64(4))) ^ state)*np.uint64(277803737)
    word &= np.uint64(0xffffffff)
    return (word >> np.uint64(22)) ^ word


def random_variates(kx, ky, seed):
    """Wavenumber-addressed PCG hash, normal amplitudes, uniform phases.

    Independent of array dimensions and device. Uses the functional branch's
    PCG permutation, with explicitly specified integer truncation (section 7.1.1).
    This is a portable stream, not the platform-dependent std distributions.
    """
    ix = np.trunc(kx*10000).astype(np.int64).astype(np.uint64)
    iy = np.trunc(ky*10000).astype(np.int64).astype(np.uint64)
    state = ((ix*73856093) ^ (iy*19349663) ^ np.uint64((seed*83492791) & 0xffffffff)) & np.uint64(0xffffffff)
    values = []
    for _ in range(4):
        state = _hash_state(state)
        values.append((_word(state).astype(np.float64)+.5) / 4294967296)
    radius = np.sqrt(-2*np.log(values[0]))
    return radius*np.cos(2*np.pi*values[1]), radius*np.sin(2*np.pi*values[1]), 2*np.pi*values[2], 2*np.pi*values[3]


@dataclass(frozen=True)
class Initial_state:
    parameters: Wave_parameters
    h_positive: torch.Tensor
    h_negative: torch.Tensor
    omega: torch.Tensor
    multipliers: torch.Tensor

    @property
    def device(self):
        return self.omega.device


@dataclass(frozen=True)
class Wave_frame:
    parameters: Wave_parameters
    time: float
    displacement: torch.Tensor  # [y,x,4]: dx,dy,height,negative minimum eigenvalue
    normal: torch.Tensor        # [y,x,4]: nx,ny,nz,0


@torch.inference_mode()
def make_initial_state(parameters=Wave_parameters(), device="auto"):
    p = parameters
    device = select_device(str(device))
    n = p.resolution
    width = n//2+1
    shape = (n, width)
    pos, neg = np.empty(shape, np.complex64), np.empty(shape, np.complex64)
    omega_out = np.empty(shape, np.float32)
    multipliers = np.empty((6, *shape), np.complex64)
    dk = 2*math.pi/p.domain
    kx = np.arange(width, dtype=np.float64)[None, :] * dk
    j = np.arange(n)
    # Matches the C++ iterator, including its positive Nyquist row.
    ky_all = np.where(j <= n//2, j, j-n)*dk
    direction = math.radians(p.wind_direction)
    for row in range(0, n, 64):
        section = slice(row, min(row+64, n))
        ky = ky_all[section, None]
        k = np.hypot(kx, ky)
        safe_k = np.maximum(k, 1e-12)
        omega, derivative = dispersion_at(p, k)
        theta = np.arctan2(-ky, kx)-direction
        theta = (theta+math.pi) % (2*math.pi)-math.pi
        opposite = (theta+2*math.pi) % (2*math.pi)-math.pi
        energy = spectrum_at(p, omega)*dk*dk*derivative/safe_k
        amp_pos, amp_neg, phase_pos, phase_neg = random_variates(kx, ky, p.seed)
        direction_pos, direction_neg = spreading_at(p, omega, np.stack((theta, opposite)))
        pos[section] = amp_pos*np.sqrt(2*energy*direction_pos)*np.exp(-1j*phase_pos)
        neg[section] = amp_neg*np.sqrt(2*energy*direction_neg)*np.exp(-1j*phase_neg)
        omega_out[section] = omega
        multipliers[0, section] = 1
        multipliers[1, section] = -1j*kx/safe_k
        multipliers[2, section] = -1j*ky/safe_k
        multipliers[3, section] = kx*kx/safe_k
        multipliers[4, section] = ky*ky/safe_k
        multipliers[5, section] = kx*ky/safe_k
    pos[0,0] = neg[0,0] = 0
    multipliers[:,0,0] = 0
    def tensor(array):
        return torch.from_numpy(array).to(device)
    return Initial_state(p, tensor(pos), tensor(neg), tensor(omega_out), tensor(multipliers))


@torch.inference_mode()
def spectral_height(state, time):
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    phase = state.omega * time
    rotation = torch.complex(torch.cos(phase), torch.sin(phase))
    return state.h_positive*rotation.conj() + state.h_negative*rotation


@torch.inference_mode()
def evaluate(state, time):
    p = state.parameters
    h = spectral_height(state, time)
    # FFTW/cufft C2R are unnormalized. norm='forward' is essential: the
    # default inverse scaling would make the sea shrink with resolution.
    fields = torch.fft.irfft2(state.multipliers*h[None], s=(p.resolution, p.resolution), norm="forward")
    height, dx, dy, dxx, dyy, dxy = fields.unbind()
    jxx, jyy, jxy = 1-p.pinch*dxx, 1-p.pinch*dyy, -p.pinch*dxy
    minimum = .5*(jxx+jyy-torch.sqrt((jxx-jyy).square()+4*jxy.square()))
    displacement = torch.stack((-p.pinch*dx, -p.pinch*dy, p.amplitude_gain*height, -minimum), dim=-1)
    # ComputeNormalsWithPinching from Normals.h: central differences of
    # actual displaced points, with periodic neighbors and unwrapped spacing.
    points=displacement[...,:3]
    tangent_x=torch.roll(points,-1,1)-torch.roll(points,1,1)
    tangent_y=torch.roll(points,-1,0)-torch.roll(points,1,0)
    tangent_x[...,0]+=2*p.domain/p.resolution
    tangent_y[...,1]+=2*p.domain/p.resolution
    cross=torch.linalg.cross(tangent_x,tangent_y)
    normal_xyz=cross/torch.linalg.vector_norm(cross,dim=-1,keepdim=True).clamp_min(1e-12)
    normal=torch.cat((normal_xyz,torch.zeros_like(normal_xyz[...,:1])),dim=-1)
    return Wave_frame(p, time, displacement, normal)


@torch.inference_mode()
def texture_arrays(frame, max_resolution=None):
    """Explicit renderer boundary: two linear RGBA16F maps, no core downsample.

    Normal maps retain the full simulation resolution by default. Half-float
    conversion affects only display textures, never the numerical fields.
    """
    stride = max(1, frame.parameters.resolution // (max_resolution or frame.parameters.resolution))
    return tuple(np.ascontiguousarray(t[::stride, ::stride].to(dtype=torch.float16).cpu().numpy())
                 for t in (frame.displacement, frame.normal))
