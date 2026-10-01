# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Periodic aeration history, separate from the directly synthesized ocean.

All state is explicit and immutable by contract, including its GPU tensors.
RGB stores surface foam, shallow bubbles and deeper bubbles in ocean space.
This is an artist-controlled appearance simulation, not part of the 2015 spectra.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
import numpy as np
import torch
import torch.nn.functional as functional
from .model import Wave_parameters, select_device, evaluate


@dataclass(frozen=True)
class Foam_parameters:
    enabled: bool = True
    resolution: int = 512
    emission: float = 1.2             # aeration density per second
    crest_start: float = .5           # original normalized crest-map threshold
    crest_width: float = .6
    surface_half_life: float = 3.0    # seconds
    shallow_half_life: float = 6.0
    deep_half_life: float = 12.0
    diffusion: float = .12            # surface m^2/s; shallow/deep spread faster
    exchange: float = .18             # shallow -> deep, per second
    breakup: float = 1.0
    noise_scale: float = 24.0         # metres, rounded to a periodic cell count
    noise_speed: float = .12          # radians/second

    def __post_init__(self):
        n = self.resolution
        if not isinstance(n, int) or n < 16 or n > 2048 or n & (n-1):
            raise ValueError("foam resolution must be a power of two from 16 to 2048")
        for name in ("emission", "diffusion", "exchange", "noise_speed"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("crest_width", "surface_half_life", "shallow_half_life", "deep_half_life", "noise_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.crest_start) or not 0 <= self.breakup <= 1:
            raise ValueError("crest threshold must be finite and breakup must be in [0, 1]")


@dataclass(frozen=True)
class Foam_basis:
    domain: float
    seed: int
    noise_scale: float
    laplacian: torch.Tensor           # [N,N/2+1], periodic discrete heat operator
    noise_cos: torch.Tensor          # [4,N,N], smooth, periodic fractal octaves
    noise_sin: torch.Tensor


@dataclass(frozen=True)
class Foam_state:
    parameters: Foam_parameters
    reference_parameters: Wave_parameters
    time: float
    density: torch.Tensor            # [3,N,N], nonnegative optical densities
    basis: Foam_basis
    crest_gain: float = 1.0
    crest_bias: float = 0.0

    @property
    def device(self):
        return self.density.device


def _periodic_noise(rng, n, cells):
    """Quintic value noise, including interpolation across both tile seams."""
    lattice = rng.uniform(-1, 1, (cells, cells)).astype(np.float32)
    coordinate = np.arange(n, dtype=np.float32)*cells/n
    lower = np.floor(coordinate).astype(np.int32)
    upper = (lower+1) % cells
    f = coordinate-lower
    w = f*f*f*(f*(f*6-15)+10)
    row0 = lattice[lower[:, None], lower[None, :]]*(1-w) + lattice[lower[:, None], upper[None, :]]*w
    row1 = lattice[upper[:, None], lower[None, :]]*(1-w) + lattice[upper[:, None], upper[None, :]]*w
    return (row0*(1-w[:, None])+row1*w[:, None]).astype(np.float32)


@torch.inference_mode()
def make_foam_state(waves, time, parameters=Foam_parameters(), device="auto"):
    if not math.isfinite(time): raise ValueError("foam time must be finite")
    device = select_device(str(device))
    n = parameters.resolution
    rng = np.random.Generator(np.random.PCG64(waves.seed ^ 0x4f414d))
    base_cells = max(1, round(waves.domain/parameters.noise_scale))
    cosine, sine = [], []
    for octave in range(4):
        cells = min(n//2, base_cells*2**octave)
        weight = .5**octave / 1.875
        cosine.append(_periodic_noise(rng, n, cells)*weight)
        sine.append(_periodic_noise(rng, n, cells)*weight)
    # exp(-D*dt*L) is the exact periodic discrete diffusion solution. Unlike an
    # explicit blur stencil, it stays stable for small patches and large maps.
    x = np.sin(np.pi*np.arange(n//2+1)/n)**2
    y = np.sin(np.pi*np.arange(n)/n)**2
    laplacian = 4*(y[:, None]+x[None, :])/(waves.domain/n)**2
    def tensor(array):
        return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)
    basis = Foam_basis(waves.domain, waves.seed, parameters.noise_scale,
                       tensor(laplacian), tensor(cosine), tensor(sine))
    return Foam_state(parameters, waves, time, torch.zeros((3,n,n), device=device), basis)


def foam_reset_reason(state, waves, parameters):
    """Compare to the sea at the last reset, so many small edits can add up."""
    if state is None: return "initialization"
    a, b = state.reference_parameters, waves
    if (a.domain, a.seed) != (b.domain, b.seed): return "ocean patch or seed changed"
    if state.density.shape[-1] != parameters.resolution: return "foam resolution changed"
    if state.basis.noise_scale != parameters.noise_scale: return "foam pattern scale changed"
    for name in ("spectrum", "spreading", "dispersion", "convention", "gravity", "surface_tension", "density", "gamma"):
        if getattr(a, name) != getattr(b, name): return "wave model changed"
    for name, ratio in (("wind_speed",1.35), ("fetch_km",2.0), ("depth",2.0)):
        if max(getattr(a,name)/getattr(b,name), getattr(b,name)/getattr(a,name)) > ratio:
            return "sea conditions changed"
    if abs(a.swell-b.swell) > .35 or abs(a.pinch-b.pinch) > .3:
        return "wave shape changed"
    # Mesh orientation and wave resolution do not invalidate ocean-space foam.
    return None


@torch.inference_mode()
def emission_mask(frame, parameters, crest_gain=1.0, crest_bias=0.0):
    # Match the actual crest texture used to calibrate the original shader.
    # In a nearly flat sea, normalizing unquantized residuals by the half-float
    # map's zero variance would otherwise manufacture widespread emission.
    crest = frame.displacement[...,3].to(torch.float16).to(torch.float32)
    crest = crest*crest_gain+crest_bias
    mask = ((crest-parameters.crest_start)/parameters.crest_width).clamp(0,1)
    mask = mask*mask*(3-2*mask)
    n = parameters.resolution
    # Threshold before area filtering so narrow crests still emit at lower foam
    # resolutions. The foam grid is independent of the exported wave grid.
    if mask.shape[0] != n:
        mode = "area" if mask.shape[0] > n else "nearest"
        mask = functional.interpolate(mask[None,None], size=(n,n), mode=mode)[0,0]
    return mask


@torch.inference_mode()
def fractal_mask(state, time, parameters):
    rates = torch.tensor((1.0,1.37,1.91,2.63), device=state.device)
    phase = rates*(time*parameters.noise_speed)
    noise = (state.basis.noise_cos*torch.cos(phase)[:,None,None] +
             state.basis.noise_sin*torch.sin(phase)[:,None,None]).sum(dim=0)
    broken = ((noise+.05)/.35).clamp(0,1)
    broken = broken*broken*(3-2*broken)
    return (1-parameters.breakup)+parameters.breakup*broken


@torch.inference_mode()
def diffuse_and_decay(density, basis, parameters, dt):
    if dt == 0: return density
    rates = torch.tensor((1.0,3.0,7.0), device=density.device)*parameters.diffusion
    attenuation = torch.exp(-dt*rates[:,None,None]*basis.laplacian)
    blurred = torch.fft.irfft2(torch.fft.rfft2(density)*attenuation, s=density.shape[-2:]).clamp_min(0)
    surface, shallow, deep = blurred.unbind()
    a = math.log(2)/parameters.surface_half_life
    b = math.log(2)/parameters.shallow_half_life+parameters.exchange
    c = math.log(2)/parameters.deep_half_life
    eb, ec = math.exp(-b*dt), math.exp(-c*dt)
    # Exact shallow -> deep exchange with independent decay. The limiting
    # expression handles equal rates without cancellation or a 0/0.
    transferred = parameters.exchange*(dt*ec if abs(b-c)*dt < 1e-7 else (ec-eb)/(b-c))
    return torch.stack((surface*math.exp(-a*dt), shallow*eb, deep*ec+shallow*transferred))


@torch.inference_mode()
def step_foam(state, frame, parameters=None):
    """Advance from an explicit previous value; do not alter any input tensor.

    The frame supplies the source held over this short interval. Blur, fade and
    exchange happen before deposition. Rates are in seconds, never per frame.
    Callers reset on time discontinuities instead of inventing missing history.
    """
    p = parameters or state.parameters
    dt = frame.time-state.time
    if not math.isfinite(dt) or dt < 0 or dt > .25+1e-8:
        raise ValueError("foam needs a forward step of at most 0.25 seconds")
    if frame.displacement.device != state.device:
        raise ValueError("waves and foam must be on the same device")
    if foam_reset_reason(state, frame.parameters, p):
        raise ValueError("foam is incompatible with the sea; reset it")
    if dt <= 1e-10:
        return state if p == state.parameters else replace(state, parameters=p)
    aged = diffuse_and_decay(state.density, state.basis, p, dt)
    emission = emission_mask(frame,p,state.crest_gain,state.crest_bias)*fractal_mask(state,state.time+dt*.5,p)*p.emission*dt
    deposited = torch.stack((emission*.8, emission*.2, torch.zeros_like(emission)))
    return replace(state, parameters=p, time=frame.time, density=aged+deposited)


def update_foam(state, frame, parameters=Foam_parameters(), crest_gain=1.0, crest_bias=0.0):
    """Viewer policy: reset incompatible history or a discontinuous timeline."""
    if not parameters.enabled: return None
    if (foam_reset_reason(state, frame.parameters, parameters) or
            frame.time < state.time-1e-9 or frame.time > state.time+.25+1e-8):
        return replace(make_foam_state(frame.parameters, frame.time, parameters, frame.displacement.device),
                       crest_gain=crest_gain,crest_bias=crest_bias)
    return step_foam(state, frame, parameters)


def save_foam(path, state):
    """Lossless portable checkpoint; the inexpensive noise basis is regenerated."""
    np.savez_compressed(path, version=1, time=state.time,
        parameters=json.dumps(asdict(state.parameters)),
        waves=json.dumps(asdict(state.reference_parameters)),
        density=state.density.cpu().numpy(),crest_gain=state.crest_gain,crest_bias=state.crest_bias)


def load_foam(path, device="auto"):
    with np.load(path, allow_pickle=False) as data:
        if int(data["version"]) != 1: raise ValueError("Unsupported foam snapshot version")
        p = Foam_parameters(**json.loads(str(data["parameters"])))
        waves = Wave_parameters(**json.loads(str(data["waves"])))
        density = data["density"]
        if (density.shape != (3,p.resolution,p.resolution) or density.dtype != np.float32 or
                not np.isfinite(density).all() or np.any(density < 0)):
            raise ValueError("Invalid foam density in snapshot")
        state = make_foam_state(waves, float(data["time"]), p, device)
        gain,bias=float(data["crest_gain"]),float(data["crest_bias"])
        if not math.isfinite(gain) or gain<=0 or not math.isfinite(bias):
            raise ValueError("Invalid foam crest normalization")
        return replace(state, density=torch.from_numpy(density.copy()).to(state.device),crest_gain=gain,crest_bias=bias)


def prepare_foam(waves, time, parameters, snapshot=None, preroll=6.0):
    """Load a shot's history, or build it at 30 Hz before its first output frame."""
    if not parameters.enabled: return None
    if not math.isfinite(preroll) or not 0 <= preroll <= 60:
        raise ValueError("foam preroll must be between 0 and 60 seconds")
    if snapshot:
        state = load_foam(snapshot,waves.device)
        if abs(state.time-time) < 1e-8 and not foam_reset_reason(state,waves.parameters,parameters):
            return replace(state, parameters=parameters)
    state = make_foam_state(waves.parameters,time-preroll,parameters,waves.device)
    # Same crest source as the original shader, whose normalization is sampled
    # once from the displayed half-float map, then held for this history.
    crest=evaluate(waves,time).displacement[...,3].cpu().numpy().astype(np.float16).astype(np.float32)
    gain=1/max(1e-8,2*float(np.std(crest)))
    state=replace(state,crest_gain=gain,crest_bias=-float(np.mean(crest))*gain)
    count = math.ceil(preroll*30)
    for i in range(count):
        t = time-preroll + (i+1)*preroll/count
        state = step_foam(state,evaluate(waves,t),parameters)
    return state


def advance_foam_to(state, waves, frame, parameters):
    """Offline time stepping, including low-FPS output with long frame gaps."""
    if not parameters.enabled: return None
    if state is None: return update_foam(None,frame,parameters)
    dt = frame.time-state.time
    if dt < 0: raise ValueError("movie foam history must advance forward")
    count = max(1,math.ceil(dt/.05-1e-8))
    start = state.time
    for i in range(count):
        source = frame if i == count-1 else evaluate(waves,start+(i+1)*dt/count)
        state = step_foam(state,source,parameters)
    return state
