"""Spatial trough damping: the original filter, bounded attenuation and replay."""
from dataclasses import asdict, replace
import json
from unittest.mock import patch
import math
import numpy as np
import pytest
import torch
from encino_waves.model import Wave_parameters, evaluate, spectral_height
from encino_waves.editing import make_wave_basis, state_from_basis, edit_state
from encino_waves.state_io import save_initial_state, load_initial_state
from encino_waves.trough import damp_troughs, trough_retention


@pytest.fixture(params=("cpu", "mps", "cuda"))
def basis(request):
    device = request.param
    if device == "mps" and not torch.backends.mps.is_available(): pytest.skip("Metal not visible")
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA not visible")
    return make_wave_basis(Wave_parameters(resolution=128, domain=64), device)


@pytest.mark.parametrize("amount", (.5, 1))
def test_trough_port_against_double_precision_fftw(basis, amount):
    fftw = pytest.importorskip("pyfftw.interfaces.numpy_fft")
    state = state_from_basis(basis, replace(basis.parameters,trough_damping=amount))
    p = state.parameters
    time = 12.5
    h = spectral_height(state, time)
    mult = state.multipliers
    fields = torch.fft.irfft2(mult*h[None], s=(p.resolution,)*2, norm="forward")
    # Independent double-precision reference, using the original C++ filter's
    # wavelength edges and negative minimum-eigenvalue interpolation guide.
    h_cpu = h.cpu().numpy().astype(np.complex128)
    mult_cpu = mult.cpu().numpy().astype(np.complex128)
    k = (mult_cpu[3]+mult_cpu[4]).real
    wavelength = 2*math.pi/np.maximum(k, 1e-12)
    t0 = np.clip(wavelength/p.trough_small_wavelength, 0, 1)
    t1 = np.clip((wavelength-p.trough_big_wavelength)/p.trough_soft_width, 0, 1)
    response = 1-t0*t0*(3-2*t0)+t1*t1*(3-2*t1)
    original = fftw.irfft2(mult_cpu*h_cpu[None], s=(p.resolution,)*2, norm="forward")
    filtered = fftw.irfft2(mult_cpu*(h_cpu*response)[None], s=(p.resolution,)*2, norm="forward")
    a, b, c = filtered[3:]
    guide = -1+.625*(a+b+np.sqrt((a-b)**2+4*c*c))
    t = np.clip((guide-guide.mean())/(2.2*guide.std()), 0, 1)
    weight = 1-p.trough_damping+p.trough_damping*t*t*(3-2*t)
    expected = filtered[:3]+weight*(original[:3]-filtered[:3])
    actual = damp_troughs(p, h, mult, fields).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=3e-5)
    frame = evaluate(state, time)
    xyz = np.stack((-p.pinch*actual[1], -p.pinch*actual[2], p.amplitude_gain*actual[0]), axis=-1)
    # Verify normals follow the final surface, including spatial mask gradients.
    dx, dy = np.roll(xyz,-1,1)-np.roll(xyz,1,1), np.roll(xyz,-1,0)-np.roll(xyz,1,0)
    dx[...,0] += 2*p.domain/p.resolution
    dy[...,1] += 2*p.domain/p.resolution
    normals = np.cross(dx,dy)
    normals /= np.linalg.norm(normals,axis=-1,keepdims=True)
    np.testing.assert_allclose(frame.normal[...,:3].cpu(),normals,atol=4e-6,rtol=3e-5)


def test_trough_mask_damps_troughs_with_requested_strength(basis):
    x = torch.arange(128, device=basis.device)*2*math.pi/128
    # A long-wave ridge: the crest guide grows on the compressed crest and is
    # flat on the expanded trough, as ComputeMinE in the original C++ code.
    guide = (-1+torch.cos(x).clamp_min(0))[None,:].expand(128,-1)
    for amount in (.5, 1):
        weight = trough_retention(guide, amount)
        assert float(weight.min()) >= 1-amount and float(weight.max()) <= 1
        assert float(weight[0,64]) == 1-amount
        assert float(weight[0,0]) > 1-.2*amount
        torch.testing.assert_close(trough_retention(torch.ones_like(guide),amount),torch.ones_like(guide))


@pytest.mark.parametrize("amount", (.5, 1))
def test_spatial_edits_reuse_spectrum_and_preserve_foam_source(basis, tmp_path, amount):
    state = state_from_basis(basis, replace(basis.parameters,trough_damping=amount))
    state = edit_state(basis, state, replace(state.parameters,depth=5), 10)
    frame = evaluate(state, 12)
    with patch("encino_waves.editing.state_from_basis", side_effect=AssertionError("Rebuilt spectral state")):
        off = edit_state(basis, state, replace(state.parameters,trough_damping=0), 12)
    for name in ("h_positive","h_negative","omega","multipliers","phase","phase_steps"):
        assert getattr(off,name) is getattr(state,name)
    unfiltered = evaluate(off,12)
    assert not torch.equal(frame.displacement[...,:3], unfiltered.displacement[...,:3])
    torch.testing.assert_close(frame.displacement[...,3], unfiltered.displacement[...,3], atol=0, rtol=0)
    halfway = evaluate(replace(state,parameters=replace(state.parameters,trough_damping=amount/2)),12)
    torch.testing.assert_close(halfway.displacement[...,:3],
                               .5*(frame.displacement[...,:3]+unfiltered.displacement[...,:3]),
                               atol=1e-6,rtol=2e-6)
    path = tmp_path/"trough.npz"
    save_initial_state(path,state)
    loaded = load_initial_state(path,str(basis.device))
    torch.testing.assert_close(evaluate(loaded,12).displacement,frame.displacement,atol=0,rtol=0)


def test_flat_and_low_energy_water_stays_finite(basis):
    original = state_from_basis(basis,basis.parameters)
    zero = torch.zeros_like(original.h_positive)
    state = replace(original,h_positive=zero,h_negative=zero)
    frame = evaluate(state,12)
    assert torch.isfinite(frame.normal).all()
    assert torch.count_nonzero(frame.displacement[...,:3]) == 0
    p = replace(original.parameters,wind_speed=1,fetch_km=1,depth=.25)
    frame = evaluate(state_from_basis(basis,p),12)
    assert torch.isfinite(frame.displacement).all() and torch.isfinite(frame.normal).all()


def test_saved_pre_trough_ocean_retains_its_surface(tmp_path):
    p = Wave_parameters(resolution=16,trough_damping=0)
    basis = make_wave_basis(p,"cpu")
    state = state_from_basis(basis,p)
    values = {name:value for name,value in asdict(p).items() if not name.startswith("trough_")}
    path = tmp_path/"old.npz"
    np.savez(path, version=3, parameters=json.dumps(values), phase_steps="[]",
             **{name:getattr(state,name).numpy() for name in ("h_positive","h_negative","omega","multipliers")})
    loaded = load_initial_state(path,"cpu")
    assert loaded.parameters.trough_damping == 0
    torch.testing.assert_close(evaluate(loaded,12).displacement,evaluate(state,12).displacement,atol=0,rtol=0)


@pytest.mark.parametrize("changes", ({"trough_damping":1.01},{"trough_damping":-1},
    {"trough_damping":float("nan")},{"trough_small_wavelength":0},
    {"trough_big_wavelength":.5},{"trough_soft_width":0}))
def test_trough_parameter_limits(changes):
    with pytest.raises(ValueError): Wave_parameters(**changes)
