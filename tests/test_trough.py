"""Spatial trough smoothing: localization, ripple scale, unchanged wave state."""
from dataclasses import asdict, replace
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import torch

from encino_waves.editing import make_wave_basis, state_from_basis, edit_state
from encino_waves.model import Wave_parameters, evaluate
from encino_waves.state_io import save_initial_state, load_initial_state
from encino_waves.trough import damp_troughs, smooth_height, trough_retention


@pytest.fixture(params=("cpu", "mps", "cuda"))
def basis(request):
    device = request.param
    if device == "mps" and not torch.backends.mps.is_available(): pytest.skip("Metal not visible")
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA not visible")
    return make_wave_basis(Wave_parameters(resolution=128, domain=16), device)


def analytic_fields(n=512, domain=16., dtype=torch.float64):
    x = torch.arange(n, dtype=dtype)*domain/n
    broad = torch.cos(2*math.pi*x/8)
    ripple = .02*torch.cos(2*math.pi*x/.25+.7)
    height = (broad+ripple)[None].expand(n, -1)
    # Arbitrary horizontal/derivative fields must pass through untouched.
    fields = torch.stack((height, height*2, height*(-3), height*4, height*5, height*6))
    return fields, broad, ripple


@pytest.mark.parametrize("amount", (.5, 1.))
def test_height_mask_only_wets_the_trough_half(amount):
    x = torch.arange(512, dtype=torch.float64)*2*math.pi/512
    height = torch.cos(x)
    retention = trough_retention(height, amount)
    assert retention.min() >= 1-amount
    assert retention.max() <= 1
    torch.testing.assert_close(retention[height >= height.mean()], torch.ones_like(retention[height >= height.mean()]), rtol=0, atol=0)
    assert float(retention[256]) == pytest.approx(1-amount, abs=1e-12)
    torch.testing.assert_close(trough_retention(torch.ones_like(height), amount), torch.ones_like(height), rtol=0, atol=0)


@pytest.mark.parametrize("amount", (.5, 1.))
def test_spatial_filter_quiets_ripples_and_preserves_broad_character(amount):
    p = Wave_parameters(resolution=512, domain=16., trough_damping=amount)
    fields, broad, ripple = analytic_fields()
    smoothed = smooth_height(fields[0], p.domain/p.resolution, p.trough_smoothing_length)
    result = damp_troughs(p, fields)
    crest = smoothed >= smoothed.mean()
    torch.testing.assert_close(result[0][crest], fields[0][crest], rtol=0, atol=0)
    torch.testing.assert_close(result[1:], fields[1:3], rtol=0, atol=0)
    broad_smoothed = smooth_height(broad[None].expand(512, -1), p.domain/p.resolution, p.trough_smoothing_length)
    trough = broad < -.95
    ratio = ((result[0, 0, trough]-broad_smoothed[0, trough]).square().mean().sqrt()
             /ripple[trough].square().mean().sqrt())
    assert ratio < 1-.90*amount
    original = torch.fft.rfft(fields[0, 0], norm="forward")
    changed = torch.fft.rfft(result[0, 0], norm="forward")
    broad_ratio = changed[2]/original[2]  # The existing 8 m wave.
    assert abs(abs(broad_ratio)-1) < .002*amount
    assert abs(torch.angle(broad_ratio)) < 1e-4
    assert abs(float(result[0].mean())) < .001*amount


def test_gaussian_uses_world_metres_and_preserves_periodicity():
    p = Wave_parameters(resolution=512, domain=16.)
    fields, _, _ = analytic_fields()
    smoothed = smooth_height(fields[0], p.domain/p.resolution, .1)
    before = torch.fft.rfft(fields[0, 0], norm="forward")
    after = torch.fft.rfft(smoothed[0], norm="forward")
    # Independent continuous Gaussian transfer, allowing the stated 3-sigma
    # truncation. Broad waves and 25 cm detail use the same world sigma.
    for wavelength in (8., .25):
        mode = round(p.domain/wavelength)
        expected = math.exp(-.5*(2*math.pi*.1/wavelength)**2)
        assert float(abs(after[mode]/before[mode])) == pytest.approx(expected, abs=.002)
    shifted = smooth_height(torch.roll(fields[0], (37, 19), (0, 1)), p.domain/p.resolution, .1)
    torch.testing.assert_close(shifted, torch.roll(smoothed, (37, 19), (0, 1)), rtol=0, atol=1e-12)
    assert float(smoothed.mean()) == pytest.approx(float(fields[0].mean()), abs=1e-12)


def test_spatial_filter_converges_at_the_same_world_scale():
    coarse, _, _ = analytic_fields(256)
    fine, _, _ = analytic_fields(512)
    a = damp_troughs(Wave_parameters(resolution=256, domain=16, trough_damping=1), coarse)
    b = damp_troughs(Wave_parameters(resolution=512, domain=16, trough_damping=1), fine)
    torch.testing.assert_close(a[0], b[0, ::2, ::2], rtol=0, atol=4e-5)
    # Enlarging the periodic domain while retaining spacing and the same waves
    # must not change the material's physical smoothing length.
    tiled = coarse.repeat(1, 2, 2)
    c = damp_troughs(Wave_parameters(resolution=512, domain=32, trough_damping=1), tiled)
    torch.testing.assert_close(c[0, :256, :256], a[0], rtol=0, atol=1e-12)


def test_extremely_fine_grid_uses_the_documented_bounded_sigma():
    height = torch.zeros((64, 64), dtype=torch.float64)
    height[32, 32] = 1
    # 16 samples = 3 sigma maximum support. Very fine grids cap smoothing,
    # rather than silently blurring the whole domain with an enormous kernel.
    bounded = smooth_height(height, .001, .1)
    explicit = smooth_height(height, .001, 16*.001/3)
    torch.testing.assert_close(bounded, explicit, rtol=0, atol=0)
    assert float(bounded.sum()) == pytest.approx(1., abs=1e-12)
    assert torch.count_nonzero(bounded[:16]) == 0


@pytest.mark.parametrize("amount,domain", [(0., 16.), (.5, 512.), (1., 512.)])
def test_disabled_or_unresolved_smoothing_is_an_exact_no_op(amount, domain):
    p = Wave_parameters(resolution=512, domain=domain, trough_damping=amount)
    fields, _, _ = analytic_fields()
    with patch("encino_waves.trough.smooth_height", side_effect=AssertionError("Unexpected spatial filtering")):
        result = damp_troughs(p, fields)
    assert result.data_ptr() == fields.data_ptr()
    torch.testing.assert_close(result, fields[:3], rtol=0, atol=0)


def test_filter_adds_no_fft_and_leaves_flat_water_unchanged():
    p = Wave_parameters(resolution=32, domain=2, trough_damping=1)
    fields = torch.ones((6, 32, 32))
    with patch("torch.fft.irfft2", side_effect=AssertionError("Extra inverse FFT")), \
         patch("torch.fft.rfft2", side_effect=AssertionError("Extra forward FFT")):
        result = damp_troughs(p, fields)
    torch.testing.assert_close(result, fields[:3], rtol=0, atol=0)


@pytest.mark.parametrize("amount", (.5, 1.))
def test_spatial_edits_reuse_spectrum_xy_and_crest_and_recompute_normals(basis, tmp_path, amount):
    state = state_from_basis(basis, replace(basis.parameters, trough_damping=amount))
    state = edit_state(basis, state, replace(state.parameters, depth=5), 10)
    with patch("torch.fft.irfft2", wraps=torch.fft.irfft2) as inverse:
        frame = evaluate(state, 12)
    assert inverse.call_count == 1  # Only the original six-field batched FFT.
    assert inverse.call_args.args[0].shape[0] == 6
    with patch("encino_waves.editing.state_from_basis", side_effect=AssertionError("Rebuilt spectrum")):
        off = edit_state(basis, state, replace(state.parameters, trough_damping=0), 12)
        resized = edit_state(basis, state, replace(state.parameters, trough_smoothing_length=.12), 12)
    for name in ("h_positive", "h_negative", "omega", "multipliers", "phase", "phase_steps"):
        assert getattr(off, name) is getattr(state, name)
        assert getattr(resized, name) is getattr(state, name)
    unfiltered = evaluate(off, 12)
    assert not torch.equal(frame.displacement[..., 2], unfiltered.displacement[..., 2])
    torch.testing.assert_close(frame.displacement[..., [0, 1, 3]], unfiltered.displacement[..., [0, 1, 3]], rtol=0, atol=0)
    halfway = evaluate(replace(state, parameters=replace(state.parameters, trough_damping=amount/2)), 12)
    torch.testing.assert_close(halfway.displacement[..., 2], .5*(frame.displacement[..., 2]+unfiltered.displacement[..., 2]), atol=1e-6, rtol=2e-6)
    xyz = frame.displacement[..., :3].cpu().numpy()
    dx, dy = np.roll(xyz, -1, 1)-np.roll(xyz, 1, 1), np.roll(xyz, -1, 0)-np.roll(xyz, 1, 0)
    dx[..., 0] += 2*state.parameters.domain/state.parameters.resolution
    dy[..., 1] += 2*state.parameters.domain/state.parameters.resolution
    normals = np.cross(dx, dy)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    np.testing.assert_allclose(frame.normal[..., :3].cpu(), normals, atol=4e-6, rtol=3e-5)
    path = tmp_path/"trough.npz"
    save_initial_state(path, state)
    torch.testing.assert_close(evaluate(load_initial_state(path, str(basis.device)), 12).displacement, frame.displacement, rtol=0, atol=0)


def test_legacy_bands_migrate_explicitly_and_new_custom_sigma_roundtrips():
    old = {"trough_damping": .75, "trough_small_wavelength": 1.,
           "trough_big_wavelength": 4., "trough_soft_width": 2.}
    migrated = Wave_parameters.from_dict(old)
    assert migrated.trough_damping == .75
    assert migrated.trough_smoothing_length == .1
    assert migrated.trough_filter_revision == 3
    custom = Wave_parameters.from_dict(dict(old, trough_big_wavelength=.6, trough_soft_width=.2))
    assert custom.trough_smoothing_length == pytest.approx(.16)
    second_revision = Wave_parameters.from_dict(dict(old, trough_small_wavelength=.02,
        trough_big_wavelength=.4, trough_soft_width=.1, trough_filter_revision=2))
    assert second_revision.trough_smoothing_length == .1
    current = Wave_parameters(trough_damping=1, trough_smoothing_length=.173)
    assert Wave_parameters.from_dict(asdict(current)) == current
    assert Wave_parameters.from_dict({}).trough_damping == 0
    assert old["trough_big_wavelength"] == 4.  # Caller input was not mutated.


def test_saved_pre_trough_ocean_retains_its_surface(tmp_path):
    p = Wave_parameters(resolution=16, trough_damping=0)
    basis = make_wave_basis(p, "cpu")
    state = state_from_basis(basis, p)
    values = {name:value for name,value in asdict(p).items() if not name.startswith("trough_")}
    path = tmp_path/"old.npz"
    np.savez(path, version=3, parameters=json.dumps(values), phase_steps="[]",
             **{name:getattr(state, name).numpy() for name in ("h_positive", "h_negative", "omega", "multipliers")})
    loaded = load_initial_state(path, "cpu")
    assert loaded.parameters.trough_damping == 0
    torch.testing.assert_close(evaluate(loaded, 12).displacement, evaluate(state, 12).displacement, rtol=0, atol=0)


@pytest.mark.parametrize("changes", ({"trough_damping":1.01}, {"trough_damping":-1},
    {"trough_damping":float("nan")}, {"trough_smoothing_length":0},
    {"trough_smoothing_length":float("nan")}, {"trough_filter_revision":2}))
def test_trough_parameter_limits(changes):
    with pytest.raises(ValueError): Wave_parameters(**changes)


def test_cpp_spatial_helper_matches_python(tmp_path):
    """Compile production Propagation.h, substituting only serial scheduling.

    The executable also checks every padded seam, exact no-ops, flat water, and
    default parameters. This does not stand in for building the legacy viewer.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path/"trough_oracle"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root/"src"),
                    str(root/"tests/reference/trough_spatial_oracle.cpp"),
                    "-o", str(executable)], check=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    samples = np.loadtxt(StringIO(result.stdout))
    assert np.isfinite(samples).all()
    for bits, n, domain, amount in np.unique(samples[:, :4], axis=0):
        selected = samples[np.all(samples[:, :4] == (bits, n, domain, amount), axis=1)]
        n = int(n)
        coordinates = np.arange(n, dtype=np.float64)*domain/n
        wx, wy = coordinates[None, :], coordinates[:, None]
        height = (np.cos(2*math.pi*wx/4)+.3*np.cos(2*math.pi*wy/8)
                  +.02*np.cos(2*math.pi*wx/.25+.7)+.01*np.sin(2*math.pi*wy/.25))
        dtype = torch.float32 if bits == 32 else torch.float64
        fields = torch.from_numpy(height).to(dtype).expand(3, -1, -1)
        parameters = Wave_parameters(resolution=n, domain=domain, trough_damping=amount)
        expected = damp_troughs(parameters, fields)[0].numpy()
        y, x = selected[:, 4:6].astype(np.int64).T
        tolerance = 5e-7 if bits == 32 else 2e-12
        np.testing.assert_allclose(selected[:, 6], fields[0, y, x], rtol=0, atol=tolerance)
        np.testing.assert_allclose(selected[:, 7], expected[y, x], rtol=0, atol=tolerance)
