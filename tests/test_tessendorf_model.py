"""Early Phillips equations, archived random stream, and live GPU editing."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch

from encino_waves.model import (Wave_parameters, make_initial_state, evaluate,
    legacy_gaussian_variates, legacy_random_coefficients, phillips_spectrum_at,
    spectral_height, phase_at)
from encino_waves.editing import (make_wave_basis, state_from_basis, edit_state,
    follow_parameters, restore_phase)


def test_baseline_uses_phillips_without_later_controls():
    p = Wave_parameters(swell=1, trough_damping=.8).tessendorf()
    assert (p.spectrum, p.spreading, p.dispersion) == ("phillips", "cosine_squared", "finite")
    assert p.swell == p.trough_damping == 0
    assert p.phillips_amplitude == 100  # explicit manual presentation preset
    assert p.phillips_pinch == 75
    k = np.geomspace(.002, 100, 200)
    angle = .7
    expected = np.cos(angle)**2*np.exp(-1/(k*p.wind_speed**2/p.gravity)**2)/k**4
    np.testing.assert_allclose(phillips_spectrum_at(p, k, angle), expected, rtol=2e-13)
    np.testing.assert_allclose(phillips_spectrum_at(p, k, angle+math.pi), expected, rtol=2e-13)
    other = replace(p, depth=.1, fetch_km=1, swell=2, spreading="hasselmann")
    np.testing.assert_array_equal(phillips_spectrum_at(p, k, angle), phillips_spectrum_at(other, k, angle))
    assert phillips_spectrum_at(p, 0, angle) == 0


def test_original_rand48_gaussian_sequence():
    # Independently obtained from the archived C++ implementation, with FMA
    # contraction disabled (as on the original machines). libm may differ 1 ULP.
    expected = np.array([.669526279, -.873199224, .445691675, .254597098,
                         .923034787, -.392144114, -1.29932761, -.360336542,
                         .258115619, 1.96545434, -1.39727676, -.657735646], np.float32)
    np.testing.assert_allclose(legacy_gaussian_variates(12, 12345), expected, rtol=2e-7, atol=1e-7)
    np.testing.assert_array_equal(legacy_gaussian_variates(12, 12345), legacy_gaussian_variates(9000, 12345)[:12])
    assert not np.array_equal(legacy_gaussian_variates(12, 12345), legacy_gaussian_variates(12, 12346))
    positive, negative = legacy_random_coefficients(16, 12345)
    assert positive[0, 0] == negative[0, 0] == 0
    assert negative[0, 1] == pytest.approx(complex(expected[0], expected[1]), abs=2e-7)
    assert positive[0, 1] == pytest.approx(complex(expected[2], -expected[3]), abs=2e-7)


def test_classic_complex_h0_time_evolution_and_patch_scaling():
    p = Wave_parameters(resolution=16, seed=12345, phillips_amplitude=55/16).tessendorf()
    state = make_initial_state(p, "cpu")
    k = 2*math.pi/p.domain
    magnitude = math.sqrt(.5*phillips_spectrum_at(p, k, 0))/p.domain**2
    gaussian = legacy_gaussian_variates(4, p.seed)
    h0 = complex(*gaussian[:2])*magnitude
    h0_negative = complex(*gaussian[2:])*magnitude
    omega = math.sqrt(p.gravity*k*math.tanh(k*p.depth))
    for time in (0, .25, 12):
        expected = h0*np.exp(1j*omega*time)+np.conj(h0_negative)*np.exp(-1j*omega*time)
        assert complex(spectral_height(state, time)[0, 1]) == pytest.approx(expected, abs=1e-8, rel=2e-6)


def test_depth_changes_speed_but_not_amplitude_and_gain_is_manual():
    p = Wave_parameters(resolution=32).tessendorf()
    deep = make_initial_state(p, "cpu")
    shallow = make_initial_state(replace(p, depth=.25, fetch_km=1), "cpu")
    for name in ("h_positive", "h_negative"):
        torch.testing.assert_close(getattr(deep, name), getattr(shallow, name), rtol=0, atol=0)
    assert torch.all(shallow.omega <= deep.omega)
    assert torch.any(shallow.omega < deep.omega*.5)
    tuned = make_initial_state(replace(p, phillips_amplitude=200), "cpu")
    pinched = make_initial_state(replace(p, phillips_pinch=150), "cpu")
    for time in (0, 2):
        a, b, c = evaluate(deep, time), evaluate(tuned, time), evaluate(pinched, time)
        torch.testing.assert_close(b.displacement[..., 2], a.displacement[..., 2]*2, rtol=0, atol=0)
        torch.testing.assert_close(b.displacement[..., :2], a.displacement[..., :2], rtol=0, atol=0)
        torch.testing.assert_close(b.displacement[..., 3], a.displacement[..., 3], rtol=0, atol=0)
        torch.testing.assert_close(c.displacement[..., :2], a.displacement[..., :2]*2, rtol=0, atol=0)
        torch.testing.assert_close(c.displacement[..., 2], a.displacement[..., 2], rtol=0, atol=0)
    # Both travelling terms retain independent nonzero complex Gaussian energy.
    assert torch.count_nonzero(deep.h_positive[1:, 1:])
    assert torch.count_nonzero(deep.h_negative[1:, 1:])
    assert not torch.equal(deep.h_positive, deep.h_negative)


@pytest.mark.parametrize("device", ("cpu", "mps", "cuda"))
def test_live_phillips_coefficients_match_reference_and_phase_edits(device):
    if device == "mps" and not torch.backends.mps.is_available(): pytest.skip("Metal not visible")
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA not visible")
    modern = Wave_parameters(resolution=32)
    basis = make_wave_basis(modern, device)
    for wind, depth, gain in ((17, 100, 100), (3, .25, 55/16), (80, 2, 500)):
        p = replace(modern.tessendorf(), wind_speed=wind, depth=depth, phillips_amplitude=gain)
        expected, actual = make_initial_state(p, "cpu"), state_from_basis(basis, p)
        for name in ("h_positive", "h_negative", "omega"):
            torch.testing.assert_close(getattr(actual, name).cpu(), getattr(expected, name), rtol=3e-5, atol=2e-8)
        assert torch.count_nonzero(actual.h_positive[:, 0]) == 0
    before = state_from_basis(basis, modern)
    after = edit_state(basis, before, modern.tessendorf(), 12.5)
    for function in (torch.sin, torch.cos):
        torch.testing.assert_close(function(phase_at(before, 12.5)), function(phase_at(after, 12.5)), rtol=2e-6, atol=2e-6)
    replay = restore_phase(basis, state_from_basis(basis, after.parameters), after.phase_steps)
    torch.testing.assert_close(after.phase, replay.phase, rtol=0, atol=0)
    smooth = follow_parameters(modern, replace(modern.tessendorf(), phillips_amplitude=1000), 1/60)
    assert 100 < smooth.phillips_amplitude < 1000
    assert smooth.trough_damping == smooth.swell == 0
    tuned = edit_state(basis, after, replace(after.parameters, phillips_amplitude=500, phillips_pinch=125), 12.5)
    for name in ("h_positive", "h_negative", "omega", "phase", "phase_steps"):
        assert getattr(tuned, name) is getattr(after, name)


@pytest.mark.parametrize("gain", (-1, float("inf"), float("nan")))
def test_invalid_phillips_gain_is_rejected(gain):
    with pytest.raises(ValueError):
        Wave_parameters(phillips_amplitude=gain)


@pytest.mark.parametrize("pinch", (float("inf"), float("nan")))
def test_invalid_phillips_pinch_is_rejected(pinch):
    with pytest.raises(ValueError):
        Wave_parameters(phillips_pinch=pinch)
