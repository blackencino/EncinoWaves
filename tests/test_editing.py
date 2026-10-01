"""Post-seed conformance, live phase continuity, and replay at export resolution."""
from dataclasses import replace
import pytest
import torch
from encino_waves.model import Wave_parameters, make_initial_state, phase_at, evaluate
from encino_waves.editing import (make_wave_basis, state_from_basis, preserve_phase,
                                  restore_phase, follow_parameters, make_edited_state)
from encino_waves.state_io import save_initial_state, load_initial_state


@pytest.fixture(params=("cpu", "mps", "cuda"))
def basis(request):
    device = request.param
    if device == "mps" and not torch.backends.mps.is_available(): pytest.skip("Metal not visible")
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA not visible")
    return make_wave_basis(Wave_parameters(resolution=32), device)


@pytest.mark.parametrize("spreading", ("donelan_banner", "hasselmann", "mitsuyasu", "cosine_squared"))
def test_gpu_post_seed_matches_double_precision_reference(basis, spreading):
    settings = (
        {}, {"swell": -1}, {"swell": 2, "depth": .25},
        {"wind_speed": 3, "fetch_km": 2, "swell": 0, "depth": 5},
        {"wind_speed": 80, "fetch_km": 1250, "wind_direction": 179},
        {"wind_speed": 500, "fetch_km": 5000, "depth": 1000, "swell": 2},
        {"spectrum": "pm", "dispersion": "deep", "swell": 0},
        {"spectrum": "jonswap", "dispersion": "finite", "depth": 4},
        {"convention": "legacy_2015", "depth": 20, "swell": .8},
        {"convention": "houndstooth", "depth": 20, "swell": .8},
    )
    for changes in settings:
        p = replace(basis.parameters, spreading=spreading, **changes)
        reference = make_initial_state(p, "cpu")
        actual = state_from_basis(basis, p)
        for name in ("h_positive", "h_negative", "omega"):
            expected = getattr(reference, name)
            value = getattr(actual, name).cpu()
            assert torch.isfinite(value).all(), (p, name)
            # Spectral bins with almost no energy need an absolute floor.
            scale = float(torch.max(torch.abs(expected)))
            torch.testing.assert_close(value, expected, atol=max(1e-12, scale*2e-5), rtol=8e-5)


def test_wind_edits_reuse_noise_and_leave_phase_unchanged(basis):
    a = state_from_basis(basis, basis.parameters)
    b = state_from_basis(basis, replace(basis.parameters, wind_speed=29, fetch_km=850))
    b = preserve_phase(a, b, 137)
    torch.testing.assert_close(phase_at(a, 137), phase_at(b, 137), rtol=0, atol=0)
    assert b.phase is None and not b.phase_steps
    assert a.multipliers is b.multipliers is basis.multipliers
    for coefficients, noise in ((b.h_positive, basis.noise_positive), (b.h_negative, basis.noise_negative)):
        populated = torch.abs(coefficients) > 1e-12
        ratio = coefficients[populated]/noise[populated]
        assert torch.all(ratio.real >= 0)
        assert float(torch.max(torch.abs(ratio.imag))) < 1e-6


def test_depth_edits_preserve_phase_then_use_new_speed(basis):
    a = state_from_basis(basis, basis.parameters)
    target = state_from_basis(basis, replace(basis.parameters, depth=3))
    t = 117.25
    b = preserve_phase(a, target, t)
    for function in (torch.sin, torch.cos):
        torch.testing.assert_close(function(phase_at(a,t)), function(phase_at(b,t)), atol=4e-5, rtol=2e-5)
    dt = .0625
    torch.testing.assert_close(phase_at(b,t+dt)-phase_at(b,t), b.omega*dt, atol=5e-7, rtol=2e-5)
    assert a.phase is None
    assert len(b.phase_steps) == 1
    assert b.parameters == target.parameters
    torch.testing.assert_close(b.h_positive, target.h_positive, atol=0, rtol=0)


def test_depth_does_not_change_phase_in_deep_water_mode(basis):
    p = replace(basis.parameters,dispersion="deep")
    a = state_from_basis(basis,p)
    b = preserve_phase(a,state_from_basis(basis,replace(p,depth=2)),137)
    assert b.phase is None and not b.phase_steps
    torch.testing.assert_close(phase_at(a,137),phase_at(b,137),atol=0,rtol=0)


def test_edited_state_replay_and_snapshot_are_lossless(basis, tmp_path):
    state = state_from_basis(basis, basis.parameters)
    for t, depth in ((10,30), (10.1,15), (10.2,7), (10.2,5), (10.3,12), (3,20)):
        target = state_from_basis(basis, replace(state.parameters, depth=depth))
        state = preserve_phase(state, target, t)
    assert len(state.phase_steps) == 5  # Paused edits coalesce.
    restored = restore_phase(basis, state_from_basis(basis,state.parameters), state.phase_steps)
    torch.testing.assert_close(restored.phase, state.phase, atol=0, rtol=0)
    assert restored.phase_steps == state.phase_steps
    a = evaluate(state,12.5)
    evaluate(state,-10)
    b = evaluate(state,12.5)
    torch.testing.assert_close(a.displacement,b.displacement,atol=0,rtol=0)
    path = tmp_path/"edited.npz"
    save_initial_state(path,state)
    loaded = load_initial_state(path,str(basis.device))
    assert loaded.phase_steps == state.phase_steps
    torch.testing.assert_close(evaluate(loaded,12.5).displacement,a.displacement,atol=0,rtol=0)
    larger = make_edited_state(replace(state.parameters,resolution=64),str(basis.device),state.phase_steps)
    torch.testing.assert_close(larger.phase[:16,:16],state.phase[:16,:16],atol=0,rtol=0)
    torch.testing.assert_close(larger.phase[-15:,:16],state.phase[-15:,:16],atol=0,rtol=0)


def test_grid_changes_require_a_new_basis(basis):
    for changes in ({"domain": 1000}, {"resolution":64}, {"seed":123}):
        with pytest.raises(ValueError,match="new wave basis"):
            state_from_basis(basis,replace(basis.parameters,**changes))


def test_control_smoothing_settles_exactly_without_overshoot():
    current = Wave_parameters(wind_speed=5,depth=100,wind_direction=179)
    target = replace(current,wind_speed=30,depth=2,wind_direction=-179)
    for i in range(200):
        current = follow_parameters(current,target,1/60)
        assert 5 <= current.wind_speed <= 30 and 2 <= current.depth <= 100
        assert abs((current.wind_direction-179+180)%360-180) <= 2.00001
    assert current == target
