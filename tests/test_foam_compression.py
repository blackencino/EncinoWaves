"""Compression birth, physical trends, and unchanged legacy history."""
from dataclasses import asdict, replace
import numpy as np
import pytest
import torch
from encino_waves.model import Wave_parameters, Wave_frame, make_initial_state, evaluate
from encino_waves.foam import (Foam_parameters, emission_mask, make_foam_state,
    step_foam, update_foam, foam_reset_reason, save_foam, load_foam)


def stretch_frame(stretch, time=.1):
    stretch = torch.as_tensor(stretch, dtype=torch.float32)
    if stretch.ndim == 0:
        stretch = stretch.expand(16, 16)
    elif stretch.ndim == 1:
        stretch = stretch[None, :].expand(stretch.numel(), -1)
    values = torch.zeros((*stretch.shape, 4))
    values[..., 3] = -stretch
    return Wave_frame(Wave_parameters(resolution=stretch.shape[0]), time,
                      values, torch.zeros_like(values))


def test_compression_threshold_has_absolute_stretch_units():
    p = Foam_parameters(resolution=16, emission_model="compression",
                        compression_threshold=.75, compression_width=.25)
    stretch = torch.tensor([1., .875, .75, .6875, .625, .5625, .5, -.25]*2)
    frame = stretch_frame(stretch)
    before = frame.displacement.clone()
    mask = emission_mask(frame, p)
    expected = torch.tensor([0., 0., 0., .15625, .5, .84375, 1., 1.]*2)
    torch.testing.assert_close(mask[0], expected, atol=0, rtol=0)
    # Display normalization must have no influence on this physical coordinate.
    torch.testing.assert_close(emission_mask(frame, p, 1e8, -1e6), mask, atol=0, rtol=0)
    torch.testing.assert_close(frame.displacement, before, atol=0, rtol=0)


def test_compression_threshold_precedes_area_filtering():
    p = Foam_parameters(resolution=16, emission_model="compression")
    stretch = torch.ones(32, 32)
    stretch[0, 0] = p.compression_threshold-p.compression_width
    mask = emission_mask(stretch_frame(stretch), p)
    assert mask[0, 0] == .25  # One fully compressed source in a 2x2 footprint.
    assert torch.count_nonzero(mask) == 1


def test_absolute_source_grows_with_wind_and_vanishes_without_pinching():
    p = Foam_parameters(resolution=128, emission_model="compression")
    sources = []
    for wind in (3., 10., 25.):
        waves = make_initial_state(Wave_parameters(resolution=256, wind_speed=wind), "cpu")
        sources.append(np.mean([float(emission_mask(evaluate(waves, t), p).mean())
                                for t in (0., 4., 12.)]))
    # Compare a time ensemble over a fixed seed and spectral bandwidth, not an
    # exact whitecap count: stronger wind should retain its stronger compression.
    assert 0 <= sources[0] < sources[1] < sources[2]
    assert sources[2] > 5*sources[0]
    waves = make_initial_state(Wave_parameters(resolution=128, wind_speed=25., pinch=0), "cpu")
    assert torch.count_nonzero(emission_mask(evaluate(waves, 4.), p)) == 0


def test_compression_birth_uses_the_existing_rgb_history():
    p = Foam_parameters(resolution=16, emission_model="compression", breakup=0,
                        diffusion=0, exchange=0, surface_half_life=1e15,
                        shallow_half_life=1e15, deep_half_life=1e15)
    frame = stretch_frame(.25)
    before = frame.displacement.clone()
    state = make_foam_state(frame.parameters, 0, p, "cpu")
    emitted = step_foam(state, frame)
    expected = torch.tensor((.8, .2, 0))*p.emission*.1
    torch.testing.assert_close(emitted.density.mean(dim=(1, 2)), expected)
    quiet = stretch_frame(1., time=.2)
    lingered = step_foam(emitted, quiet)
    torch.testing.assert_close(lingered.density, emitted.density, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(frame.displacement, before, atol=0, rtol=0)
    assert torch.count_nonzero(state.density) == 0


def test_legacy_mask_keeps_its_original_float32_bits():
    crest = torch.tensor([-1., -.25, 0., .3, .5001, .6, .7, .8001,
                          .9, 1., 1.1, 1.2, 1.4, 1.6, 2., 3.])
    frame = stretch_frame(-crest)
    # Captured from the legacy CPU implementation before adding this option.
    expected = torch.tensor([0, 0, 0, 0, 1029657194, 1043058405, 1050972730,
        1056814284, 1059804889, 1062410367, 1064349406, 1065309498,
        1065353216, 1065353216, 1065353216, 1065353216], dtype=torch.int32)
    p = Foam_parameters(resolution=16, emission_model="legacy")
    assert p.emission_model == "legacy"
    mask = emission_mask(frame, p, .71, .23)
    assert torch.equal(mask[0].view(torch.int32), expected)
    changed_inactive_settings = replace(p, compression_threshold=.2, compression_width=.05)
    assert torch.equal(mask, emission_mask(frame, changed_inactive_settings, .71, .23))


def test_emitter_switch_resets_history_but_threshold_edits_do_not():
    frame = stretch_frame(.25)
    legacy = Foam_parameters(resolution=16, emission_model="legacy")
    compression = replace(legacy, emission_model="compression")
    state = make_foam_state(frame.parameters, 0, legacy, "cpu")
    assert foam_reset_reason(state, frame.parameters, compression) == "foam emission model changed"
    reset = update_foam(state, frame, compression)
    assert reset.parameters.emission_model == "compression"
    assert torch.count_nonzero(reset.density) == 0
    assert foam_reset_reason(reset, frame.parameters,
                             replace(compression, compression_threshold=.7)) is None


def test_old_parameters_default_to_legacy_and_compression_checkpoint_roundtrips(tmp_path):
    old = asdict(Foam_parameters(resolution=16))
    for name in ("emission_model", "compression_threshold", "compression_width"):
        del old[name]
    assert Foam_parameters().emission_model == "compression"
    assert Foam_parameters.from_dict(old) == Foam_parameters(resolution=16, emission_model="legacy")
    p = Foam_parameters(resolution=16, emission_model="compression", breakup=0)
    frame = stretch_frame(.25)
    state = step_foam(make_foam_state(frame.parameters, 0, p, "cpu"), frame)
    path = tmp_path/"compression.npz"
    save_foam(path, state)
    loaded = load_foam(path, "cpu")
    assert loaded.parameters == p
    torch.testing.assert_close(loaded.density, state.density, atol=0, rtol=0)


@pytest.mark.parametrize("changes", [
    {"emission_model": "unknown"}, {"compression_threshold": -.1},
    {"compression_threshold": 1.01}, {"compression_threshold": float("nan")},
    {"compression_width": 0}, {"compression_width": 1.01},
    {"compression_width": float("inf")},
])
def test_invalid_compression_controls(changes):
    with pytest.raises(ValueError):
        Foam_parameters(**changes)
