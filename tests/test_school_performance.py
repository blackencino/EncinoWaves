# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Selected presentation controls persist and retain replayable toggles."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from encino_waves.school_performance import CONTROLS, Performance
from encino_waves.model import Wave_parameters


def performance():
    return Performance(asdict(Wave_parameters(resolution=16)), initial_look={"foam": 1.0})


def test_control_key_repeat_release_toggle_and_replacement(monkeypatch):
    from encino_waves import school_viewer

    monkeypatch.setattr(school_viewer.imgui, "get_io", lambda: SimpleNamespace(want_text_input=False))
    viewer = school_viewer.SchoolViewer.__new__(school_viewer.SchoolViewer)
    viewer.replay = False
    viewer.tessendorf_only = False
    viewer._held_keys = set()
    viewer.performance = performance()
    clock = [0.]
    viewer._wall_time = lambda: clock[0]

    viewer.on_key({"key": "w"})
    clock[0] = 60.
    viewer.on_key({"key": "w"})  # A held key's repeat is not a second press.
    assert len(viewer.performance.events) == 1
    selected = viewer.performance.sample(clock[0])["overlay"]
    assert selected["mode"] == "wind" and selected["active"]
    assert selected["opacity"] == 1.

    viewer._on_key_up({"key": "w"})
    viewer.on_key({"key": "w"})
    assert len(viewer.performance.events) == 2
    assert not viewer.performance.sample(clock[0])["overlay"]["active"]

    viewer._on_key_up({"key": "w"})
    clock[0] = 61.
    viewer.on_key({"key": "w"})
    clock[0] = 62.
    viewer.on_key({"key": "d"})
    replaced = viewer.performance.sample(63.)["overlay"]
    assert replaced["mode"] == "depth" and replaced["active"]
    assert replaced["opacity"] == 1.
    assert len(viewer.performance.events) == 4


@pytest.mark.parametrize("mode", CONTROLS)
def test_selected_parameter_stays_visible_after_arrows_and_long_idle(mode):
    controller = performance()
    controller.mode_key(mode, 0.)
    controller.direction_key("right", True, .2)
    controller.direction_key("right", False, .5)
    sampled = controller.sample(60.)
    assert sampled["overlay"]["mode"] == mode
    assert sampled["overlay"]["active"]
    assert sampled["overlay"]["opacity"] == 1.
    assert sampled["overlay"]["right_weight"] < 1e-10
    assert Performance.from_dict(controller.to_dict()).sample(60.) == sampled


def test_hotkey_toggles_off_after_idle_and_another_control_replaces_it():
    controller = performance()
    controller.mode_key("w", 0.)
    controller.mode_key("w", 60.)
    assert not controller.sample(60.)["overlay"]["active"]
    assert controller.sample(60.)["overlay"]["opacity"] == 1.
    assert 0. < controller.sample(60.11)["overlay"]["opacity"] < 1.
    assert controller.sample(60.3)["overlay"]["opacity"] == 0.

    controller.mode_key("w", 61.)
    controller.mode_key("d", 62.)
    entering = controller.sample(62.08)["overlay"]
    assert entering["mode"] == "depth" and entering["active"]
    assert 0. < entering["opacity"] < 1.
    assert controller.sample(90.)["overlay"]["opacity"] == 1.
    controller.mode_key("d", 91.)
    assert not controller.sample(92.)["overlay"]["active"]
    assert controller.sample(92.)["overlay"]["opacity"] == 0.


def test_example_titles_still_fade_and_recording_rebase_preserves_selection():
    controller = performance()
    controller.transition({"wind_speed": 24.}, 0., name="Storm")
    assert controller.sample(2.)["overlay"]["opacity"] == 1.
    assert controller.sample(4.)["overlay"]["opacity"] == 0.
    controller.mode_key("e", 5.)  # Toggle off the named example.
    controller.mode_key("e", 6.)  # Show the generic Examples title.
    assert controller.sample(7.)["overlay"]["opacity"] == 1.
    assert controller.sample(8.)["overlay"]["opacity"] == 0.

    controller.mode_key("g", 10.)
    controller.mode_key("g", 70.)
    rebased = Performance.from_dict(controller.recording_data(50., 72.))
    for relative_time in (0., 10., 20., 20.11, 21.):
        assert rebased.sample(relative_time) == controller.sample(50. + relative_time)
