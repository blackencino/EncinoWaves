"""Presentation mode changes preserve modern settings and expose real tuning."""
from dataclasses import asdict
from types import SimpleNamespace

from encino_waves import school_viewer
from encino_waves.school_viewer import SchoolViewer
from encino_waves.school_performance import Performance
from encino_waves.viewer import Viewer
from encino_waves.model import Wave_parameters
from encino_waves.render import Look


def test_mode_switch_preserves_modern_settings_and_clears_foam():
    viewer = Viewer.__new__(Viewer)
    viewer.parameters = Wave_parameters(resolution=16, swell=1.2, depth=3)
    viewer.look = Look(foam=.8, aeration=1.7, crest_crumble=True)
    parameters, look = viewer.parameters, viewer.look
    viewer.foam_state = viewer.comparison_foam_state = object()
    viewer.comparison_state = object()
    viewer.comparing = True
    viewer.set_tessendorf_mode(True)
    assert viewer.tessendorf_only and not viewer.comparing
    assert viewer.foam_state is viewer.comparison_foam_state is None
    assert viewer.comparison_state is None
    viewer.set_tessendorf_mode(False)
    assert not viewer.tessendorf_only
    assert viewer.parameters is parameters and viewer.look is look


def test_school_shortcuts_change_gain_and_keep_unavailable_controls_inactive(monkeypatch):
    monkeypatch.setattr(school_viewer.imgui, "get_io", lambda: SimpleNamespace(want_text_input=False))
    viewer = SchoolViewer.__new__(SchoolViewer)
    viewer.replay = False
    viewer._applying_sample = False
    viewer.reset_foam_serial = 0
    viewer._held_keys = set()
    viewer.parameters = Wave_parameters(resolution=16)
    viewer.performance = Performance(asdict(viewer.parameters), initial_look=asdict(Look()))
    viewer.tessendorf_only = False
    clock = [0.]
    viewer._wall_time = lambda: clock[0]
    viewer.on_key({"key": "t"})
    assert viewer.tessendorf_only
    viewer.on_key({"key": "g"})
    viewer.on_key({"key": "ArrowRight"})
    at_one = viewer.performance.sample(1.)
    assert at_one["parameters"]["phillips_amplitude"] > 100
    assert at_one["overlay"]["label"] == "Phillips Gain"
    count = len(viewer.performance.events)
    for key in ("f", "s", "m"):
        viewer.on_key({"key": key})
    assert len(viewer.performance.events) == count
    viewer._on_key_up({"key": "t"})
    viewer.on_key({"key": "t"})
    assert not viewer.tessendorf_only


def test_manual_gain_and_pinch_survive_performance_roundtrip():
    performance = Performance(asdict(Wave_parameters(resolution=16)), initial_look=asdict(Look()))
    performance.mode_key("g", 0.)
    performance.direction_key("right", True, .1)
    performance.direction_key("right", False, 1.)
    performance.mode_key("p", 1.5)
    performance.direction_key("left", True, 1.6)
    saved = Performance.from_dict(performance.to_dict())
    assert saved.sample(2.) == performance.sample(2.)
    values = saved.sample(2.)["parameters"]
    assert values["phillips_amplitude"] > 100
    assert values["pinch"] < .75


def test_switching_models_stops_hidden_control_motion_and_replays():
    performance = Performance(asdict(Wave_parameters(resolution=16)), initial_look=asdict(Look()))
    performance.mode_key("s", 0.)
    performance.direction_key("right", True, .1)
    performance.stop_controls(1.)
    stopped = performance.sample(1.)
    assert performance.sample(3.)["parameters"] == stopped["parameters"]
    assert not performance.sample(3.)["overlay"]["active"]
    restored = Performance.from_dict(performance.to_dict())
    assert restored.sample(3.) == performance.sample(3.)
