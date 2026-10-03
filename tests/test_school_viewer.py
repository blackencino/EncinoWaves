# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Replay semantics and Retina input mapping without a graphics device."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from encino_waves import school_viewer, viewer as base_viewer
from encino_waves.school_timeline import Rehearsal
from encino_waves.school_viewer import SchoolViewer, _GuiCanvas, WIDTH, HEIGHT
from encino_waves.camera import frame_domain
from encino_waves.editing import make_wave_basis, state_from_basis
from encino_waves.foam import Foam_parameters
from encino_waves.model import Wave_parameters
from encino_waves.render import Look


@pytest.fixture
def replay(monkeypatch):
    result = SchoolViewer.__new__(SchoolViewer)
    result.replay = True
    result.replay_elapsed = 1/24
    result.render_resolution = None
    result.parameters = Wave_parameters(resolution=16)
    result.camera = frame_domain(512)
    result.look = Look()
    result.foam_parameters = Foam_parameters()
    result.foam_state = object()
    result.comparison_foam_state = object()
    result.selected_scene = 2
    result.comparing = False
    result.tessendorf_only = False
    result.playing = True
    result.speed = 1.0
    result.show_ui = True
    result.time = 27.5
    result.last_time = 100
    result.seek_serial = 0
    result.seek_time = 10.0
    result.reset_foam_serial = 0
    result.ui_state = {"headers": {}, "scroll": 0.0}
    result._applying_sample = False
    result._has_replay_sample = False
    result.performance = None
    result._performance_replay = False
    result._recording_initial_values = None
    result.rehearsal = None
    positions = []
    io = SimpleNamespace(mouse_pos=SimpleNamespace(x=1440, y=500),
                         add_mouse_pos_event=lambda *point: positions.append(point),
                         mouse_draw_cursor=False)
    monkeypatch.setattr(school_viewer.imgui, "get_io", lambda: io)
    result._test_positions = positions
    return result


def test_first_replay_sample_preserves_checkpoint_time_and_foam(replay):
    values = deepcopy(replay.rehearsal_values())
    values.update(seek_serial=8, seek_time=12, reset_foam_serial=13)
    foam, comparison_foam = replay.foam_state, replay.comparison_foam_state
    replay.apply_rehearsal_sample(values)
    assert replay.time == 27.5
    assert replay.foam_state is foam
    assert replay.comparison_foam_state is comparison_foam
    assert (replay.seek_serial, replay.reset_foam_serial) == (8, 13)
    assert replay._test_positions == [(-100, -100)]
    assert replay._has_replay_sample and not replay._applying_sample


def test_explicit_seek_applies_once_and_clears_both_foam_histories(replay):
    values = deepcopy(replay.rehearsal_values())
    replay.apply_rehearsal_sample(values)
    values.update(seek_serial=1, seek_time=87, reset_foam_serial=1)
    replay.apply_rehearsal_sample(values)
    assert replay.time == 87
    assert replay.foam_state is None and replay.comparison_foam_state is None
    replay.time += 1/24
    resumed_foam = object()
    replay.foam_state = resumed_foam
    replay.apply_rehearsal_sample(values)
    assert replay.time == pytest.approx(87+1/24)
    assert replay.foam_state is resumed_foam
    assert replay.reset_foam_serial == 1


def test_explicit_reset_without_seek_preserves_time(replay):
    values = deepcopy(replay.rehearsal_values())
    replay.apply_rehearsal_sample(values)
    values["reset_foam_serial"] = 1
    replay.apply_rehearsal_sample(values)
    assert replay.time == 27.5
    assert replay.foam_state is None and replay.comparison_foam_state is None
    assert replay.reset_foam_serial == 1


def test_live_seek_records_an_explicit_event_instead_of_ongoing_time(replay):
    replay._seek_time(45)
    values = replay.rehearsal_values()
    assert (values["seek_serial"], values["seek_time"]) == (1, 45)
    assert values["reset_foam_serial"] == 1
    assert "time" not in values
    assert replay.time == 45
    assert replay.foam_state is None


def test_recorded_camera_moves_when_paused_and_controls_step(replay):
    replay.playing = False
    first = deepcopy(replay.rehearsal_values())
    rehearsal = Rehearsal("initial_scene.json", first)
    rehearsal.append(2, first)
    camera = replay.camera.orbit(180, -45, WIDTH, HEIGHT)
    later = deepcopy(first)
    later.update(camera=asdict(camera), parameters=asdict(replace(replay.parameters, wind_speed=32)),
                 look=asdict(replace(replay.look, exposure=1.2)))
    rehearsal.append(3, later)
    timeline = rehearsal.finish(4)
    replay.apply_rehearsal_sample(timeline.sample(0))
    replay.apply_rehearsal_sample(timeline.sample(2.5))
    assert not replay.playing and replay.time == 27.5
    assert replay.camera != frame_domain(512) and replay.camera != camera
    assert replay.parameters.wind_speed == 17
    replay.apply_rehearsal_sample(timeline.sample(3))
    assert replay.camera == camera
    assert replay.parameters.wind_speed == 32
    assert replay.look.exposure == 1.2


def test_offline_resolution_override_and_grid_change_are_synchronous(replay):
    replay.device_name = "cpu"
    replay.basis = make_wave_basis(replay.parameters, "cpu")
    replay.state = state_from_basis(replay.basis, replay.parameters)
    replay.comparison_state = None
    replay.future = None
    replay.reset_requested = False
    replay.changed = False
    replay.render_resolution = 32
    values = deepcopy(replay.rehearsal_values())
    values["parameters"]["domain"] = 640
    replay.apply_rehearsal_sample(values)
    replay._update_parameters(1/24)
    assert replay.parameters.resolution == 32
    assert replay.basis.parameters.resolution == 32
    assert replay.basis.parameters.domain == 640
    assert replay.state.parameters == replay.parameters
    assert replay.future is None and not replay.changed


def test_draw_uses_fixed_simulation_steps_and_respects_pause(replay, monkeypatch):
    # Keep the real Viewer.draw timing branch and replace only graphics work.
    replay._poll_files = lambda: None
    replay._update_parameters = lambda delta: None
    replay._upload_state = lambda *args: None
    replay.context = SimpleNamespace(get_current_texture=lambda: SimpleNamespace(create_view=lambda: None))
    replay.canvas = SimpleNamespace(get_physical_size=lambda: (WIDTH, HEIGHT))
    replay.renderer = SimpleNamespace(frame=None, shading_statistics=SimpleNamespace(crest_gain=1, crest_bias=0),
                                      upload_foam=lambda value: None, draw=lambda *args: None)
    replay.gui = SimpleNamespace(render=lambda: None)
    replay.future = None
    replay.state = object()
    replay.rendered_state = None
    replay.saved_shading = None
    replay.save_requested = False
    replay.frames = 0
    replay.frame_times = []
    replay.max_frames = 0
    monkeypatch.setattr(base_viewer, "update_foam", lambda state, *args: state)
    replay.speed = 2
    replay.draw()
    assert replay.time == pytest.approx(27.5+2/24)
    replay.playing = False
    replay.draw()
    assert replay.time == pytest.approx(27.5+2/24)
    replay.playing = True
    replay.replay_elapsed = 0
    replay.draw()
    assert replay.time == pytest.approx(27.5+2/24)
    assert replay.frames == 3


def test_retina_gui_proxy_maps_to_output_pixels_and_preserves_native_input():
    handlers = []
    context = SimpleNamespace(_config={"format": "rgba8unorm"}, texture=object(), physical_size=(1920, 1080))
    native = SimpleNamespace(get_wgpu_context=lambda: context, get_logical_size=lambda: (960, 540),
                             add_event_handler=lambda callback, *events, **options:
                                 handlers.append((callback, events, options)))
    proxy = _GuiCanvas(native)
    seen = []

    def consume(event):
        seen.append(dict(event))
        event["stop_propagation"] = True

    proxy.add_event_handler(consume, "pointer_move", "resize", order=-99)
    handler, events, options = handlers[0]
    assert events == ("pointer_move", "resize") and options == {"order": -99}
    pointer = {"event_type": "pointer_move", "x": 480, "y": 270}
    handler(pointer)
    assert seen[-1]["x"] == 960 and seen[-1]["y"] == 540
    assert (pointer["x"], pointer["y"]) == (480, 270)
    assert pointer["stop_propagation"] is True
    handler({"event_type": "resize", "width": 960, "height": 540, "pixel_ratio": 2})
    assert (seen[-1]["width"], seen[-1]["height"], seen[-1]["pixel_ratio"]) == (WIDTH, HEIGHT, 1)
    assert proxy.get_wgpu_context().logical_size == (WIDTH, HEIGHT)
    assert proxy.get_wgpu_context().pixel_ratio == 1
    assert proxy.get_wgpu_context().texture is context.texture


def test_retina_canvas_locks_physical_size_and_title_safe_panel():
    dimensions = []
    canvas = SimpleNamespace(get_pixel_ratio=lambda: 2,
                             set_logical_size=lambda *size: dimensions.append(size))
    SchoolViewer._fit_canvas(canvas)
    assert dimensions == [(960, 540)]
    viewer = SchoolViewer.__new__(SchoolViewer)
    x, y, width, height = viewer._control_panel_rect(960, 540)
    assert 0 <= x < x+width <= WIDTH
    assert 200 <= y < y+height <= 880
