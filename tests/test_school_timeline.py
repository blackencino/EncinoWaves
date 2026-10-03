# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from encino_waves.school_timeline import (
    School_timeline, MAX_REHEARSAL_SECONDS, Rehearsal, Timeline_sample,
    load_timeline, playback_delta,
)
from encino_waves.camera import frame_domain, interpolate_camera


def snapshot(**changes):
    values = {"parameters": {"wind_speed": 17}, "camera": asdict(frame_domain(512)),
              "look": {"exposure": 0}, "foam": {"enabled": True},
              "selected_scene": 2, "comparing": False, "tessendorf_only": False,
              "playing": True, "speed": 1.0, "show_ui": True,
              "ui_state": {"headers": {"Camera & light": False}, "scroll": 0},
              "seek_serial": 0, "seek_time": 10.0}
    return values | changes


def test_recording_keeps_changes_and_roundtrips_full_ui_state(tmp_path):
    original = snapshot()
    rehearsal = Rehearsal("initial.json", original)
    assert not rehearsal.append(1, original)
    changed = snapshot(parameters={"wind_speed": 24}, playing=False,
                       ui_state={"headers": {"Camera & light": True}, "scroll": 120})
    assert rehearsal.append(2, changed)
    assert not rehearsal.append(3, changed)
    changed["parameters"]["wind_speed"] = 999
    original["look"]["exposure"] = 999
    path = tmp_path/"take.json"
    timeline = rehearsal.save(path, 4)
    restored = load_timeline(path)
    assert restored == timeline
    assert len(restored.samples) == 2
    assert restored.sample(1)["parameters"]["wind_speed"] == 17
    assert restored.sample(2)["parameters"]["wind_speed"] == 24
    assert restored.sample(4)["playing"] is False
    assert restored.sample(0)["look"]["exposure"] == 0
    sampled = restored.sample(2)
    sampled["ui_state"]["scroll"] = -1
    assert restored.sample(2)["ui_state"]["scroll"] == 120


def test_camera_interpolates_after_hold_even_while_simulation_is_paused():
    camera = frame_domain(512)
    initial = snapshot(playing=False)
    rehearsal = Rehearsal("initial.json", initial)
    rehearsal.append(9, initial)
    moved = camera.orbit(90, -30, 1920, 1080).dolly(100, 1920)
    rehearsal.append(10, snapshot(playing=False, camera=asdict(moved)))
    timeline = rehearsal.finish(12)
    assert [item.time for item in timeline.samples] == [0, 9, 10]
    assert timeline.sample(8)["camera"] == asdict(camera)
    assert timeline.sample(9)["camera"] == asdict(camera)
    midpoint = timeline.sample(9.5)
    expected = interpolate_camera(camera, moved, .5)
    np.testing.assert_allclose(list(midpoint["camera"].values()), list(asdict(expected).values()))
    assert midpoint["playing"] is False
    assert timeline.sample(10)["camera"] == asdict(moved)
    assert timeline.sample(100)["camera"] == asdict(moved)


def test_seek_and_target_controls_step_instead_of_interpolating():
    first = snapshot()
    last = snapshot(parameters={"wind_speed": 30}, seek_serial=1, seek_time=50,
                    camera=asdict(replace(frame_domain(512), yaw=45)))
    timeline = School_timeline("initial.json", 2,
        (Timeline_sample(0, first), Timeline_sample(2, last)))
    middle = timeline.sample(1)
    assert middle["parameters"] == first["parameters"]
    assert middle["seek_serial"] == 0 and middle["seek_time"] == 10
    assert middle["camera"] != first["camera"]
    assert timeline.sample(2) == last
    assert timeline.sample(-1) == first


def test_simulation_uses_output_timestep_and_play_state():
    assert playback_delta(snapshot(), 1/24) == pytest.approx(1/24)
    assert playback_delta(snapshot(speed=2), 1/24) == pytest.approx(1/12)
    assert playback_delta(snapshot(playing=False, speed=2), 1/24) == 0


def test_same_tick_changes_replace_sample_without_duplicate_timestamps():
    rehearsal = Rehearsal("initial.json", snapshot())
    rehearsal.append(0, snapshot(show_ui=False))
    rehearsal.append(1, snapshot(show_ui=False))
    rehearsal.append(1, snapshot(camera=asdict(replace(frame_domain(512), yaw=45))))
    timeline = rehearsal.finish(1)
    assert [item.time for item in timeline.samples] == [0, 1]
    assert timeline.sample(0)["show_ui"] is False
    assert timeline.sample(1)["camera"]["yaw"] == 45


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -1, 299, True])
def test_invalid_sample_times_are_rejected(time):
    with pytest.raises(ValueError): Timeline_sample(time, snapshot())


@pytest.mark.parametrize("change", [
    {"duration": 299}, {"duration": float("nan")}, {"duration": 1},
    {"samples": []}, {"initial_scene": ""}, {"version": 2},
    {"samples": [{"time": 0, "values": {"speed": float("nan")}}]},
    {"samples": [{"time": 0, "values": {}}, {"time": 0, "values": {}}]},
    {"samples": [{"time": 2, "values": {}}, {"time": 1, "values": {}}]},
    {"samples": [{"time": 1, "values": {}}]},
    {"samples": [{"time": 0}]},
])
def test_invalid_imported_timeline_is_rejected(tmp_path, change):
    data = {"version": 1, "initial_scene": "initial.json", "duration": 3,
            "samples": [{"time": 0, "values": snapshot()},
                        {"time": 2, "values": snapshot(playing=False)}]}
    path = tmp_path/"invalid.json"
    path.write_text(json.dumps(data | deepcopy(change)))
    with pytest.raises(ValueError): load_timeline(path)


def test_duration_limit_reserves_black_head_and_tail():
    rehearsal = Rehearsal("initial.json", snapshot())
    assert rehearsal.finish(MAX_REHEARSAL_SECONDS).duration+2 == 300
    with pytest.raises(ValueError): rehearsal.finish(MAX_REHEARSAL_SECONDS+.01)
    rehearsal.append(2, snapshot())
    with pytest.raises(ValueError): rehearsal.finish(1)
    with pytest.raises(ValueError): rehearsal.append(1, snapshot())


def test_failed_autosave_preserves_previous_tape(tmp_path, monkeypatch):
    from encino_waves import school_timeline
    rehearsal = Rehearsal("initial.json", snapshot())
    path = tmp_path/"rehearsal.json"
    rehearsal.save(path, 1)
    previous = path.read_bytes()
    rehearsal.append(2, snapshot(playing=False))

    def interrupted_replace(source, destination):
        assert source.parent == path.parent
        assert source.read_text().endswith("\n")
        assert destination == path
        raise OSError("Autosave interrupted")

    monkeypatch.setattr(school_timeline.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted"):
        rehearsal.save(path, 3)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    assert load_timeline(path).duration == 1
