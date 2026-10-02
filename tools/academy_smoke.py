"""One focused GPU check of the Academy performance and movie path."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from rendercanvas.offscreen import RenderCanvas

from encino_waves.academy_viewer import AcademyViewer
from encino_waves.academy_performance import Performance
from encino_waves.academy_timeline import Rehearsal


directory = Path("renders/academy_performance_check")
directory.mkdir(parents=True, exist_ok=True)
canvas = RenderCanvas(size=(960, 540), pixel_ratio=2)
viewer = AcademyViewer(canvas=canvas)
clock = [0.0]
viewer._wall_time = lambda: clock[0]


def draw(at, name=None):
    clock[0] = at
    previous = viewer.frames
    pixels = np.asarray(canvas.draw())
    assert viewer.frames == previous + 1, "Canvas swallowed a draw error"
    assert pixels.shape == (1080, 1920, 4)
    if name:
        Image.fromarray(pixels).save(directory / f"{name}.png")
    return pixels


def key(name, at, down=True):
    clock[0] = at
    handler = viewer.on_key if down else viewer._on_key_up
    handler({"event_type": "key_down" if down else "key_up", "key": name})


try:
    viewer.playing = False
    draw(0)
    assert not viewer._show_title(), "Preparation must not show or record a title"
    assert viewer.rehearsal is None
    prepared_camera = viewer.camera
    viewer.start_rehearsal()
    assert viewer._show_title() and viewer.camera == prepared_camera
    title = draw(0, "title")
    assert title[0, 0, :3].sum() == 0
    assert title[438:620, 500:1420, :3].max() > 200
    viewer.stop_rehearsal()
    assert not viewer._show_title() and viewer.rehearsal is None
    assert viewer.rehearsal_path.is_file() and viewer.last_take_duration is not None
    assert viewer.parameters.resolution == 2048 and viewer.parameters.domain == 1024
    assert viewer.foam_parameters.resolution == 1024
    assert viewer.parameters.trough_damping == 1
    assert viewer.look.crest_crumble_strength == .5
    assert (viewer.camera.height, viewer.camera.pitch, viewer.camera.yaw) == (145.2, -18.1, -139.9)
    viewer.save_scene(directory / "initial_scene.json")
    initial = deepcopy(viewer.rehearsal_values())
    initial["playing"] = True
    draw(4.5, "opening")
    key("w", 4.6)
    key("w", 4.61, False)
    key("ArrowRight", 4.8)
    draw(5.1, "wind")
    assert viewer.parameters.wind_speed > 17
    assert viewer.performance_overlay["right_weight"] > .5
    key("ArrowRight", 5.2, False)
    at_release = viewer.performance.sample(5.2)["parameters"]["wind_speed"]
    assert viewer.performance.sample(5.3)["parameters"]["wind_speed"] > at_release
    key("e", 5.5)
    key("e", 5.51, False)
    draw(7, "example")
    assert viewer.performance_overlay["label"] == "NOAA severe sea state"
    assert viewer.parameters.domain == 1024
    draw(10)
    before = viewer.camera
    width, height = canvas.get_logical_size()
    viewer.on_pointer({"event_type": "pointer_down", "x": 200, "y": 160, "buttons": (1,), "modifiers": ("Alt",)})
    clock[0] = 10.1
    viewer.on_pointer({"event_type": "pointer_move", "x": 230, "y": 166, "buttons": (1,), "modifiers": ("Alt",)})
    assert viewer.camera == before.orbit(30, 6, width, height), "Maya input must stay direct"
    viewer.on_pointer({"event_type": "pointer_up"})
    endpoint = viewer.camera
    draw(14.2)
    assert viewer.performance.camera_moves[-1]["endcam"] == asdict(endpoint)
    midpoint = viewer.performance.sample(12)["camera"]
    assert midpoint not in (asdict(before), asdict(endpoint))
    restored = Performance.from_dict(viewer.performance.to_dict())
    assert restored.sample(5.1) == viewer.performance.sample(5.1)
    recorder = Rehearsal("initial_scene.json", initial)
    recorder.set_performance(viewer.performance.to_dict())
    recorder.save(directory / "rehearsal.json", 8)
    print("Academy title, 2K/1K opening, inertia, examples, unchanged Maya input, endpoint playback and tape replay passed", flush=True)
finally:
    viewer.executor.shutdown(wait=True, cancel_futures=True)
    canvas.close()
