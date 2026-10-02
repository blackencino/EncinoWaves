"""Render the complete viewer UI and exercise scene / camera / comparison paths.

Uses the real GPU and renderer on an offscreen canvas, so it can also run while
macOS has the desktop locked. Live window interaction remains a separate check.
"""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import math
from PIL import Image
import numpy as np
from rendercanvas.offscreen import RenderCanvas
from encino_waves.viewer import Viewer
from encino_waves.foam import prepare_foam
from imgui_bundle import imgui

canvas = RenderCanvas(size=(1440, 900))
viewer = Viewer(resolution=512, canvas=canvas)
viewer.playing = False
Path('renders').mkdir(exist_ok=True)
try:
    canvas.draw()
    pixels = np.asarray(canvas.draw())
    assert viewer.frames == 2
    Image.fromarray(pixels).save('renders/viewer_ui.png')
    before = viewer.camera
    viewer.on_pointer({'event_type':'pointer_down','x':400,'y':300,'buttons':(1,),'modifiers':('Alt',)})
    viewer.on_pointer({'event_type':'pointer_move','x':450,'y':310,'buttons':(1,),'modifiers':('Alt',)})
    viewer.on_pointer({'event_type':'pointer_up'})
    assert viewer.camera != before
    np.testing.assert_allclose(viewer.camera.pivot,before.pivot,atol=1e-10)
    viewer.on_key({'key':'f'})
    np.testing.assert_allclose(viewer.camera.eye,before.eye,atol=1e-10)
    viewer.on_key({'key':'c'})
    pixels = np.asarray(canvas.draw())
    assert viewer.comparison_state is not None
    basis = viewer.basis
    viewer.edit_parameters(wind_speed=24,depth=8)
    for i in range(90):
        viewer.time += 1/60
        viewer._update_parameters(1/60)
    assert viewer.basis is basis
    assert viewer.future is None and not viewer.changed
    assert viewer.state.parameters.wind_speed == 24
    assert viewer.state.parameters.depth == 8
    assert viewer.state.phase_steps
    assert viewer.comparison_state.phase is viewer.state.phase
    viewer.foam_state=prepare_foam(viewer.state,viewer.time,viewer.foam_parameters,preroll=1)
    viewer.comparison_foam_state=prepare_foam(viewer.comparison_state,viewer.time,viewer.foam_parameters,preroll=1)
    assert float(viewer.foam_state.density.sum().cpu()) > 0
    foam_before=viewer.foam_state
    pixels = np.asarray(canvas.draw())
    Image.fromarray(pixels).save('renders/viewer_comparison.png')
    # Rotating the mesh changes placement only, including while paused.
    states = (viewer.state, viewer.comparison_state)
    renderers = (viewer.renderer, viewer.comparison_renderer)
    frames = tuple(renderer.frame for renderer in renderers)
    statistics = tuple(renderer.shading_statistics for renderer in renderers)
    images = tuple(renderer.render_image(640,400,viewer.camera,viewer.look) for renderer in renderers)
    viewer.edit_parameters(wind_direction=73)
    with patch('encino_waves.editing.state_from_basis',side_effect=AssertionError('Direction rebuilt the spectrum')):
        for i in range(120): viewer._update_parameters(1/60)
    with patch('encino_waves.viewer.evaluate',side_effect=AssertionError('Direction recomputed the FFT')):
        canvas.draw()
    assert not viewer.changed and viewer.state.parameters.wind_direction == 73
    assert viewer.foam_state is foam_before
    assert viewer.state.h_positive is states[0].h_positive
    assert viewer.comparison_state.h_positive is states[1].h_positive
    for renderer, frame, shading in zip(renderers,frames,statistics):
        assert renderer.frame.displacement is frame.displacement
        assert renderer.frame.normal is frame.normal
        assert renderer.shading_statistics == shading
    # Rotate camera and environment with the mesh: the image must stay the same.
    # Checks coordinates, displacement vectors, normals, lighting and both seas.
    camera = viewer.camera
    angle = math.radians(73)
    c,s = math.cos(angle),math.sin(angle)
    rotated_camera = replace(camera,x=c*camera.x-s*camera.y,y=s*camera.x+c*camera.y,yaw=camera.yaw-73)
    rotated_look = replace(viewer.look,sky_rotation=viewer.look.sky_rotation-73)
    for renderer, expected in zip(renderers,images):
        actual = renderer.render_image(640,400,rotated_camera,rotated_look)
        error = np.abs(actual.astype(np.float32)-expected.astype(np.float32))
        print(f'Rigid rotation pixel error: mean {error.mean():.4f}, 99th percentile {np.percentile(error,99):.1f}',flush=True)
        assert error.mean() < .2 and np.percentile(error,99) <= 2
    Image.fromarray(np.asarray(canvas.draw())).save('renders/viewer_rotation.png')
    viewer.save_scene('renders/viewer_roundtrip.json')
    before = viewer.renderer.render_image(480,270,viewer.camera,viewer.look,left_renderer=viewer.comparison_renderer)
    viewer.load_scene('renders/viewer_roundtrip.json')
    assert viewer.comparing
    canvas.draw()
    after = viewer.renderer.render_image(480,270,viewer.camera,viewer.look,left_renderer=viewer.comparison_renderer)
    np.testing.assert_array_equal(before,after)
    assert float(viewer.foam_state.density.sum().cpu()) > 0
    wave_frame=viewer.renderer.frame
    viewer.reset_foam()
    canvas.draw()
    assert float(viewer.foam_state.density.sum().cpu()) == 0
    assert viewer.renderer.frame is wave_frame
    original_header = imgui.collapsing_header
    def expanded_header(label):
        imgui.set_next_item_open(True)
        return original_header(label)
    imgui.collapsing_header = expanded_header
    canvas.draw()
    print('Viewer UI, wave edits, rotation, RGB foam replay/reset, Maya events, comparison, and all panels passed',flush=True)
finally:
    viewer.executor.shutdown(wait=True,cancel_futures=True)
    canvas.close()
