"""Render the complete viewer UI and exercise scene / camera / comparison paths.

Uses the real GPU and renderer on an offscreen canvas, so it can also run while
macOS has the desktop locked. Live window interaction remains a separate check.
"""
from pathlib import Path
from PIL import Image
import numpy as np
from rendercanvas.offscreen import RenderCanvas
from encino_waves.viewer import Viewer
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
    pixels = np.asarray(canvas.draw())
    Image.fromarray(pixels).save('renders/viewer_comparison.png')
    viewer.save_scene('renders/viewer_roundtrip.json')
    before = viewer.renderer.render_image(480,270,viewer.camera,viewer.look,left_renderer=viewer.comparison_renderer)
    viewer.load_scene('renders/viewer_roundtrip.json')
    assert viewer.comparing
    canvas.draw()
    after = viewer.renderer.render_image(480,270,viewer.camera,viewer.look,left_renderer=viewer.comparison_renderer)
    np.testing.assert_array_equal(before,after)
    original_header = imgui.collapsing_header
    def expanded_header(label):
        imgui.set_next_item_open(True)
        return original_header(label)
    imgui.collapsing_header = expanded_header
    canvas.draw()
    print('Viewer UI, live wind/depth edits, phase replay, Maya events, comparison, and all panels passed',flush=True)
finally:
    viewer.executor.shutdown(wait=True,cancel_futures=True)
    canvas.close()
