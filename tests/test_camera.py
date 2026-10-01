import numpy as np
import pytest
from encino_waves.camera import frame_domain, look_at


def test_maya_tumble_preserves_center_of_interest():
    camera=look_at((30,-40,25))
    moved=camera.orbit(120,-45,1200,900)
    np.testing.assert_allclose(moved.pivot,camera.pivot,atol=1e-12)
    assert moved.center_of_interest==camera.center_of_interest
    assert moved.yaw==pytest.approx(camera.yaw+40)
    assert moved.pitch==pytest.approx(camera.pitch+20)


def test_maya_dolly_preserves_pivot_and_is_reversible():
    camera=frame_domain(512)
    moved=camera.dolly(100,1200)
    np.testing.assert_allclose(moved.pivot,camera.pivot,atol=1e-12)
    assert moved.center_of_interest<camera.center_of_interest
    np.testing.assert_allclose(moved.dolly(-100,1200).eye,camera.eye,atol=1e-12)


def test_track_moves_eye_and_pivot_together():
    camera=frame_domain(512)
    moved=camera.track(50,20,1200,900)
    np.testing.assert_allclose(moved.eye-camera.eye,moved.pivot-camera.pivot,atol=1e-12)
    assert moved.pitch==camera.pitch
    assert moved.yaw==camera.yaw


def test_initial_frame_matches_original_bounds():
    camera=frame_domain(512)
    np.testing.assert_allclose(camera.pivot,(0,0,0),atol=1e-12)
    assert camera.height==pytest.approx(np.sqrt(3)*.25*512)
    assert camera.fov==45
