# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""CPU checks for a clean ocean horizon with unchanged linear HDR sky color."""
import numpy as np
import pytest

from encino_waves.sky import ocean_dome


def latitude_image(height=181):
    latitude = 90-(np.arange(height)+.5)*180/height
    rgba = np.ones((height, 2*height, 4), np.float32)
    rgba[...,0] = ((latitude+90)/180)[:,None]
    rgba[...,1] = .5
    rgba[...,2] = .25
    return rgba, latitude


def test_spherical_remap_preserves_zenith_and_maps_horizon_to_trim():
    source, latitude = latitude_image()
    dome = ocean_dome(source)
    # An odd-height map has an exact horizon sample. The red ramp is linear
    # in latitude, so its interpolated value directly identifies the source ray.
    np.testing.assert_allclose(dome[len(latitude)//2,:,0], (90+8)/180, atol=1e-7)
    np.testing.assert_array_equal(dome[0], source[0])
    for row in (25, 50, 75):
        expected_latitude = 8+(90-8)*latitude[row]/90
        np.testing.assert_allclose(dome[row,:,0], (90+expected_latitude)/180, atol=1e-7)
    np.testing.assert_array_equal(dome[...,3], np.ones(source.shape[:2], np.float32))


@pytest.mark.parametrize("height", [2, 17, 180])
@pytest.mark.parametrize("trim", [0, 8])
def test_original_ground_never_enters_dome(height, trim):
    source = np.ones((height, 2*height, 4), np.float32)
    source[...,:3] = [.1, .2, .3]
    # Include the source horizon row for odd heights: it can contain land too.
    source[height//2:,:,:3] = [9000, 8000, 7000]
    dome = ocean_dome(source, trim)
    assert np.all(dome[...,:3] <= np.array([.1, .2, .3], np.float32)+1e-7)
    assert np.all(dome[...,:3] > 0)


def test_lower_hemisphere_darkens_horizon_without_recoloring():
    source = np.ones((180, 360, 4), np.float32)
    source[...,:3] = [2, 4, 8]
    dome = ocean_dome(source)
    np.testing.assert_allclose(dome[:90,:,:3], np.broadcast_to([2, 4, 8], (90, 360, 3)))
    assert np.all(np.diff(dome[90:,:,:3], axis=0) < 0)
    assert np.all(dome[-1,:,:3] < .21*np.array([2, 4, 8]))
    np.testing.assert_allclose(dome[...,1], 2*dome[...,0])
    np.testing.assert_allclose(dome[...,2], 4*dome[...,0])
    np.testing.assert_array_equal(dome[...,3], np.ones((180, 360), np.float32))


def test_horizontal_rotation_commutes_with_dome_conversion():
    source = np.random.default_rng(19).uniform(0, 12, (31, 62, 4)).astype(np.float32)
    dome = ocean_dome(source)
    for shift in (1, 13, -9):
        np.testing.assert_array_equal(ocean_dome(np.roll(source, shift, axis=1)),
                                      np.roll(dome, shift, axis=1))


def test_result_is_new_finite_nonnegative_linear_hdr_array():
    source = np.random.default_rng(4).uniform(-.5, 10, (32, 64, 4)).astype(np.float32)
    original = source.copy()
    source.flags.writeable = False
    dome = ocean_dome(source)
    assert dome.shape == source.shape
    assert dome.dtype == np.float32
    assert not np.shares_memory(dome, source)
    assert np.isfinite(dome).all()
    assert np.min(dome) >= 0
    assert np.max(dome[...,:3]) > 1  # This operation must not tone-map HDR values.
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("trim", [-1, 90, 91, np.inf, np.nan])
def test_invalid_trim_is_rejected(trim):
    with pytest.raises(ValueError, match="Horizon trim"):
        ocean_dome(np.ones((4, 8, 4)), trim)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, np.nan])
def test_nonfinite_source_is_rejected(bad_value):
    source = np.ones((4, 8, 4), np.float32)
    source[0,0,0] = bad_value
    with pytest.raises(ValueError, match="finite pixels"):
        ocean_dome(source)


@pytest.mark.parametrize("shape", [(4, 8, 3), (1, 2, 4), (4, 0, 4), (4, 8)])
def test_invalid_image_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="RGBA latitude/longitude"):
        ocean_dome(np.ones(shape))
