"""Radiometric checks for the display-only HDR lighting analysis."""
import math

import numpy as np
import pytest

from encino_waves.lighting import analyze_sky, evaluate_diffuse


def dome(height=180, width=360, color=(1., 1., 1.)):
    return np.broadcast_to(np.array(color, np.float32), (height, width, 3)).copy()


def directions(height, width):
    theta = (np.arange(height)+.5)*math.pi/height
    phi = (np.arange(width)+.5)*2*math.pi/width-math.pi
    return np.stack(np.broadcast_arrays(np.sin(theta)[:, None]*np.cos(phi)[None, :],
                                         np.sin(theta)[:, None]*np.sin(phi)[None, :],
                                         np.cos(theta)[:, None]), axis=-1)


@pytest.mark.parametrize("height", [16, 17, 180])
def test_constant_upper_dome_has_exact_lambertian_illumination(height):
    color = np.array([.2, .7, 1.8])
    lighting = analyze_sky(dome(height, 2*height, color))
    # For a uniform upper hemisphere, E(n)/pi = C*(1+n.z)/2.
    normals = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0],
                        [math.sqrt(.75), 0, .5]])
    expected = (1+normals[:, 2, None])/2*color
    np.testing.assert_allclose(evaluate_diffuse(lighting.diffuse_sh, normals), expected,
                               rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(lighting.mean_radiance, color, rtol=1e-7)
    np.testing.assert_allclose(lighting.dominant_direction, [0, 0, 1], atol=1e-7)
    assert not lighting.has_direct
    assert lighting.direct_angular_radius == 0


def test_sky_gradient_orientation_and_longitude_rotation():
    height, width = 180, 360
    xyz = directions(height, width)
    pixels = np.repeat((1+.6*xyz[..., 0])[..., None], 3, axis=-1)
    original = analyze_sky(pixels)
    rotated = analyze_sky(np.roll(pixels, width//4, axis=1))
    normal_x, normal_y = [1, 0, 0], [0, 1, 0]
    # Integrating the x-gradient over the quarter sphere gives .5 +/- .6/3.
    np.testing.assert_allclose(evaluate_diffuse(original.diffuse_sh, normal_x), .7, atol=3e-5)
    np.testing.assert_allclose(evaluate_diffuse(original.diffuse_sh, [-1, 0, 0]), .3, atol=3e-5)
    np.testing.assert_allclose(evaluate_diffuse(rotated.diffuse_sh, normal_y), .7, atol=3e-5)
    np.testing.assert_allclose(evaluate_diffuse(original.diffuse_sh, normal_y), .5, atol=3e-5)
    assert not original.has_direct


def test_lower_hemisphere_does_not_light_foam_or_create_a_sun():
    pixels = dome()
    baseline = analyze_sky(pixels)
    pixels[90:] = [1e5, 1e4, 1e3]
    changed = analyze_sky(pixels)
    np.testing.assert_array_equal(changed.diffuse_sh, baseline.diffuse_sh)
    np.testing.assert_array_equal(changed.mean_radiance, baseline.mean_radiance)
    assert not changed.has_direct


def test_uniform_bright_cap_has_analytic_irradiance_and_angular_size():
    pixels = dome()
    source_radiance = np.array([900., 600., 300.])
    pixels[:3] += source_radiance
    lighting = analyze_sky(pixels)
    angular_radius = math.radians(3)
    # A polar uniform cap illuminates its perpendicular plane by L*pi*sin(a)^2.
    expected_irradiance = source_radiance*math.pi*math.sin(angular_radius)**2
    assert lighting.has_direct
    np.testing.assert_allclose(lighting.direct_irradiance, expected_irradiance, rtol=2e-6)
    np.testing.assert_allclose(lighting.dominant_direction, [0, 0, 1], atol=1e-7)
    assert lighting.direct_angular_radius == pytest.approx(angular_radius, abs=2e-7)
    # Removing the cap leaves the original unit dome; its energy is not counted twice.
    np.testing.assert_allclose(evaluate_diffuse(lighting.ambient_sh, [0, 0, 1]), 1, atol=2e-7)
    expected_mean = 1+source_radiance*(1-math.cos(angular_radius))
    np.testing.assert_allclose(lighting.mean_radiance, expected_mean, rtol=2e-7)


def test_rotating_compact_source_preserves_energy_and_rotates_direction():
    height, width = 180, 360
    xyz = directions(height, width)
    target = np.array([math.sqrt(.5), 0, math.sqrt(.5)])
    pixels = dome(height, width)
    mask = xyz @ target > math.cos(math.radians(2))
    pixels[mask] += [1000, 700, 400]
    original = analyze_sky(pixels)
    rotated = analyze_sky(np.roll(pixels, width//4, axis=1))
    assert original.has_direct and rotated.has_direct
    np.testing.assert_allclose(rotated.direct_irradiance, original.direct_irradiance, rtol=2e-6)
    np.testing.assert_allclose(rotated.dominant_direction,
                               [-original.dominant_direction[1], original.dominant_direction[0],
                                original.dominant_direction[2]], atol=1e-6)
    assert rotated.direct_angular_radius == pytest.approx(original.direct_angular_radius, abs=1e-7)


def test_broad_bright_cloud_is_not_a_directional_light():
    xyz = directions(180, 360)
    pixels = dome()
    pixels += (20*np.maximum(xyz[..., 0], 0)**8)[..., None]
    lighting = analyze_sky(pixels)
    assert not lighting.has_direct
    np.testing.assert_array_equal(lighting.diffuse_sh, lighting.ambient_sh)


def test_analysis_preserves_input_and_returns_readonly_arrays():
    pixels = np.concatenate((dome(16, 32), np.ones((16, 32, 1), np.float32)), axis=-1)
    before = pixels.copy()
    lighting = analyze_sky(pixels)
    np.testing.assert_array_equal(pixels, before)
    with pytest.raises(ValueError):
        lighting.diffuse_sh[0, 0] = 0
    with pytest.raises(ValueError):
        lighting.dominant_direction[0] = 0


def test_black_sky_is_finite_and_has_no_source():
    lighting = analyze_sky(np.zeros((16, 32, 3)))
    assert not lighting.has_direct
    np.testing.assert_array_equal(lighting.diffuse_sh, np.zeros((9, 3)))
    np.testing.assert_array_equal(lighting.dominant_direction, [0, 0, 1])


@pytest.mark.parametrize("bad", [np.full((8, 16, 3), np.nan), np.full((8, 16, 3), -1),
                                  np.ones((8, 16)), np.ones((1, 2, 3))])
def test_invalid_radiance_is_rejected(bad):
    with pytest.raises(ValueError):
        analyze_sky(bad)
