"""Render the production water/presentation shaders against controlled skies.

These tests exercise the graphics path without changing, or depending on, wave
or foam evolution. Flat input maps isolate lighting and presentation invariants.
"""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch
import wgpu

from encino_waves.camera import Camera
from encino_waves.foam import Foam_parameters, make_foam_state
from encino_waves.model import Wave_frame, Wave_parameters
from encino_waves.render import Look, Ocean_renderer


@pytest.fixture(scope="module")
def device():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except RuntimeError as error:
        if "No suitable graphics adapter" in str(error):
            pytest.skip("Hardware graphics adapter not visible")
        raise
    if adapter is None or adapter.info["adapter_type"] == "CPU":
        pytest.skip("Hardware graphics adapter not visible")
    return adapter.request_device_sync()


def constant_sky(color=(1., 1., 1.)):
    pixels = np.ones((32, 64, 4), np.float32)
    pixels[..., :3] = color
    return pixels


def flat_frame(normal=(0., 0., 1.)):
    parameters = Wave_parameters(resolution=32, domain=64.)
    displacement = torch.zeros((32, 32, 4), dtype=torch.float32)
    displacement[..., 3] = -1.
    normals = torch.zeros_like(displacement)
    normals[..., :3] = torch.tensor(normal, dtype=torch.float32)
    return Wave_frame(parameters, 0., displacement, normals)


@pytest.fixture
def renderer_factory(device, monkeypatch):
    def make(pixels=None, *, samples=1, normal=(0., 0., 1.), foam_density=None):
        pixels = constant_sky() if pixels is None else pixels
        monkeypatch.setattr("encino_waves.render.load_sky", lambda _: (pixels.copy(), "test sky"))
        renderer = Ocean_renderer(device, mesh_resolution=(64, 48), transfer="host",
                                  sample_count=samples)
        frame = flat_frame(normal)
        renderer.upload(frame)
        if foam_density is not None:
            state = make_foam_state(frame.parameters, 0., Foam_parameters(resolution=32), "cpu")
            density = torch.tensor(foam_density, dtype=torch.float32)[:, None, None].expand(3, 32, 32).clone()
            renderer.upload_foam(replace(state, density=density))
        return renderer
    return make


CAMERA = Camera(x=3., y=-11., height=5., yaw=13., pitch=-14., fov=55.)
LOOK = Look(exposure=0., sky_rotation=0., sky_gain=1., haze=0., foam=0.)


def linear_hdr(renderer):
    """Read the real intermediate, before exposure and the display shoulder."""
    width, height = renderer.hdr.size[:2]
    stride = (width*8+255)//256*256
    data = renderer.device.queue.read_texture({"texture": renderer.hdr},
        {"bytes_per_row": stride, "rows_per_image": height}, (width, height, 1))
    return np.frombuffer(data, np.float16).reshape(height, stride//2)[:, :width*4].reshape(height, width, 4).astype(np.float32)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("normal,foam_density", [
    ((0., 0., 1.), None), ((0., 0., -1.), None),
    ((1., 0., 0.), None), ((0., 0., 0.), None),
    ((0., 0., 1.), (1., 2., 3.)), ((0., 0., -1.), (1., 2., 3.)),
])
def test_white_dome_has_nonnegative_bounded_radiance(renderer_factory, samples, normal, foam_density):
    renderer = renderer_factory(samples=samples, normal=normal, foam_density=foam_density)
    renderer.render_image(112, 80, CAMERA, replace(LOOK, foam=float(foam_density is not None)))
    hdr = linear_hdr(renderer)
    assert np.isfinite(hdr).all()
    assert hdr[..., :3].min() >= 0.
    # A passive water/foam layer cannot brighten a unit-radiance white dome.
    assert hdr[..., :3].max() <= 1.005
    np.testing.assert_array_equal(hdr[..., 3], 1.)
    # The fixture includes actual water, rather than accidentally testing only sky.
    assert hdr[-20:, :, :3].mean() < .95


def test_black_dome_does_not_emit_light(renderer_factory):
    renderer = renderer_factory(constant_sky((0., 0., 0.)), foam_density=(1., 2., 3.))
    image = renderer.render_image(112, 80, CAMERA, replace(LOOK, foam=1.))
    np.testing.assert_array_equal(linear_hdr(renderer)[..., :3], 0.)
    np.testing.assert_array_equal(image[..., :3], 0)


@pytest.mark.parametrize("samples", [1, 4])
def test_hdr_above_half_float_range_survives_exposure(renderer_factory, samples):
    """A low camera exposure must recover HDR values above 65504, not NaNs."""
    color = np.array((.5, .8, 1.), np.float32)
    reference = renderer_factory(constant_sky(color), samples=samples)
    bright = renderer_factory(constant_sky(color*100000.), samples=samples)
    expected = reference.render_image(112, 80, CAMERA,
        replace(LOOK, exposure=math.log2(100000.)-17.))
    actual = bright.render_image(112, 80, CAMERA, replace(LOOK, exposure=-17.))
    hdr = linear_hdr(bright)
    assert np.isfinite(hdr).all()
    assert hdr[..., :3].min() >= 0.
    assert actual[..., :3].max() > 128
    np.testing.assert_allclose(actual.astype(float), expected.astype(float), atol=2.)


def test_sky_and_camera_rotation_preserve_the_image(renderer_factory):
    theta = (np.arange(32)+.5)*math.pi/32
    phi = (np.arange(64)+.5)*2*math.pi/64-math.pi
    pixels = constant_sky()
    pixels[..., 0] = .5+.3*np.sin(theta)[:, None]*np.cos(phi)[None, :]
    pixels[..., 1] = .5+.3*np.sin(theta)[:, None]*np.sin(phi)[None, :]
    pixels[..., 2] = .5+.3*np.cos(theta)[:, None]
    renderer = renderer_factory(pixels)
    renderer.render_image(112, 80, CAMERA, LOOK)
    before = linear_hdr(renderer)
    # Increasing Maya yaw rotates world directions clockwise; rotating the
    # environment by the same amount must leave all relative angles unchanged.
    rotated = replace(CAMERA, x=CAMERA.y, y=-CAMERA.x, yaw=CAMERA.yaw+90.)
    renderer.render_image(112, 80, rotated, replace(LOOK, sky_rotation=90.))
    np.testing.assert_allclose(linear_hdr(renderer), before, rtol=.005, atol=.002)


def test_wave_normal_bends_the_prefiltered_reflection(renderer_factory):
    # Encode sky directions in RGB, then remove body lighting so this measures
    # reflection alone. Opposite surface slopes should look into opposite sides
    # of the sky even when the camera looks straight down.
    theta=(np.arange(64)+.5)*math.pi/64
    phi=(np.arange(128)+.5)*2*math.pi/128-math.pi
    pixels=np.ones((64,128,4),np.float32)
    pixels[...,0]=.5+.4*np.sin(theta)[:,None]*np.cos(phi)[None,:]
    pixels[...,1]=.5+.4*np.sin(theta)[:,None]*np.sin(phi)[None,:]
    pixels[...,2]=.5+.4*np.cos(theta)[:,None]
    colors=[]
    for tilt in (-.4,.4):
        renderer=renderer_factory(pixels,normal=(tilt,0.,math.sqrt(1-tilt*tilt)))
        renderer.lighting=replace(renderer.lighting,
            ambient_sh=np.zeros_like(renderer.lighting.ambient_sh),
            direct_irradiance=np.zeros_like(renderer.lighting.direct_irradiance))
        renderer.render_image(65,65,replace(CAMERA,pitch=-90.),LOOK)
        colors.append(linear_hdr(renderer)[32,32,:3])
    negative,positive=colors
    # Reflection bends through twice the surface tilt. Sampling the flat normal
    # (or the normal direction itself) cannot produce this directional contrast.
    assert (positive[0]-negative[0])/(positive[0]+negative[0]) > .45
    np.testing.assert_allclose(positive[1:],negative[1:],rtol=.02,atol=1e-5)


@pytest.mark.parametrize("samples", [1, 4])
def test_compact_hdr_source_adds_finite_positive_scattering(renderer_factory, samples):
    theta = (np.arange(128)+.5)*math.pi/128
    phi = (np.arange(256)+.5)*2*math.pi/256-math.pi
    elevation, azimuth = math.radians(35.), math.radians(77.)
    cosine = (np.sin(theta)[:, None]*math.cos(elevation)*np.cos(phi[None, :]-azimuth)
              +np.cos(theta)[:, None]*math.sin(elevation))
    pixels = np.ones((128, 256, 4), np.float32)
    pixels[..., :3] = .03
    pixels[cosine > math.cos(math.radians(3.)), :3] = (100000., 80000., 50000.)
    # Submerged bubbles exercise scattering; zero surface density leaves its
    # radiance visible instead of covering the test with opaque foam.
    renderer = renderer_factory(pixels, samples=samples, foam_density=(0., 1., 1.))
    assert renderer.lighting.has_direct
    look = replace(LOOK, foam=1., exposure=-12.)
    renderer.render_image(112, 80, CAMERA, look)
    lit = linear_hdr(renderer)
    assert np.isfinite(lit).all()
    assert lit[..., :3].min() >= 0.
    assert lit[..., :3].max() <= 32000.*1.01
    lighting = renderer.lighting
    renderer.lighting = replace(lighting, direct_irradiance=np.zeros(3))
    renderer.render_image(112, 80, CAMERA, look)
    difference = lit[..., :3]-linear_hdr(renderer)[..., :3]
    assert difference.min() >= 0.
    assert difference[-20:].mean() > 1e-5
    # This also checks the compact source direction, separately from SH and
    # reflection orientation exercised by the broad-sky rotation test above.
    renderer.lighting = lighting
    rotated = replace(CAMERA, x=CAMERA.y, y=-CAMERA.x, yaw=CAMERA.yaw+90.)
    renderer.render_image(112, 80, rotated, replace(look, sky_rotation=90.))
    np.testing.assert_allclose(linear_hdr(renderer), lit, rtol=.006, atol=.003)


@pytest.mark.parametrize("samples", [1, 4])
def test_comparison_preserves_both_odd_width_viewports(renderer_factory, samples):
    left = renderer_factory(constant_sky((.7, .3, .15)), samples=samples)
    right = renderer_factory(constant_sky((.15, .3, .7)), samples=samples)
    expected_left = left.render_image(73, 80, CAMERA, LOOK)
    expected_right = right.render_image(74, 80, CAMERA, LOOK)
    combined = right.render_image(147, 80, CAMERA, LOOK, left_renderer=left)
    np.testing.assert_allclose(combined[:, :73].astype(float), expected_left.astype(float), atol=1.)
    np.testing.assert_allclose(combined[:, 73:].astype(float), expected_right.astype(float), atol=1.)


@pytest.mark.parametrize("samples", [1, 4])
def test_sky_gain_and_camera_exposure_have_matching_units(renderer_factory, samples):
    renderer = renderer_factory(constant_sky((.2, .4, .8)), samples=samples,
                                foam_density=(.5, 1., 2.))
    doubled_light = renderer.render_image(112, 80, CAMERA, replace(LOOK, sky_gain=2., foam=1.))
    doubled_exposure = renderer.render_image(112, 80, CAMERA, replace(LOOK, exposure=1., foam=1.))
    np.testing.assert_allclose(doubled_light.astype(float), doubled_exposure.astype(float), atol=1.)


def test_legacy_round_trip_restores_physical_targets_and_bindings(renderer_factory):
    renderer = renderer_factory(constant_sky((.2, .4, .8)), samples=4)
    before = renderer.render_image(112, 80, CAMERA, LOOK)
    renderer.render_image(112, 80, CAMERA, replace(LOOK, material="2015"))
    after = renderer.render_image(112, 80, CAMERA, LOOK)
    np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("pitch", [-45., -67., -80., -89., -90., -100., -125.])
@pytest.mark.parametrize("height", [.5, 5., 100.])
def test_downward_frustum_is_covered_by_water(renderer_factory, pitch, height):
    renderer=renderer_factory()
    camera=replace(CAMERA,height=height,pitch=pitch)
    renderer.render_image(160,96,camera,LOOK)
    hdr=linear_hdr(renderer)
    # Every ray intersects flat water well inside the far bound. A clipped grid
    # exposes the unit-white sky, especially in the near corners when tilted.
    assert np.isfinite(hdr).all()
    assert hdr[...,:3].max() < .9


@pytest.mark.parametrize("size", [(160,96), (257,96), (96,160)])
def test_legacy_material_covers_downward_views_in_different_aspects(renderer_factory, size):
    renderer=renderer_factory()
    look=replace(LOOK,material="2015",exposure=-2.)
    sky=renderer.render_image(*size,replace(CAMERA,pitch=80.),look)
    water=renderer.render_image(*size,replace(CAMERA,pitch=-80.),look)
    # Legacy shading uses a direct display target; compare to its own sky value.
    assert not np.any(np.all(water[...,:3]==sky[0,0,:3],axis=-1))
