"""Real-GPU radiance, orientation and seam checks for GGX sky filtering."""
import math

import numpy as np
import pytest
import wgpu

from encino_waves.environment import prefilter_environment


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


def source_texture(device, pixels):
    """Build ordinary radiance mips independently on CPU for small fixtures."""
    height, width = pixels.shape[:2]
    levels = int(math.log2(max(height, width)))+1
    texture = device.create_texture(size=(width, height, 1), format="rgba16float",
        mip_level_count=levels,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING
              | wgpu.TextureUsage.COPY_SRC)
    current = pixels.astype(np.float32)
    for level in range(levels):
        h, w = current.shape[:2]
        packed = current.astype(np.float16)
        device.queue.write_texture({"texture":texture, "mip_level":level}, packed,
            {"bytes_per_row":w*8, "rows_per_image":h}, (w, h, 1))
        if w > 1:
            current = .5*(current[:, 0::2]+current[:, 1::2])
        if h > 1:
            current = .5*(current[0::2]+current[1::2])
    return texture


def read_mip(device, texture, level):
    width, height = max(1, texture.size[0]>>level), max(1, texture.size[1]>>level)
    stride = (width*8+255)//256*256
    data = device.queue.read_texture({"texture":texture, "mip_level":level},
        {"bytes_per_row":stride, "rows_per_image":height}, (width, height, 1))
    return np.frombuffer(data, np.float16).reshape(height, stride//2)[:, :width*4].reshape(height, width, 4).astype(np.float32)


def latlong_directions(height, width):
    theta = (np.arange(height)+.5)*math.pi/height
    phi = (np.arange(width)+.5)*2*math.pi/width-math.pi
    return np.stack(np.broadcast_arrays(np.sin(theta)[:, None]*np.cos(phi)[None, :],
                                        np.sin(theta)[:, None]*np.sin(phi)[None, :],
                                        np.cos(theta)[:, None]), axis=-1)


@pytest.mark.parametrize("color", [(0., 0., 0.), (.125, .5, 2.), (60000., 20000., 1000.)])
def test_constant_radiance_is_preserved_at_every_roughness(device, color):
    pixels = np.ones((32, 64, 4), np.float32)
    pixels[..., :3] = color
    source = source_texture(device, pixels)
    filtered = prefilter_environment(device, source, max_width=64, sample_count=64)
    for level in range(filtered.mip_level_count):
        actual = read_mip(device, filtered, level)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual[..., :3], np.broadcast_to(color, actual[..., :3].shape),
                                   rtol=.0012, atol=1e-5)
        np.testing.assert_array_equal(actual[..., 3], 1)
    filtered.destroy()
    source.destroy()


def test_mip_zero_keeps_orientation_and_does_not_change_source(device):
    xyz = latlong_directions(32, 64)
    pixels = np.concatenate((.5+.4*xyz, np.ones((32, 64, 1))), axis=-1)
    source = source_texture(device, pixels)
    before = read_mip(device, source, 0)
    filtered = prefilter_environment(device, source, max_width=64, sample_count=64)
    np.testing.assert_allclose(read_mip(device, filtered, 0), before, atol=.001)
    np.testing.assert_array_equal(read_mip(device, source, 0), before)
    assert filtered is not source
    filtered.destroy()
    source.destroy()


@pytest.mark.parametrize("axis", [(-1., 0., 0.), (0., 0., 1.)])
def test_bright_source_remains_finite_bounded_and_filtered_at_seam_and_pole(device, axis):
    xyz = latlong_directions(64, 128)
    pixels = np.ones((64, 128, 4), np.float32)
    pixels[..., :3] = [.05, .1, .2]
    cap = xyz @ np.asarray(axis) > math.cos(math.radians(8))
    pixels[cap, :3] += [20, 10, 4]
    source = source_texture(device, pixels)
    filtered = prefilter_environment(device, source, max_width=128, sample_count=128)
    for level in range(filtered.mip_level_count):
        actual = read_mip(device, filtered, level)
        assert np.isfinite(actual).all()
        assert actual[..., :3].min() >= .049
        assert np.all(actual[..., :3] <= np.array([20.05, 10.1, 4.2])*1.002)
    # A rough reflection spreads the compact cap over more angular area.
    sharp, rough = read_mip(device, filtered, 0), read_mip(device, filtered, 3)
    assert rough[..., 0].max() < sharp[..., 0].max()*.75
    # Both sides of the longitude seam sample the same continuous radiance.
    # Compare symmetric seam texels while there is adequate angular resolution.
    for level in (0, 1, 2):
        actual = read_mip(device, filtered, level)
        np.testing.assert_allclose(actual[:, 0, :3], actual[:, -1, :3], rtol=.08, atol=.03)
    filtered.destroy()
    source.destroy()


def test_output_reduces_large_source_to_requested_base_without_upscaling(device):
    source = source_texture(device, np.ones((64, 128, 4), np.float32))
    filtered = prefilter_environment(device, source, max_width=32, sample_count=32)
    assert filtered.size == (32, 16, 1)
    assert filtered.mip_level_count == 6
    for level in range(filtered.mip_level_count):
        np.testing.assert_allclose(read_mip(device, filtered, level), 1, atol=.001)
    filtered.destroy()
    source.destroy()
