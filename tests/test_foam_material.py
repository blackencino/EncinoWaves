"""Offscreen GPU contracts for coverage filtering, material aging and resource reuse."""
from unittest.mock import patch

import numpy as np
import pytest
import wgpu

from encino_waves.foam_material import Foam_material


@pytest.fixture(scope="module")
def device():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except RuntimeError:
        pytest.skip("Hardware graphics adapter unavailable")
    if adapter is None or adapter.info["adapter_type"] == "CPU":
        pytest.skip("Hardware graphics adapter unavailable")
    return adapter.request_device_sync()


def raw_texture(device, data):
    height, width, _ = data.shape
    texture = device.create_texture(size=(width, height, 1), format="rgba16float",
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    write_raw(device, texture, data)
    return texture


def write_raw(device, texture, data):
    height, width, _ = data.shape
    device.queue.write_texture({"texture": texture}, np.asarray(data, np.float16),
        {"bytes_per_row": width*8, "rows_per_image": height}, (width, height, 1))


def read_level(device, texture, level=0):
    width, height = (max(1, value >> level) for value in texture.size[:2])
    stride = (width*8+255)//256*256
    pixels = device.queue.read_texture({"texture": texture, "mip_level": level},
        {"bytes_per_row": stride, "rows_per_image": height}, (width, height, 1))
    return np.frombuffer(pixels, np.float16).reshape(height, stride//2)[:, :width*4].reshape(height, width, 4).astype(np.float32)


def update(device, material, texture, strength=1., **kwargs):
    encoder = device.create_command_encoder()
    result = material.update(encoder, texture, strength, **kwargs)
    device.queue.submit([encoder.finish()])
    return result


@pytest.mark.parametrize("pattern", ["checkerboard", "dense_patch"])
def test_area_and_material_moments_survive_every_mip(device, pattern):
    n = 32
    y, x = np.mgrid[:n, :n]
    deposits = ((x//4+y//4) % 2) if pattern == "checkerboard" else ((x >= 8) & (x < 24) & (y >= 4) & (y < 24))
    data = np.zeros((n, n, 4), np.float16)
    data[..., 0] = deposits*2
    data[..., 1] = data[..., 0]*.25
    data[..., 2] = deposits*(x/n)
    data[..., 3] = .5
    material = Foam_material(device)
    texture = update(device, material, raw_texture(device, data))
    base = read_level(device, texture)
    mean = base[..., :3].mean(axis=(0, 1))
    for level in range(texture.mip_level_count):
        pixels = read_level(device, texture, level)
        assert np.isfinite(pixels).all()
        assert np.all((pixels >= 0) & (pixels <= 1))
        assert np.all(pixels[..., 1:3] <= pixels[..., :1])
        np.testing.assert_allclose(pixels[..., :3].mean(axis=(0, 1)), mean, atol=6e-4, rtol=0)
    # Applying coverage only after density reduction would turn both these
    # partially covered surfaces fully white, since their mean R exceeds .5.
    assert data[..., 0].mean() > .5
    assert 0 < mean[0] < .85


@pytest.mark.parametrize("size", [1, 16])
def test_dry_wet_and_disabled_endpoints_are_finite(device, size):
    material = Foam_material(device)
    data = np.zeros((size, size, 4), np.float16)
    data[..., 1:3] = 60000  # Underwater air alone cannot create surface coverage.
    data[..., 3] = .5
    raw = raw_texture(device, data)
    dry = read_level(device, update(device, material, raw))
    np.testing.assert_array_equal(dry[..., :3], 0)
    data[..., 0] = 2
    data[..., 1] = .5
    data[..., 2] = 0
    write_raw(device, raw, data)
    wet = read_level(device, update(device, material, raw))
    np.testing.assert_array_equal(wet[..., :2], 1)
    np.testing.assert_array_equal(wet[..., 2], .5)
    disabled = read_level(device, update(device, material, raw, 0))
    np.testing.assert_array_equal(disabled[..., :3], 0)
    assert np.isfinite(disabled).all()


def test_fresh_material_connects_sheets_and_aging_opens_holes(device):
    material = Foam_material(device)
    # A 1x1 source removes interpolation/LOD from this material-only comparison.
    data = np.array([[[.3, .075, 0., 0.]]], np.float16)
    raw = raw_texture(device, data)
    fresh_dark = read_level(device, update(device, material, raw))[0, 0]
    data[..., 3] = 1
    write_raw(device, raw, data)
    fresh_light = read_level(device, update(device, material, raw))[0, 0]
    data[..., 2] = .3
    data[..., 3] = 0
    write_raw(device, raw, data)
    mature_dark = read_level(device, update(device, material, raw))[0, 0]
    data[..., 3] = 1
    write_raw(device, raw, data)
    mature_light = read_level(device, update(device, material, raw))[0, 0]
    assert fresh_dark[0] > mature_dark[0]
    assert mature_light[0]-mature_dark[0] > fresh_light[0]-fresh_dark[0]
    assert fresh_light[1]/fresh_light[0] > .99
    assert 0 < mature_light[1]/mature_light[0] < .3


def test_large_density_and_strength_stay_bounded(device):
    data = np.full((8, 8, 4), 60000, np.float16)
    material = Foam_material(device)
    result = read_level(device, update(device, material, raw_texture(device, data), 1e30))
    assert np.isfinite(result).all()
    assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_array_equal(result[..., 0], 1)


def test_repeated_updates_reuse_resources_and_never_read_the_gpu(device):
    data = np.full((16, 16, 4), .5, np.float16)
    data[..., 0] = 0
    raw = raw_texture(device, data)
    material = Foam_material(device)
    original = update(device, material, raw)
    data[..., 0] = 2
    write_raw(device, raw, data)
    with patch.object(device, "create_texture", side_effect=AssertionError("Texture allocation")), \
         patch.object(device, "create_bind_group", side_effect=AssertionError("Bind group allocation")), \
         patch.object(device, "create_sampler", side_effect=AssertionError("Sampler allocation")), \
         patch.object(device, "create_render_pipeline", side_effect=AssertionError("Pipeline allocation")), \
         patch.object(device.queue, "read_texture", side_effect=AssertionError("Texture readback")), \
         patch.object(device.queue, "read_buffer", side_effect=AssertionError("Buffer readback")):
        changed = update(device, material, raw)
        assert changed is original
        # A changed strength also keeps every texture, view and bind group.
        update(device, material, raw, .75)
    assert np.all(read_level(device, changed)[..., 0] > .99)


def test_new_raw_object_rebinds_and_resolution_changes_reallocate(device):
    material = Foam_material(device)
    old = update(device, material, raw_texture(device, np.zeros((8, 8, 4), np.float16)))
    same_size = np.full((8, 8, 4), .5, np.float16)
    same_size[..., 0] = 2
    rebound = update(device, material, raw_texture(device, same_size))
    assert rebound is old
    assert np.all(read_level(device, rebound)[..., 0] == 1)
    resized = update(device, material, raw_texture(device, np.zeros((1, 1, 4), np.float16)))
    assert resized is not old
    assert resized.size == (1, 1, 1) and resized.mip_level_count == 1
    np.testing.assert_array_equal(read_level(device, resized)[..., :3], 0)


@pytest.mark.parametrize("strength", [-1, float("nan"), float("inf"), 1e40])
def test_invalid_strength_is_rejected_before_encoding(device, strength):
    material = Foam_material(device)
    with pytest.raises(ValueError, match="strength"):
        material.update(None, None, strength)


def test_crest_exclusive_area_preserves_history_channels_and_survives_mips(device):
    y, x = np.mgrid[:32, :32]
    data = np.zeros((32, 32, 4), np.float16)
    data[..., 0] = ((x//8+y//8) % 2)*2
    data[..., 1] = data[..., 0]*.25
    data[..., 3] = .5
    raw = raw_texture(device, data)
    compressed = np.zeros_like(data)
    compressed[..., 3] = -.4
    crests = raw_texture(device, compressed)
    material = Foam_material(device)
    texture = update(device, material, raw)
    original = [read_level(device, texture, level) for level in range(texture.mip_level_count)]
    update(device, material, raw, crest_texture=crests)
    base = read_level(device, texture)
    assert .08 < base[..., 3].mean() < .5
    for level, before in enumerate(original):
        after = read_level(device, texture, level)
        np.testing.assert_array_equal(after[..., :3], before[..., :3])
        assert np.all(after[..., 3] >= 0)
        assert np.all(after[..., 0]+after[..., 3] <= 1.001)
        np.testing.assert_allclose(after[..., 3].mean(), base[..., 3].mean(), atol=6e-4, rtol=0)
    # Where compressed water already has opaque foam, there is no second layer.
    np.testing.assert_array_equal(base[..., 3][base[..., 0] == 1], 0)
    update(device, material, raw)
    np.testing.assert_array_equal(read_level(device, texture), original[0])


def test_narrow_crests_are_not_skipped_by_a_coarse_foam_grid(device):
    raw = raw_texture(device, np.full((16, 16, 4), (0., 0., 0., .5), np.float16))
    data = np.zeros((128, 128, 4), np.float16)
    crests = raw_texture(device, data)
    material = Foam_material(device)
    data[..., 3] = -.4
    write_raw(device, crests, data)
    texture = update(device, material, raw, crest_texture=crests)
    full_area = read_level(device, texture)[..., 3].mean()
    assert full_area > .2
    areas = []
    # Move a one-wave-texel crest across all eight phases of a history texel.
    # Fixed 4x4 quadrature misses every other phase completely.
    for column in range(8):
        data[..., 3] = -1
        data[:, column, 3] = -.4
        write_raw(device, crests, data)
        texture = update(device, material, raw, crest_texture=crests)
        areas.append(read_level(device, texture)[..., 3].mean())
    np.testing.assert_allclose(areas, np.full(8, full_area/128), atol=1e-5, rtol=0)


def test_crest_binding_updates_without_new_history_and_reuses_steady_resources(device):
    data = np.full((16, 16, 4), (0., 0., 0., .5), np.float16)
    raw = raw_texture(device, data)
    data[..., 3] = -.4
    crests = raw_texture(device, data)
    material = Foam_material(device)
    texture = update(device, material, raw, crest_texture=crests)
    assert read_level(device, texture)[..., 3].mean() > .2
    data[..., 3] = -1
    write_raw(device, crests, data)
    with patch.object(device, "create_texture", side_effect=AssertionError("Texture allocation")), \
         patch.object(device, "create_bind_group", side_effect=AssertionError("Bind group allocation")), \
         patch.object(device.queue, "read_texture", side_effect=AssertionError("Texture readback")), \
         patch.object(device.queue, "read_buffer", side_effect=AssertionError("Buffer readback")):
        assert update(device, material, raw, crest_texture=crests) is texture
    np.testing.assert_array_equal(read_level(device, texture)[..., 3], 0)
    # Resolution changes replace the wave texture, without changing RGB history.
    replacement = raw_texture(device, np.full((32, 32, 4), (0., 0., 0., -.4), np.float16))
    assert update(device, material, raw, crest_texture=replacement) is texture
    assert read_level(device, texture)[..., 3].mean() > .2


@pytest.mark.parametrize("grain", [0., .5, 1.])
def test_extreme_grain_cannot_create_crest_on_uncompressed_water(device, grain):
    raw = raw_texture(device, np.full((16, 16, 4), (0., 0., 0., grain), np.float16))
    crests = raw_texture(device, np.full((32, 32, 4), (0., 0., 0., -1.), np.float16))
    texture = update(device, Foam_material(device), raw, crest_texture=crests)
    np.testing.assert_array_equal(read_level(device, texture), 0)
