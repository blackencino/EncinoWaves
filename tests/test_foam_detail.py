"""Static foam scale, isotropy and preserved slope/material energy."""
import numpy as np
import pytest
import wgpu

from encino_waves.foam_detail import detail_mips, Foam_detail, FOAM_DETAIL_WGSL


@pytest.fixture(scope="module")
def levels():
    return detail_mips()


def test_deterministic_bounded_material_detail_has_no_directional_streaks(levels):
    base = levels[0]
    np.testing.assert_array_equal(base, detail_mips()[0])
    assert not base.flags.writeable
    assert np.isfinite(base).all()
    assert np.abs(base[..., :2]).max() < .4
    assert base[..., 2].min() >= .92-1e-6 and base[..., 2].max() <= 1.08+1e-6
    # Isotropic cells should not acquire a privileged wind/texture axis.
    variance = np.square(base[..., :2]).mean(axis=(0, 1))
    assert .75 < variance[0]/variance[1] < 1.33
    assert abs(np.corrcoef(base[..., 0].ravel(), base[..., 1].ravel())[0, 1]) < .15
    assert not np.array_equal(base, detail_mips(seed=123)[0])


def test_every_mip_preserves_reflectance_and_total_slope_energy(levels):
    mean = levels[0].mean(axis=(0, 1), dtype=np.float64)
    for level in levels:
        np.testing.assert_allclose(level.mean(axis=(0, 1), dtype=np.float64), mean, atol=2e-7, rtol=0)
        missing = level[..., 3]-np.square(level[..., :2]).sum(axis=-1)
        assert missing.min() > -1e-8
    np.testing.assert_allclose(levels[-1][0, 0, :2], 0, atol=2e-8)
    assert levels[-1][0, 0, 2] == pytest.approx(1, abs=2e-7)
    assert levels[-1][0, 0, 3] > 0


def test_periodic_boundary_has_the_same_relief_statistics_as_the_interior(levels):
    base = levels[0]
    # Compare texture jumps across the wrap against ordinary neighboring texels.
    # First and last texels are centers, so they should not be duplicated.
    for axis in (0, 1):
        all_steps = np.diff(base[..., :3], axis=axis)
        seam = np.take(base[..., :3], 0, axis=axis)-np.take(base[..., :3], -1, axis=axis)
        rms = np.sqrt(np.square(all_steps).mean(axis=(0, 1)))
        seam_rms = np.sqrt(np.square(seam).mean(axis=0))
        assert np.all(seam_rms < 1.6*rms)


@pytest.mark.parametrize("kwargs", [{"resolution": 31}, {"resolution": 96}, {"period": 0}, {"period": float("nan")}])
def test_invalid_detail_parameters(kwargs):
    with pytest.raises(ValueError):
        detail_mips(**kwargs)


def test_static_gpu_upload_matches_mips_and_shared_shader_compiles():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except RuntimeError:
        pytest.skip("Hardware graphics adapter unavailable")
    if adapter is None or adapter.info["adapter_type"] == "CPU":
        pytest.skip("Hardware graphics adapter unavailable")
    device = adapter.request_device_sync()
    texture = Foam_detail(device, resolution=64)
    expected = detail_mips(resolution=64)
    for level, pixels in enumerate(expected):
        n = pixels.shape[0]
        stride = (n*8+255)//256*256
        raw = device.queue.read_texture({"texture": texture.texture, "mip_level": level},
            {"bytes_per_row": stride, "rows_per_image": n}, (n, n, 1))
        actual = np.frombuffer(raw, np.float16).reshape(n, stride//2)[:, :n*4].reshape(n, n, 4)
        np.testing.assert_array_equal(actual, pixels.astype(np.float16))
    shader = device.create_shader_module(code=FOAM_DETAIL_WGSL + """
        @group(0) @binding(0) var<storage,read_write> output: array<vec4f>;
        @compute @workgroup_size(1) fn check() {
            let detail=vec4f(.1,-.1,1.0,.025);
            let normal=foam_detail_normal(vec3f(0,0,1),detail,0.0);
            output[0]=vec4f(normal,foam_detail_variance(detail));
        }
    """)
    device.create_compute_pipeline(layout="auto", compute={"module": shader, "entry_point": "check"})
    texture.destroy()
