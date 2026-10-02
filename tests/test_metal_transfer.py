"""Real GPU handoff: precision, ordering, lifetime, resize and host-copy guards."""
from dataclasses import astuple, replace
import gc
from unittest.mock import patch
import numpy as np
import pytest
import torch
import wgpu
from encino_waves.model import Wave_parameters, Wave_frame, make_initial_state, evaluate
from encino_waves.foam import Foam_parameters, make_foam_state, prepare_foam
from encino_waves.render import Ocean_renderer, make_device


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(),reason="Metal not visible")


@pytest.fixture(scope="module")
def device():
    return make_device()


def renderer(device,transfer):
    sky = np.ones((16,32,4),np.float32)
    sky[...,:3] = (.2,.4,.7)
    with patch("encino_waves.render.load_sky",return_value=(sky,"Test sky")):
        return Ocean_renderer(device,mesh_resolution=(64,48),transfer=transfer)


def read_texture(device,texture):
    width,height,_ = texture.size
    stride = (width*8+255)//256*256
    data = device.queue.read_texture({"texture":texture},
        {"bytes_per_row":stride,"rows_per_image":height},(width,height,1))
    return np.frombuffer(data,np.float16).reshape(height,stride//2)[:,:width*4].reshape(height,width,4).copy()


def test_exact_wave_and_foam_packing_including_storage_offsets(device):
    p = Wave_parameters(resolution=64)
    rng = np.random.default_rng(24)
    values = rng.normal(size=(2,64,64,4)).astype(np.float32)
    values.reshape(-1)[:6] = (0,1e-8,2**-24,-2**-24,1.00048828125,65504)
    storage = torch.from_numpy(values).to("mps")
    frame = Wave_frame(p,12,storage[0],storage[1])
    gpu = renderer(device,"metal")
    host = renderer(device,"host")
    gpu.upload(frame)
    host.upload(frame)
    foam = make_foam_state(p,12,Foam_parameters(resolution=32),"mps")
    density = torch.from_numpy(rng.uniform(0,2,size=(4,32,32)).astype(np.float32)).to("mps")[1:]
    density[0,0,0] = 70000
    foam = replace(foam,density=density)
    gpu.upload_foam(foam)
    host.upload_foam(foam)
    gpu.gpu_transfer.before_graphics()
    for actual,expected in zip(gpu.wave_textures+[gpu.foam_texture],host.wave_textures+[host.foam_texture]):
        np.testing.assert_array_equal(read_texture(device,actual),read_texture(device,expected))
    np.testing.assert_allclose(astuple(gpu.shading_statistics),astuple(host.shading_statistics),rtol=2e-6,atol=1e-8)


def test_render_matches_and_animation_never_reads_fields_to_cpu(device):
    state = make_initial_state(Wave_parameters(resolution=128),"mps")
    frame = evaluate(state,12)
    foam = prepare_foam(state,12,Foam_parameters(resolution=64),preroll=.3)
    gpu,host = renderer(device,"metal"),renderer(device,"host")
    gpu.upload(frame)
    host.upload(frame)
    np.testing.assert_allclose(astuple(gpu.shading_statistics),astuple(host.shading_statistics),rtol=2e-6,atol=1e-8)
    # Floating reduction order can differ by an ulp; isolate the field handoff.
    gpu.restore_shading_statistics(host.shading_statistics)
    next_frame = evaluate(state,12.1)
    with patch.object(torch.Tensor,"cpu",side_effect=AssertionError("Host readback")), \
         patch.object(torch.Tensor,"numpy",side_effect=AssertionError("NumPy packing")), \
         patch("encino_waves.render.texture_arrays",side_effect=AssertionError("Host field packing")):
        gpu.upload(next_frame)
        gpu.upload_foam(foam)
        actual = gpu.render_image(320,240)
    host.upload(next_frame)
    host.upload_foam(foam)
    expected = host.render_image(320,240)
    np.testing.assert_array_equal(actual,expected)


def test_gpu_statistics_read_only_three_scalars_and_preserve_flat_crests(device):
    gpu = renderer(device,"metal")
    p = Wave_parameters(resolution=64)
    values = torch.zeros((64,64,4),device="mps")
    values[...,3] = -1
    values[...,3] += torch.linspace(-1e-6,1e-6,64,device="mps")
    cpu = torch.Tensor.cpu
    sizes = []
    def read(tensor,*args,**kwargs):
        sizes.append(tensor.numel())
        return cpu(tensor,*args,**kwargs)
    with patch.object(torch.Tensor,"cpu",read):
        gpu.upload(Wave_frame(p,0,values,values))
    assert sizes == [3]
    assert astuple(gpu.shading_statistics) == (.001,1e8,1e8)


def test_ordering_reallocation_and_tensor_lifetime_under_queued_work(device):
    gpu = renderer(device,"metal")
    saved = []
    for i in range(16):
        n = 64 if i%4 < 2 else 128
        p = Wave_parameters(resolution=n)
        # Ephemeral storage is intentionally discarded/reused across frames.
        values = torch.full((n,n,4),i/16,device="mps")
        gpu.upload(Wave_frame(p,i,values,values.clone()))
        gpu.gpu_transfer.before_graphics()
        texture = device.create_texture(size=(n,n,1),format="rgba16float",
            usage=wgpu.TextureUsage.COPY_SRC|wgpu.TextureUsage.COPY_DST)
        encoder = device.create_command_encoder()
        encoder.copy_texture_to_texture({"texture":gpu.wave_textures[0]}, {"texture":texture},(n,n,1))
        device.queue.submit([encoder.finish()])
        saved.append((texture,i/16))
        del values
        gc.collect()
    for texture,value in saved:
        np.testing.assert_array_equal(read_texture(device,texture),np.float16(value))
        texture.destroy()


def test_invalid_tensor_is_rejected_before_native_access(device):
    from encino_waves.metal_transfer import _buffer
    with pytest.raises(ValueError): _buffer(torch.zeros((8,8)))
    with pytest.raises(ValueError): _buffer(torch.zeros((8,8),device="mps",dtype=torch.float16))
    with pytest.raises(ValueError): _buffer(torch.zeros((8,8),device="mps").T)
