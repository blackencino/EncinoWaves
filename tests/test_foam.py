"""History, periodic diffusion, depth exchange, checkpoints and wave isolation."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from encino_waves.model import Wave_parameters, Wave_frame, make_initial_state, evaluate
from encino_waves.foam import (Foam_parameters, make_foam_state, step_foam, update_foam,
    diffuse_and_decay, foam_reset_reason, save_foam, load_foam, prepare_foam, advance_foam_to, emission_mask)


@pytest.fixture(params=("cpu","mps","cuda"))
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available(): pytest.skip("Metal not visible")
    if request.param == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA not visible")
    return request.param


def source(time, device, crest=2, parameters=None):
    p = parameters or Wave_parameters(resolution=32)
    values = torch.zeros((p.resolution,p.resolution,4),device=device)
    values[...,3] = crest
    return Wave_frame(p,time,values,torch.zeros_like(values))


def test_periodic_diffusion_preserves_mass_and_wraps_edges(device):
    p = Foam_parameters(resolution=32, surface_half_life=1e15,shallow_half_life=1e15,deep_half_life=1e15,exchange=0,diffusion=1)
    state = make_foam_state(Wave_parameters(resolution=32,domain=32),0,p,device)
    density = torch.zeros_like(state.density)
    density[:,0,0] = 1
    old = density.clone()
    diffused = diffuse_and_decay(density,state.basis,p,.2)
    torch.testing.assert_close(density,old,atol=0,rtol=0)
    torch.testing.assert_close(diffused.sum(dim=(1,2)),torch.ones(3,device=device),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(diffused[:,0,1],diffused[:,0,-1],atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(diffused[:,1,0],diffused[:,-1,0],atol=2e-7,rtol=2e-6)
    assert torch.all(diffused[:,0,-1] > 0)
    shifted = diffuse_and_decay(torch.roll(density,(7,-9),(1,2)),state.basis,p,.2)
    torch.testing.assert_close(shifted,torch.roll(diffused,(7,-9),(1,2)),atol=2e-7,rtol=2e-5)


def test_independent_decay_and_shallow_to_deep_exchange(device):
    p = Foam_parameters(resolution=32,diffusion=0,exchange=.2,surface_half_life=2,shallow_half_life=4,deep_half_life=8)
    state = make_foam_state(Wave_parameters(resolution=32),0,p,device)
    density = torch.ones_like(state.density)
    t = .23
    result = diffuse_and_decay(density,state.basis,p,t)
    a,b,c = math.log(2)/2,math.log(2)/4+.2,math.log(2)/8
    expected = (math.exp(-a*t),math.exp(-b*t),math.exp(-c*t)+.2*(math.exp(-c*t)-math.exp(-b*t))/(b-c))
    torch.testing.assert_close(result.mean(dim=(1,2)),torch.tensor(expected,device=device),atol=1e-6,rtol=1e-6)
    # The aging/exchange operator composes, independently of display frame rate.
    twice = diffuse_and_decay(diffuse_and_decay(density,state.basis,p,t/2),state.basis,p,t/2)
    torch.testing.assert_close(twice,result,atol=1e-6,rtol=1e-6)
    same_rates = replace(p,exchange=0,shallow_half_life=8)
    assert torch.isfinite(diffuse_and_decay(density,state.basis,same_rates,t)).all()


def test_emission_has_history_and_does_not_change_wave_fields(device):
    p = Foam_parameters(resolution=32,breakup=0,diffusion=0,exchange=0,
                        surface_half_life=1e15,shallow_half_life=1e15,deep_half_life=1e15)
    state = make_foam_state(Wave_parameters(resolution=32),0,p,device)
    calm = source(.1,device,crest=-1)
    assert torch.count_nonzero(step_foam(state,calm).density) == 0
    frame = source(.1,device)
    original = frame.displacement.clone()
    emitted = step_foam(state,frame)
    assert torch.count_nonzero(state.density) == 0
    torch.testing.assert_close(frame.displacement,original,atol=0,rtol=0)
    expected = torch.tensor((.8,.2,0),device=device)*p.emission*.1
    torch.testing.assert_close(emitted.density.mean(dim=(1,2)),expected,atol=1e-7,rtol=1e-6)
    lingered = step_foam(emitted,source(.2,device,crest=-1))
    torch.testing.assert_close(lingered.density,emitted.density,atol=1e-7,rtol=1e-6)
    assert step_foam(emitted,frame) is emitted  # Paused drawing cannot deposit twice.


def test_reset_policy_preserves_rotation_and_resets_large_edits(device):
    p = Foam_parameters(resolution=32)
    waves = Wave_parameters(resolution=32)
    state = make_foam_state(waves,0,p,device)
    state = step_foam(state,source(.1,device))
    for changes in ({"wind_direction":127},{"resolution":64},{"wind_speed":18},{"depth":95}):
        assert foam_reset_reason(state,replace(waves,**changes),p) is None
    for changes in ({"seed":8},{"domain":100},{"depth":3},{"wind_speed":30},{"swell":1},{"spreading":"hasselmann"}):
        frame = source(.2,device,parameters=replace(waves,**changes))
        reset = update_foam(state,frame,p)
        assert reset.time == .2 and torch.count_nonzero(reset.density) == 0
    for t in (-1,5):
        assert torch.count_nonzero(update_foam(state,source(t,device),p).density) == 0
    assert update_foam(state,source(.2,device),replace(p,enabled=False)) is None


def test_flat_display_crest_map_does_not_amplify_sub_texel_residuals(device):
    frame=source(.1,device,crest=-1)
    frame.displacement[...,3]+=torch.linspace(-1e-6,1e-6,32,device=device)
    mask=emission_mask(frame,Foam_parameters(resolution=32),crest_gain=1e8,crest_bias=1e8)
    assert torch.count_nonzero(mask) == 0


def test_foam_checkpoint_replays_and_survives_wave_resolution_change(device,tmp_path):
    waves = make_initial_state(Wave_parameters(resolution=32),device)
    p = Foam_parameters(resolution=32)
    state = prepare_foam(waves,10,p,preroll=.2)
    repeated = prepare_foam(waves,10,p,preroll=.2)
    torch.testing.assert_close(state.density,repeated.density,atol=0,rtol=0)
    path = tmp_path/"foam.npz"
    save_foam(path,state)
    loaded = load_foam(path,device)
    torch.testing.assert_close(state.density,loaded.density,atol=0,rtol=0)
    a = advance_foam_to(state,waves,evaluate(waves,10.5),p)
    b = advance_foam_to(loaded,waves,evaluate(waves,10.5),p)
    torch.testing.assert_close(a.density,b.density,atol=0,rtol=0)
    larger = make_initial_state(replace(waves.parameters,resolution=64),device)
    restored = prepare_foam(larger,10,p,path)
    torch.testing.assert_close(restored.density,state.density,atol=0,rtol=0)


@pytest.mark.parametrize("changes",({"resolution":100},{"diffusion":-1},{"emission":float("nan")},{"surface_half_life":0},{"breakup":2}))
def test_invalid_foam_controls(changes):
    with pytest.raises(ValueError): Foam_parameters(**changes)
