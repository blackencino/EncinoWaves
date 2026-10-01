"""History, periodic diffusion, depth exchange, checkpoints and wave isolation."""
from dataclasses import replace
import json
import math
import numpy as np
import pytest
import torch
from encino_waves.model import Wave_parameters, Wave_frame, make_initial_state, evaluate
from encino_waves.foam import (Foam_parameters, make_foam_state, step_foam, update_foam,
    diffuse_and_decay, foam_reset_reason, save_foam, load_foam, prepare_foam, advance_foam_to, emission_mask,
    windrow_activity, windrow_velocity, transport_windrows)


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
    for changes in ({"seed":8},{"domain":100},{"depth":3},{"wind_speed":30},{"swell":1},{"spreading":"donelan_banner"}):
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
    waves = make_initial_state(Wave_parameters(resolution=32,wind_speed=24,swell=.8),device)
    p = Foam_parameters(resolution=32,windrows=.45)
    state = prepare_foam(waves,10,p,preroll=.2)
    repeated = prepare_foam(waves,10,p,preroll=.2)
    torch.testing.assert_close(state.density,repeated.density,atol=0,rtol=0)
    torch.testing.assert_close(state.windrows,repeated.windrows,atol=0,rtol=0)
    assert torch.count_nonzero(state.windrows) > 0
    path = tmp_path/"foam.npz"
    save_foam(path,state)
    loaded = load_foam(path,device)
    torch.testing.assert_close(state.density,loaded.density,atol=0,rtol=0)
    torch.testing.assert_close(state.windrows,loaded.windrows,atol=0,rtol=0)
    a = advance_foam_to(state,waves,evaluate(waves,10.5),p)
    b = advance_foam_to(loaded,waves,evaluate(waves,10.5),p)
    torch.testing.assert_close(a.density,b.density,atol=0,rtol=0)
    torch.testing.assert_close(a.windrows,b.windrows,atol=0,rtol=0)
    larger = make_initial_state(replace(waves.parameters,resolution=64),device)
    restored = prepare_foam(larger,10,p,path)
    torch.testing.assert_close(restored.density,state.density,atol=0,rtol=0)
    torch.testing.assert_close(restored.windrows,state.windrows,atol=0,rtol=0)


def test_windrow_transport_wraps_and_concentrates_without_creating_mass(device):
    p = Foam_parameters(resolution=32,windrow_spacing=16)
    waves = Wave_parameters(resolution=32,domain=32,wind_speed=24,swell=.8)
    state = make_foam_state(waves,0,p,device)
    dye = torch.zeros((32,32),device=device)
    dye[0,-1] = 1
    original = dye.clone()
    moved = transport_windrows(dye,(1,torch.zeros_like(dye),0),1,.25)
    assert float(moved[0,0].cpu()) == .25
    assert float(moved[0,-1].cpu()) == .75
    torch.testing.assert_close(dye,original,atol=0,rtol=0)
    # Uniform residue must collect into bands: passive colour advection alone
    # would keep it uniform and could not model the effect we want.
    uniform = torch.ones_like(dye)
    velocity = windrow_velocity(state.basis,0,p,waves.wind_speed)
    collected = transport_windrows(uniform,velocity,.05,.25)
    assert torch.isfinite(collected).all() and torch.all(collected >= 0)
    assert float(collected.std().cpu()) > .1
    torch.testing.assert_close(collected.sum(),uniform.sum(),rtol=2e-6,atol=2e-5)
    assert torch.all(uniform == 1)


def test_low_frequency_warp_moves_existing_streaks_without_new_emission(device):
    p = Foam_parameters(resolution=32,windrow_gathering=0,windrow_warp=2)
    waves = Wave_parameters(resolution=32,domain=32,wind_speed=24,swell=.8)
    state = make_foam_state(waves,0,p,device)
    density = torch.zeros((32,32),device=device)
    density[16,:] = 1
    original = density.clone()
    warped = density
    still = density
    for i in range(20):
        velocity = windrow_velocity(state.basis,(i+.5)*.1,p,waves.wind_speed)
        warped = transport_windrows(warped,velocity,1,.1)
        velocity = windrow_velocity(state.basis,(i+.5)*.1,replace(p,windrow_warp=0),waves.wind_speed)
        still = transport_windrows(still,velocity,1,.1)
    torch.testing.assert_close(still,original,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(density,original,atol=0,rtol=0)
    torch.testing.assert_close(warped.sum(),original.sum(),atol=1e-5,rtol=2e-6)
    centroid = (warped*torch.arange(32,device=device)[:,None]).sum(dim=0)
    assert float(centroid.std().cpu()) > .05
    # Time variation is slow, rather than independent random offsets per frame.
    flow = windrow_velocity(state.basis,0,p,waves.wind_speed)[1]
    soon = windrow_velocity(state.basis,.1,p,waves.wind_speed)[1]
    later = windrow_velocity(state.basis,30,p,waves.wind_speed)[1]
    assert float((soon-flow).abs().max().cpu()) < .02
    assert float((later-flow).abs().max().cpu()) > .05


def test_streaks_need_strong_wind_and_swell_and_fade_after_gate_closes(device):
    p = Foam_parameters(resolution=32,breakup=0,windrows=.45)
    strong = Wave_parameters(resolution=32,wind_speed=24,swell=.8)
    for waves in (replace(strong,swell=.5),replace(strong,wind_speed=13.9)):
        state = make_foam_state(waves,0,p,device)
        assert step_foam(state,source(.1,device,parameters=waves)).windrows is None
    state = make_foam_state(strong,0,p,device)
    frame = source(.1,device,parameters=strong)
    emitted = step_foam(state,frame)
    # Windrow emission is taken from fresh surface foam, not added to the source.
    total = emitted.density.mean(dim=(1,2)).sum()+emitted.windrows.mean()
    torch.testing.assert_close(total,torch.tensor(p.emission*.1,device=device),atol=1e-7,rtol=1e-6)
    eased = source(.2,device,parameters=replace(strong,swell=.5))
    lingered = step_foam(emitted,eased)
    expected = emitted.windrows.sum()*math.exp(-math.log(2)*.1/p.windrow_half_life)
    torch.testing.assert_close(lingered.windrows.sum(),expected,atol=1e-6,rtol=2e-6)
    assert step_foam(emitted,frame) is emitted
    disabled = step_foam(emitted,frame,replace(p,windrows=0))
    assert disabled.windrows is None and disabled.density is emitted.density
    assert state.windrows is None and torch.count_nonzero(state.density) == 0


def test_windrow_gate_is_continuous_and_monotonic():
    p = Wave_parameters(wind_speed=24,swell=.8)
    assert windrow_activity(p) == 1
    assert windrow_activity(replace(p,swell=.5)) == 0
    assert windrow_activity(replace(p,swell=.5+1e-4)) < 1e-6
    for name,values in (("swell",np.linspace(.4,.9,20)),("wind_speed",np.linspace(12,24,20))):
        activity = [windrow_activity(replace(p,**{name:v})) for v in values]
        assert all(a <= b for a,b in zip(activity,activity[1:]))


def test_old_checkpoint_without_windrow_history_loads(tmp_path):
    state = make_foam_state(Wave_parameters(resolution=32),0,Foam_parameters(resolution=32),"cpu")
    path = tmp_path/"foam.npz"
    save_foam(path,state)
    with np.load(path,allow_pickle=False) as data:
        arrays = dict(data)
    arrays["version"] = 1
    saved = json.loads(str(arrays["parameters"]))
    arrays["parameters"] = json.dumps({name:value for name,value in saved.items() if not name.startswith("windrow")})
    np.savez_compressed(path,**arrays)
    loaded = load_foam(path,"cpu")
    assert loaded.windrows is None
    torch.testing.assert_close(state.density,loaded.density,atol=0,rtol=0)


@pytest.mark.parametrize("changes",({"resolution":100},{"diffusion":-1},{"emission":float("nan")},
    {"surface_half_life":0},{"breakup":2},{"windrows":float("nan")},{"windrow_spacing":0},
    {"windrow_gathering":-1},{"windrow_half_life":0},{"windrow_warp":-1},{"windrow_warp_period":0}))
def test_invalid_foam_controls(changes):
    with pytest.raises(ValueError): Foam_parameters(**changes)
