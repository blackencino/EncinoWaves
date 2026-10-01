from dataclasses import asdict, replace
import json
import numpy as np
import torch
from encino_waves.model import Wave_parameters,make_initial_state,evaluate
from encino_waves.state_io import save_initial_state,load_initial_state


def test_lossless_spectral_snapshot(tmp_path):
    state=make_initial_state(Wave_parameters(resolution=32,wind_direction=37),"cpu")
    path=tmp_path/"state.npz"
    save_initial_state(path,state)
    loaded=load_initial_state(path,"cpu")
    assert loaded.parameters == state.parameters
    for name in ("h_positive","h_negative","omega","multipliers"):
        torch.testing.assert_close(getattr(state,name),getattr(loaded,name),rtol=0,atol=0)
    torch.testing.assert_close(evaluate(state,12.5).displacement,evaluate(loaded,12.5).displacement,rtol=0,atol=0)


def test_old_snapshot_does_not_rotate_its_baked_spectrum(tmp_path):
    state=make_initial_state(Wave_parameters(resolution=32),"cpu")
    arrays={name:getattr(state,name).numpy() for name in ("h_positive","h_negative","omega","multipliers")}
    for version in (1,2):
        path=tmp_path/f"old_{version}.npz"
        np.savez(path,version=version,parameters=json.dumps(asdict(replace(state.parameters,wind_direction=90))),
                 phase_steps="[]",**arrays)
        loaded=load_initial_state(path,"cpu")
        assert loaded.parameters.wind_direction == 0
        torch.testing.assert_close(evaluate(loaded,12.5).displacement,evaluate(state,12.5).displacement,rtol=0,atol=0)
