from dataclasses import replace
import numpy as np
import torch
from encino_waves.model import Wave_parameters,make_initial_state,evaluate
from encino_waves.state_io import save_initial_state,load_initial_state


def test_lossless_spectral_snapshot(tmp_path):
    state=make_initial_state(Wave_parameters(resolution=32),"cpu")
    path=tmp_path/"state.npz"
    save_initial_state(path,state)
    loaded=load_initial_state(path,"cpu")
    for name in ("h_positive","h_negative","omega","multipliers"):
        torch.testing.assert_close(getattr(state,name),getattr(loaded,name),rtol=0,atol=0)
    torch.testing.assert_close(evaluate(state,12.5).displacement,evaluate(loaded,12.5).displacement,rtol=0,atol=0)
