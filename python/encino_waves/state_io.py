# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Portable, lossless spectral state snapshots for exact realization replay."""
from dataclasses import asdict
import json
import numpy as np
import torch
from .model import Wave_parameters, Initial_state, select_device


def save_initial_state(path,state):
    arrays={name:getattr(state,name).detach().cpu().numpy() for name in ("h_positive","h_negative","omega","multipliers")}
    np.savez_compressed(path,parameters=json.dumps(asdict(state.parameters)),version=1,**arrays)


def load_initial_state(path,device="auto"):
    device=select_device(str(device))
    with np.load(path,allow_pickle=False) as data:
        if int(data["version"])!=1: raise ValueError("Unsupported spectral snapshot version")
        p=Wave_parameters(**json.loads(str(data["parameters"])))
        shape=(p.resolution,p.resolution//2+1)
        tensors=[]
        for name in ("h_positive","h_negative","omega","multipliers"):
            array=data[name]
            expected=(6,*shape) if name=="multipliers" else shape
            dtype=np.float32 if name=="omega" else np.complex64
            if array.shape!=expected or array.dtype!=dtype or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name} in spectral snapshot")
            tensors.append(torch.from_numpy(array.copy()).to(device))
    return Initial_state(p,*tensors)
