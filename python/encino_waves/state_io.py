# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Portable, lossless spectral state snapshots for exact realization replay."""
from dataclasses import asdict, replace
import json
import numpy as np
import torch
from .model import Wave_parameters, Initial_state, Phase_step, select_device


def save_initial_state(path,state):
    arrays={name:getattr(state,name).detach().cpu().numpy() for name in ("h_positive","h_negative","omega","multipliers")}
    if state.phase is not None:
        arrays["phase"]=state.phase.detach().cpu().numpy()
    np.savez_compressed(path,parameters=json.dumps(asdict(state.parameters)),version=3,
                        phase_steps=json.dumps([asdict(step) for step in state.phase_steps]),**arrays)


def load_initial_state(path,device="auto"):
    device=select_device(str(device))
    with np.load(path,allow_pickle=False) as data:
        version=int(data["version"])
        if version not in (1,2,3): raise ValueError("Unsupported spectral snapshot version")
        p=Wave_parameters.from_dict(json.loads(str(data["parameters"])))
        if version < 3:
            # Older snapshots already baked direction into the stored spectrum.
            # Preserve their exact field without applying a second rotation.
            p=replace(p,wind_direction=0.0)
        shape=(p.resolution,p.resolution//2+1)
        tensors=[]
        for name in ("h_positive","h_negative","omega","multipliers"):
            array=data[name]
            expected=(6,*shape) if name=="multipliers" else shape
            dtype=np.float32 if name=="omega" else np.complex64
            if array.shape!=expected or array.dtype!=dtype or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name} in spectral snapshot")
            tensors.append(torch.from_numpy(array.copy()).to(device))
        steps=tuple(Phase_step(**step) for step in json.loads(str(data["phase_steps"]))) if version>=2 else ()
        if ("phase" in data)!=bool(steps):
            raise ValueError("Spectral snapshot phase and phase steps must be present together")
        phase=None
        if steps:
            array=data["phase"]
            if array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():
                raise ValueError("Invalid phase in spectral snapshot")
            phase=torch.from_numpy(array.copy()).to(device)
    return Initial_state(p,*tensors,phase,steps)
