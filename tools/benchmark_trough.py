"""Measure synchronized wave propagation with spatial trough damping off/on."""
from dataclasses import replace
import argparse
import gc
import json
import time
import numpy as np
import torch
from encino_waves.model import Wave_parameters, evaluate, synchronize
from encino_waves.editing import make_edited_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--resolutions",type=int,nargs="+",default=(1024,2048,4096))
    parser.add_argument("--frames",type=int,default=10)
    args=parser.parse_args()
    if args.frames < 1: parser.error("frames must be positive")
    for n in args.resolutions:
        state=make_edited_state(Wave_parameters(resolution=n),args.device)
        row={"resolution":n,"device":str(state.device)}
        for amount in (0,.5):
            variant=replace(state,parameters=replace(state.parameters,trough_damping=amount))
            for i in range(3): evaluate(variant,10+i/24)
            synchronize(state.device)
            elapsed=[]
            for i in range(args.frames):
                start=time.perf_counter()
                frame=evaluate(variant,12+i/24)
                synchronize(state.device)
                elapsed.append((time.perf_counter()-start)*1000)
            assert bool(torch.isfinite(frame.displacement).all()) and bool(torch.isfinite(frame.normal).all())
            row["off" if amount==0 else "on"]={
                "median_ms":float(np.median(elapsed)),"p95_ms":float(np.percentile(elapsed,95))}
        print(json.dumps(row),flush=True)
        device=state.device
        del state,variant,frame
        gc.collect()
        if device.type=="mps": torch.mps.empty_cache()
        elif device.type=="cuda": torch.cuda.empty_cache()


if __name__ == "__main__": main()
