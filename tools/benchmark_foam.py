"""Measure GPU foam work separately from wave synthesis and graphics transfer."""
import argparse
import json
import time
import numpy as np
from encino_waves.model import Wave_parameters, evaluate, synchronize
from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, update_foam

parser=argparse.ArgumentParser()
parser.add_argument("--resolution",type=int,default=1024)
parser.add_argument("--foam-resolution",type=int,default=512)
parser.add_argument("--device",default="auto")
parser.add_argument("--frames",type=int,default=30)
args=parser.parse_args()
if args.frames < 1: parser.error("frames must be positive")
waves=make_edited_state(Wave_parameters(resolution=args.resolution),args.device)
p=Foam_parameters(resolution=args.foam_resolution)
foam=None
times=[]
for i in range(args.frames+3):
    frame=evaluate(waves,10+i/30)
    synchronize(waves.device)
    start=time.perf_counter()
    foam=update_foam(foam,frame,p,crest_gain=3,crest_bias=3)
    synchronize(waves.device)
    if i>=3: times.append((time.perf_counter()-start)*1000)
print(json.dumps({"device":str(waves.device),"waves":args.resolution,"foam":args.foam_resolution,
    "foam_step_median_ms":float(np.median(times)),"foam_step_p95_ms":float(np.percentile(times,95)),
    "density_bytes":foam.density.numel()*foam.density.element_size()},indent=2))
