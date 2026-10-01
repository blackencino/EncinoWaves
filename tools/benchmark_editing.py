"""Measure a wind/depth edit and subsequent FFT work on the selected device."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from encino_waves.model import Wave_parameters, evaluate, synchronize, texture_arrays
from encino_waves.editing import make_wave_basis, state_from_basis, preserve_phase

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--device", default="auto")
parser.add_argument("--frames", type=int, default=30)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
if args.frames < 1:
    parser.error("frames must be positive")
p = Wave_parameters(resolution=args.resolution)
start = time.perf_counter()
basis = make_wave_basis(p, args.device)
synchronize(basis.device)
basis_seconds = time.perf_counter()-start
state = state_from_basis(basis, p)
for i in range(3):
    state = preserve_phase(state, state_from_basis(basis, replace(p, depth=90-i)), 10+i/60)
    evaluate(state, 10+i/60)
synchronize(basis.device)
spectral_times, frame_times = [], []
for i in range(args.frames):
    parameters = replace(p, wind_speed=17+i*.03, depth=80-i*.02)
    t = 11+i/60
    start = time.perf_counter()
    state = preserve_phase(state, state_from_basis(basis, parameters), t)
    synchronize(basis.device)
    spectral_times.append(1000*(time.perf_counter()-start))
    start = time.perf_counter()
    frame = evaluate(state, t)
    synchronize(basis.device)
    frame_times.append(1000*(time.perf_counter()-start))
start = time.perf_counter()
maps = texture_arrays(frame)
readback_ms = 1000*(time.perf_counter()-start)
result = {
    "device": str(basis.device), "resolution": p.resolution, "frames": args.frames,
    "basis_seconds": basis_seconds, "distinct_radial_wavenumbers": len(basis.radial_k),
    "spectrum_and_phase_median_ms": float(np.median(spectral_times)),
    "spectrum_and_phase_p95_ms": float(np.percentile(spectral_times, 95)),
    "propagation_fft_normals_median_ms": float(np.median(frame_times)),
    "texture_readback_ms": readback_ms,
}
print(json.dumps(result, indent=2))
if args.output:
    args.output.write_text(json.dumps(result, indent=2)+"\n")
