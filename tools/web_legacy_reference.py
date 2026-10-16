#!/usr/bin/env python3
"""Generate the historical browser fixture using a native checkout with Phillips.

Example: python tools/web_legacy_reference.py --model-root /path/to/EncinoWaves
The model checkout must include Wave_parameters.tessendorf().
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-root', type=Path, default=ROOT)
args = parser.parse_args()
sys.path.insert(0, str(args.model_root / 'python'))
from encino_waves.model import Wave_parameters, make_initial_state, evaluate
from encino_waves.editing import preserve_phase

p = Wave_parameters(resolution=16, domain=128, seed=12345).tessendorf()
state = make_initial_state(p, 'cpu')
cases = []


def append(dt, parameters, time):
    frame = evaluate(state, time)
    cases.append(dict(dt=dt, parameters=parameters,
                      displacement=frame.displacement.flatten().tolist(),
                      normal=frame.normal.flatten().tolist()))


append(0, dict(domain=128, spectrum='phillips', seed=12345, phillips_amplitude=100), 0)
append(1, {}, 1)
p = replace(p, depth=2, fetch_km=1000, swell=1.5)
state = preserve_phase(state, make_initial_state(p, 'cpu'), 1)
append(0, dict(depth=2, fetch_km=1000, swell=1.5), 1)
append(.125, {}, 1.125)
p = replace(p, phillips_amplitude=200)
state = preserve_phase(state, make_initial_state(p, 'cpu'), 1.125)
append(0, dict(phillips_amplitude=200), 1.125)
path = ROOT / 'web' / 'tests' / 'legacy-reference.json'
path.write_text(json.dumps(dict(resolution=16, cases=cases), separators=(',', ':')) + '\n')
print(path)
