#!/usr/bin/env python3
"""Regenerate the small browser GPU parity fixture with the CPU School model."""
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from encino_waves.model import Wave_parameters, make_initial_state, evaluate
from encino_waves.editing import preserve_phase


def main():
    p = Wave_parameters(resolution=16, domain=128)
    state = make_initial_state(p, "cpu")
    cases = []
    def append(dt, parameters, time):
        frame = evaluate(state, time)
        cases.append(dict(dt=dt, parameters=parameters,
                          displacement=frame.displacement.flatten().tolist(),
                          normal=frame.normal.flatten().tolist()))
    append(0, {"domain": 128}, 0)
    append(1, {}, 1)
    p = replace(p, depth=12, wind_speed=12, swell=.8)
    state = preserve_phase(state, make_initial_state(p, "cpu"), 1)
    append(0, {"depth": 12, "wind_speed": 12, "swell": .8}, 1)
    append(.125, {}, 1.125)
    result = dict(resolution=16, cases=cases)
    path = ROOT / "web" / "tests" / "reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(path)


if __name__ == "__main__":
    main()
