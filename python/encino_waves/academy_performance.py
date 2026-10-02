# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Wall-clock performance controls with analytically replayable inertia.

Only input edges are recorded. Velocity approaches a held direction in 300 ms
and decays after release in 450 ms; evaluating between events does not depend
on the preview's frame rate. The first three controls move in log space.
"""
from copy import deepcopy
from dataclasses import asdict
import math

from .camera import Camera, interpolate_camera


CONTROLS = {
    "wind": ("wind_speed", "Wind Speed", 1.0, 500.0, True, .16),
    "depth": ("depth", "Ocean Depth", .25, 1000.0, True, .17),
    "fetch": ("fetch_km", "Fetch", 1.0, 5000.0, True, .17),
    "swell": ("swell", "Swell", -1.0, 2.0, False, .22),
    "foam": ("foam_amount", "Foam", 0.0, 2.0, False, .15),
}
_MODES = {"w": "wind", "wind_speed": "wind", "d": "depth", "f": "fetch",
          "fetch_km": "fetch", "s": "swell", "m": "foam", "foam_amount": "foam", "e": "examples"}
_FIELDS = {control[0]: control for control in CONTROLS.values()}
ACCELERATION_SECONDS = .30
FRICTION_SECONDS = .45
IDLE_SECONDS = 1.5


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _quintic(amount):
    amount = min(1.0, max(0.0, amount))
    return amount**3*(10+amount*(-15+6*amount))


def _position(name, value):
    _, _, low, high, logarithmic, _ = _FIELDS[name]
    value = min(high, max(low, _number(value, name)))
    return math.log(value/low)/math.log(high/low) if logarithmic else (value-low)/(high-low)


def _value(name, position):
    _, _, low, high, logarithmic, _ = _FIELDS[name]
    position = min(1.0, max(0.0, position))
    if position == 0: return low
    if position == 1: return high
    return low*math.exp(position*math.log(high/low)) if logarithmic else low+position*(high-low)


def _free_motion(position, velocity, target, seconds, response):
    decay = math.exp(-seconds/response)
    return (position+target*seconds+(velocity-target)*response*(1-decay),
            target+(velocity-target)*decay)


def _motion(position, velocity, target, seconds, acceleration_seconds, friction_seconds):
    """Integrate one constant input, including exact turning/boundary times."""
    response = acceleration_seconds if target else friction_seconds
    if seconds <= 0: return position, velocity
    # A change of direction can hit a boundary before the velocity reverses.
    turn = None
    if target and velocity*target < 0:
        turn = -response*math.log(-target/(velocity-target))
    if turn is not None and 0 < turn < seconds:
        position, velocity = _motion(position, velocity, target, turn, acceleration_seconds, friction_seconds)
        return _motion(position, velocity, target, seconds-turn, acceleration_seconds, friction_seconds)
    final, final_velocity = _free_motion(position, velocity, target, seconds, response)
    if 0 <= final <= 1:
        return final, final_velocity
    boundary = 0.0 if final < 0 else 1.0
    low, high = 0.0, seconds
    for _ in range(45):
        middle = (low+high)/2
        value, _ = _free_motion(position, velocity, target, middle, response)
        if (value < boundary) == (boundary == 0): high = middle
        else: low = middle
    remaining = seconds-high
    if (boundary == 0 and target > 0) or (boundary == 1 and target < 0):
        return _motion(boundary, 0.0, target, remaining, acceleration_seconds, friction_seconds)
    return boundary, 0.0


def _initial_state(parameters):
    values = deepcopy(parameters)
    positions = {name: _position(name, values[name]) for name in _FIELDS if name in values}
    for name in positions:
        values[name] = min(_FIELDS[name][3], max(_FIELDS[name][2], values[name]))
    return {"time": 0.0, "parameters": values, "positions": positions,
            "velocities": {name: 0.0 for name in positions}, "mode": None,
            "overlay_mode": None, "entered": 0.0, "last_input": 0.0,
            "exit_time": None, "exit_opacity": 0.0,
            "held": {"left": False, "right": False}, "transition": None,
            "example_name": None, "example_time": None}


def _transition_values(transition, time):
    amount = _quintic((time-transition["start"])/transition["duration"])
    result = deepcopy(transition["from"])
    for name, target in transition["target"].items():
        original = transition["from"].get(name, target)
        if amount == 1:
            result[name] = deepcopy(target)
        elif name in _FIELDS:
            result[name] = _value(name, (1-amount)*_position(name, original)+amount*_position(name, target))
        elif (isinstance(original, (int, float)) and not isinstance(original, bool)
                and isinstance(target, (int, float)) and not isinstance(target, bool)
                and name not in ("resolution", "seed", "trough_filter_revision")):
            distance = target-original
            if name == "wind_direction": distance = (distance+180)%360-180
            result[name] = original+amount*distance
        elif amount == 1:
            result[name] = deepcopy(target)
    return result


def _advance(state, time, acceleration_seconds, friction_seconds):
    transition = state["transition"]
    if transition is not None:
        end = transition["start"]+transition["duration"]
        state["parameters"] = _transition_values(transition, time)
        state["positions"] = {name: _position(name, state["parameters"][name])
                              for name in state["positions"]}
        state["velocities"] = {name: 0.0 for name in state["positions"]}
        state["time"] = min(time, end)
        if time < end: return
        state["transition"] = None
    elapsed = time-state["time"]
    direction = int(state["held"]["right"])-int(state["held"]["left"])
    active = CONTROLS.get(state["mode"])
    for name in state["positions"]:
        control = _FIELDS[name]
        target = direction*control[5] if active and active[0] == name else 0.0
        if not elapsed or (not target and not state["velocities"][name]): continue
        position, velocity = _motion(state["positions"][name], state["velocities"][name], target, elapsed,
                                     acceleration_seconds, friction_seconds)
        state["positions"][name], state["velocities"][name] = position, velocity
        state["parameters"][name] = _value(name, position)
    state["time"] = time


def _overlay(state, time):
    mode = state["overlay_mode"]
    if mode is None:
        return {"mode": None, "label": "", "opacity": 0.0, "left_weight": 0.0,
                "right_weight": 0.0, "value": None}
    opacity = _quintic((time-state["entered"])/.16)
    if state["mode"] is None:
        opacity = state["exit_opacity"]*(1-_quintic((time-state["exit_time"])/.22))
    elif mode == "examples" and state["example_name"]:
        opacity *= 1-_quintic((time-state["example_time"]-3.0)/.4)
    elif not any(state["held"].values()):
        opacity *= 1-_quintic((time-state["last_input"]-IDLE_SECONDS)/.3)
    if mode == "examples":
        return {"mode": mode, "label": state["example_name"] or "Examples", "opacity": opacity,
                "left_weight": 0.0, "right_weight": 0.0, "value": None}
    name, label, _, _, _, rate = CONTROLS[mode]
    velocity = state["velocities"][name]/rate
    return {"mode": mode, "label": label, "opacity": opacity,
            "left_weight": min(1.0, max(0.0, -velocity)),
            "right_weight": min(1.0, max(0.0, velocity)), "value": state["parameters"][name]}


def _apply(state, event):
    kind, time = event["kind"], event["time"]
    if kind == "mode":
        mode = event["mode"]
        if mode == state["mode"]:
            state["exit_opacity"] = _overlay(state, time)["opacity"]
            state["mode"], state["exit_time"] = None, time
        else:
            state.update(mode=mode, overlay_mode=mode, entered=time, exit_time=None)
            if mode == "examples": state["example_name"] = None
        if mode != "examples": state["transition"] = None
    elif kind == "direction":
        state["held"][event["direction"]] = event["down"]
        if event["down"] and state["mode"] in CONTROLS: state["transition"] = None
    elif kind == "transition":
        state["transition"] = {"start": time, "duration": event["duration"],
                               "from": deepcopy(state["parameters"]), "target": deepcopy(event["target"])}
        state["velocities"] = {name: 0.0 for name in state["positions"]}
        state.update(mode="examples", overlay_mode="examples", entered=time,
                     example_name=event.get("name"), example_time=time, exit_time=None)
    state["last_input"] = time


class Performance:
    def __init__(self, initial_parameters, initial_state=None, *,
                 acceleration_seconds=ACCELERATION_SECONDS, friction_seconds=FRICTION_SECONDS,
                 initial_look=None):
        self.acceleration_seconds = _number(acceleration_seconds, "Acceleration response")
        self.friction_seconds = _number(friction_seconds, "Friction response")
        if self.acceleration_seconds <= 0 or self.friction_seconds <= 0:
            raise ValueError("Performance response times must be positive")
        self.initial_parameters = deepcopy(initial_parameters)
        self.initial_parameters.pop("foam_amount", None)
        self.initial_look = deepcopy(initial_look)
        internal = deepcopy(self.initial_parameters)
        if initial_look is not None:
            internal["foam_amount"] = min(2.0, max(0.0, _number(initial_look.get("foam", 1.0), "Foam amount")))
        self.initial_state = deepcopy(initial_state) if initial_state is not None else _initial_state(internal)
        # Older controller states contain only wave controls. Add foam only
        # when a look was explicitly recorded, preserving legacy look samples.
        if initial_look is not None:
            values = self.initial_state["parameters"]
            values.setdefault("foam_amount", internal["foam_amount"])
            self.initial_state["positions"].setdefault("foam_amount", _position("foam_amount", values["foam_amount"]))
            self.initial_state["velocities"].setdefault("foam_amount", 0.0)
            transition = self.initial_state.get("transition")
            if transition is not None: transition["from"].setdefault("foam_amount", values["foam_amount"])
        self.events = []
        self.camera_moves = []
        self._cached_state = deepcopy(self.initial_state)
        self._cached_events = 0

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 1: raise ValueError("Unsupported performance version")
        # Tapes made before response settings were stored used the lighter feel.
        result = cls(data["initial_parameters"], data.get("initial_state"),
                     acceleration_seconds=data.get("acceleration_seconds", .10),
                     friction_seconds=data.get("friction_seconds", .15),
                     initial_look=data.get("initial_look"))
        previous = -math.inf
        for event in data.get("events", []):
            timestamp = _number(event["time"], "Event time")
            if timestamp < 0 or timestamp < previous: raise ValueError("Performance events must be ordered")
            previous = timestamp
            if event["kind"] not in ("mode", "direction", "transition"):
                raise ValueError("Unknown performance event")
        result.events = deepcopy(data.get("events", []))
        result.camera_moves = deepcopy(data.get("camera_moves", []))
        return result

    def _state_at(self, time):
        time = _number(time, "Performance time")
        if time < 0: raise ValueError("Performance time cannot be negative")
        if time < self._cached_state["time"]:
            self._cached_state = deepcopy(self.initial_state)
            self._cached_events = 0
        while self._cached_events < len(self.events) and self.events[self._cached_events]["time"] <= time:
            event = self.events[self._cached_events]
            _advance(self._cached_state, event["time"], self.acceleration_seconds, self.friction_seconds)
            _apply(self._cached_state, event)
            self._cached_events += 1
        # Cache only event boundaries: preview queries never alter integration.
        state = deepcopy(self._cached_state)
        _advance(state, time, self.acceleration_seconds, self.friction_seconds)
        return state

    def _event(self, kind, time, **values):
        time = _number(time, "Event time")
        if self.events and time < self.events[-1]["time"]:
            raise ValueError("Performance event time cannot go backwards")
        state = self._state_at(time)
        event = {"kind": kind, "time": time, "values": deepcopy(state["parameters"]), **values}
        self.events.append(event)
        return event

    def mode_key(self, mode, time):
        mode = _MODES.get(mode.lower(), mode.lower())
        if mode not in (*CONTROLS, "examples"): raise ValueError("Unknown performance mode")
        if mode == "foam" and "foam_amount" not in self.initial_state["positions"]:
            raise ValueError("Foam control requires initial_look")
        return self._event("mode", time, mode=mode)

    def direction_key(self, direction, down, time):
        direction = {"arrowleft": "left", "arrowright": "right"}.get(direction.lower(), direction.lower())
        if direction not in ("left", "right"): raise ValueError("Unknown performance direction")
        down = bool(down)
        if self._state_at(time)["held"][direction] == down: return None
        return self._event("direction", time, direction=direction, down=down)

    def transition(self, target_parameters, time, duration=2.5, name=None):
        duration = _number(duration, "Transition duration")
        if duration <= 0: raise ValueError("Transition duration must be positive")
        target = deepcopy(target_parameters)
        target.pop("foam_amount", None)  # Examples retain the performer's visual foam amount.
        for key in _FIELDS:
            if key in target:
                target[key] = min(_FIELDS[key][3], max(_FIELDS[key][2], _number(target[key], key)))
        return self._event("transition", time, duration=duration, target=target, name=name)

    def add_camera_move(self, start, end, startcam, endcam):
        start, end = _number(start, "Camera start"), _number(end, "Camera end")
        if start < 0 or end < start: raise ValueError("Camera move times must be ordered")
        if end == start: return
        startcam = asdict(startcam) if isinstance(startcam, Camera) else deepcopy(startcam)
        endcam = asdict(endcam) if isinstance(endcam, Camera) else deepcopy(endcam)
        Camera(**startcam), Camera(**endcam)
        # A newly started move supersedes scheduled future moves. Keep earlier
        # endpoints intact so their path before this start time cannot change.
        self.camera_moves = [move for move in self.camera_moves if move["start"] < start]
        self.camera_moves.append({"start": start, "end": end, "startcam": startcam, "endcam": endcam})

    def sample(self, time):
        state = self._state_at(time)
        result = {"parameters": deepcopy(state["parameters"]), "overlay": _overlay(state, time)}
        if "foam_amount" in result["parameters"]:
            result["look"] = {"foam": result["parameters"].pop("foam_amount")}
        result["overlay"]["active"] = state["mode"] is not None
        for move in self.camera_moves:
            if time < move["start"]: break
            amount = _quintic((time-move["start"])/(move["end"]-move["start"]))
            result["camera"] = asdict(interpolate_camera(Camera(**move["startcam"]), Camera(**move["endcam"]), amount))
        return result

    def _ease_boundaries(self):
        """Document each input's ease endpoints without frame-sampled targets."""
        boundaries = []
        for index, event in enumerate(self.events):
            start = event["time"]
            state = self._state_at(start)
            if event["kind"] == "transition":
                end = start+event["duration"]
            else:
                held = state["mode"] in CONTROLS and any(state["held"].values())
                end = start+6*(self.acceleration_seconds if held else self.friction_seconds)
            if index+1 < len(self.events): end = min(end, self.events[index+1]["time"])
            sampled = self.sample(end)
            start_values = deepcopy(event["values"])
            boundary = {"start": start, "end": end, "start_values": start_values,
                        "end_values": sampled["parameters"]}
            if "foam_amount" in start_values:
                boundary["start_look"] = {"foam": start_values.pop("foam_amount")}
                boundary["end_look"] = sampled["look"]
            boundaries.append(boundary)
        return boundaries

    def to_dict(self):
        return {"version": 1, "time_basis": "wall_clock_seconds", "initial_parameters": deepcopy(self.initial_parameters),
                "acceleration_seconds": self.acceleration_seconds, "friction_seconds": self.friction_seconds,
                "initial_look": deepcopy(self.initial_look),
                "initial_state": deepcopy(self.initial_state), "events": deepcopy(self.events),
                "camera_moves": deepcopy(self.camera_moves), "eases": self._ease_boundaries()}

    def recording_data(self, start, end=None):
        """Rebase a rehearsal while preserving any inertia already in flight."""
        start = _number(start, "Recording start")
        if end is not None and _number(end, "Recording end") < start:
            raise ValueError("Recording end precedes its start")
        state = self._state_at(start)
        for name in ("time", "entered", "last_input", "exit_time", "example_time"):
            if state[name] is not None: state[name] -= start
        if state["transition"] is not None: state["transition"]["start"] -= start
        events = []
        for event in self.events:
            if event["time"] <= start or (end is not None and event["time"] > end): continue
            event = deepcopy(event)
            event["time"] -= start
            events.append(event)
        moves = []
        for move in self.camera_moves:
            if move["end"] < start or (end is not None and move["start"] > end): continue
            move = deepcopy(move)
            move["start"] -= start
            move["end"] -= start
            moves.append(move)
        initial_parameters = deepcopy(state["parameters"])
        initial_look = deepcopy(self.initial_look)
        if "foam_amount" in initial_parameters:
            initial_look = {**(initial_look or {}), "foam": initial_parameters.pop("foam_amount")}
        return {"version": 1, "time_basis": "wall_clock_seconds", "initial_parameters": initial_parameters,
                "acceleration_seconds": self.acceleration_seconds, "friction_seconds": self.friction_seconds,
                "initial_look": initial_look,
                "initial_state": state, "events": events, "camera_moves": moves}


def sample_performance(data, time):
    """Sample a serialized performance; use Performance for repeated queries."""
    return Performance.from_dict(data).sample(time)
