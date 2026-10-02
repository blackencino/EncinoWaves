# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Record UI decisions separately from the time needed to render an ocean frame.

Snapshots contain desired controls, not wall-clock-derived ocean time. Playback
advances the simulation using its own fixed timestep; explicit time scrubs can
be represented by stepped ``seek_serial`` and ``seek_time`` snapshot values.
"""
from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .camera import Camera, interpolate_camera


MAX_REHEARSAL_SECONDS = 298.0


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _snapshot(values):
    if not isinstance(values, dict):
        raise ValueError("Timeline values must be a JSON object")

    def validate(value):
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("Timeline object keys must be strings")
            for item in value.values(): validate(item)
        elif isinstance(value, (list, tuple)):
            for item in value: validate(item)
        elif isinstance(value, float):
            if not math.isfinite(value): raise ValueError("Timeline values must be finite")
        elif value is not None and not isinstance(value, (str, int, bool)):
            raise ValueError("Timeline values must be JSON serializable")

    validate(values)
    # Normalize tuples to JSON arrays so equality and playback survive a save.
    result = json.loads(json.dumps(values, allow_nan=False))
    if "camera" in result:
        try: Camera(**result["camera"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid timeline camera: {error}") from error
    if "playing" in result and not isinstance(result["playing"], bool):
        raise ValueError("Timeline playing must be a boolean")
    if "speed" in result:
        if _finite_number(result["speed"], "Playback speed") < 0:
            raise ValueError("Playback speed must be nonnegative")
    return result


@dataclass(frozen=True)
class Timeline_sample:
    time: float
    values: dict

    def __post_init__(self):
        timestamp = _finite_number(self.time, "Sample time")
        if not 0 <= timestamp <= MAX_REHEARSAL_SECONDS:
            raise ValueError(f"Sample time must be between 0 and {MAX_REHEARSAL_SECONDS:g} seconds")
        object.__setattr__(self, "time", timestamp)
        object.__setattr__(self, "values", _snapshot(self.values))


@dataclass(frozen=True)
class Academy_timeline:
    initial_scene: str
    duration: float
    samples: tuple[Timeline_sample, ...]
    performance: dict | None = None
    _performance_controller: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.initial_scene, str) or not self.initial_scene.strip():
            raise ValueError("Timeline requires an initial scene checkpoint")
        duration = _finite_number(self.duration, "Timeline duration")
        if not 0 <= duration <= MAX_REHEARSAL_SECONDS:
            raise ValueError(f"Timeline duration must be between 0 and {MAX_REHEARSAL_SECONDS:g} seconds")
        samples = tuple(self.samples)
        if not samples or not all(isinstance(item, Timeline_sample) for item in samples):
            raise ValueError("Timeline requires an initial sample")
        if samples[0].time != 0:
            raise ValueError("The first timeline sample must be at time zero")
        if any(a.time >= b.time for a, b in zip(samples, samples[1:])):
            raise ValueError("Timeline sample times must be strictly increasing")
        if samples[-1].time > duration:
            raise ValueError("Timeline samples cannot exceed the duration")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "samples", samples)
        if self.performance is not None:
            from .academy_performance import Performance
            performance = _snapshot(self.performance)
            object.__setattr__(self, "performance", performance)
            object.__setattr__(self, "_performance_controller", Performance.from_dict(performance))

    def sample(self, time):
        """Return stepped UI targets with a continuous Maya camera path."""
        time = _finite_number(time, "Playback time")
        time = min(self.duration, max(0.0, time))
        index = max(0, bisect_right(self.samples, time, key=lambda item: item.time)-1)
        current = self.samples[index]
        values = deepcopy(current.values)
        if index+1 < len(self.samples) and time > current.time:
            following = self.samples[index+1]
            if "camera" in values and "camera" in following.values:
                amount = (time-current.time)/(following.time-current.time)
                values["camera"] = asdict(interpolate_camera(
                    Camera(**values["camera"]), Camera(**following.values["camera"]), amount))
        if self._performance_controller is not None:
            performance = self._performance_controller.sample(time)
            values["parameters"] = {**values.get("parameters", {}), **performance["parameters"]}
            values["performance_time"] = time
            values["performance_overlay"] = performance["overlay"]
            if "look" in performance:
                values["look"] = {**values.get("look", {}), **performance["look"]}
            if "camera" in performance: values["camera"] = performance["camera"]
        return values

    def to_dict(self):
        result = {"version": 1, "initial_scene": self.initial_scene,
                "duration": self.duration,
                "samples": [{"time": item.time, "values": deepcopy(item.values)}
                            for item in self.samples]}
        if self.performance is not None: result["performance"] = deepcopy(self.performance)
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)+"\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                    prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)


def load_timeline(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Unsupported Academy timeline version")
    try:
        return Academy_timeline(data["initial_scene"], data["duration"],
            tuple(Timeline_sample(item["time"], item["values"]) for item in data["samples"]),
            data.get("performance"))
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid Academy timeline structure") from error


def save_rehearsal_as(source, destination) -> Path:
    """Copy a take and its scene/foam checkpoints, preserving the original."""
    source, destination = Path(source), Path(destination)
    if destination.suffix.lower() != ".json":
        destination = destination.with_name(destination.name+".json")
    if source.resolve() == destination.resolve(): return destination
    timeline = load_timeline(source)
    initial = Path(timeline.initial_scene)
    if not initial.is_absolute(): initial = source.parent/initial
    scene = json.loads(initial.read_text())
    if not isinstance(scene, dict): raise ValueError("Initial scene must be a JSON object")
    destination.parent.mkdir(parents=True, exist_ok=True)
    assets = destination.parent/f"{destination.stem}_assets_{uuid.uuid4().hex[:12]}"
    assets.mkdir()
    try:
        for name in ("foam_state", "comparison_foam_state"):
            if scene.get(name):
                checkpoint = Path(scene[name])
                if not checkpoint.is_absolute(): checkpoint = initial.parent/checkpoint
                filename = f"{name}.npz"
                shutil.copyfile(checkpoint, assets/filename)
                scene[name] = filename
        initial_copy = assets/"initial_scene.json"
        initial_copy.write_text(json.dumps(scene, indent=2, allow_nan=False)+"\n")
        copied = replace(timeline, initial_scene=(Path(assets.name)/initial_copy.name).as_posix())
        copied.save(destination)
    except BaseException:
        # This unique directory was created by this operation alone; a failed
        # copy leaves the previous destination tape and all originals intact.
        shutil.rmtree(assets)
        raise
    return destination


def playback_delta(values, elapsed):
    """Advance ocean time from a renderer timestep, never rehearsal wall time."""
    elapsed = _finite_number(elapsed, "Playback elapsed time")
    speed = _finite_number(values.get("speed", 1.0), "Playback speed")
    if elapsed < 0 or speed < 0:
        raise ValueError("Playback elapsed time and speed must be nonnegative")
    return elapsed*speed if values.get("playing", True) else 0.0


class Rehearsal:
    """Collect changed snapshots and retain the start of each camera motion.

    Call append every rehearsal tick. Unchanged snapshots consume no tape
    space, except one hold anchor when the camera starts moving after a pause.
    """
    def __init__(self, initial_scene, initial_values=None):
        self.initial_scene = str(initial_scene)
        self.samples = []
        self._last_seen = None
        self.performance = None
        if initial_values is not None: self.append(0.0, initial_values)

    def append(self, timestamp, values):
        sample = Timeline_sample(timestamp, values)
        if self._last_seen is not None and sample.time < self._last_seen.time:
            raise ValueError("Rehearsal timestamps must not go backwards")
        if not self.samples:
            if sample.time != 0:
                raise ValueError("The first rehearsal sample must be at time zero")
            self.samples.append(sample)
            self._last_seen = sample
            return True
        previous = self._last_seen
        self._last_seen = sample
        if sample.values == self.samples[-1].values:
            return False
        # Without this anchor, a camera movement after ten quiet seconds would
        # interpolate across the entire ten-second hold during offline replay.
        if (previous.time > self.samples[-1].time
                and previous.values.get("camera") != sample.values.get("camera")):
            self.samples.append(previous)
        if sample.time == self.samples[-1].time:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        return True

    def set_performance(self, data):
        """Attach continuous input events instead of frame-sampled targets."""
        self.performance = _snapshot(data) if data is not None else None

    def finish(self, duration):
        duration = _finite_number(duration, "Timeline duration")
        if self._last_seen is not None and duration < self._last_seen.time:
            raise ValueError("Timeline duration cannot precede the last rehearsal tick")
        return Academy_timeline(self.initial_scene, duration, tuple(self.samples), self.performance)

    def save(self, path, duration):
        timeline = self.finish(duration)
        timeline.save(path)
        return timeline
