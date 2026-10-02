# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
from dataclasses import dataclass, replace
from .model import Wave_parameters
from .render import Camera, Look
from .camera import frame_domain, look_at
import math


@dataclass(frozen=True)
class Scene:
    name: str
    description: str
    parameters: Wave_parameters
    camera: Camera
    look: Look = Look()


def _scene(name, description, parameters, look=Look()):
    # Stage large patches within the original shader's useful fog distance.
    # F in the viewer still applies the original full-domain framing rule.
    return Scene(name, description, parameters,
                 frame_domain(min(parameters.domain, 512)), look)


SCENES = (
    _scene("Light wind", "3 m/s wind, 8 km fetch, 30 m depth",
           Wave_parameters(domain=160, wind_speed=3, fetch_km=8, depth=30, swell=.1)),
    _scene("Moderate wind", "8 m/s wind, 70 km fetch, 80 m depth",
           Wave_parameters(domain=350, wind_speed=8, fetch_km=70, depth=80, swell=.15)),
    _scene("Open ocean", "17 m/s wind, 300 km fetch, 100 m depth", Wave_parameters()),
    _scene("Swell", "22 m/s wind, 800 km fetch, swell 1",
           Wave_parameters(domain=1400, wind_speed=22, fetch_km=800, depth=250, swell=1)),
    _scene("Shallow water", "17 m/s wind, 300 km fetch, 8 m depth",
           Wave_parameters(domain=700, wind_speed=17, fetch_km=300, depth=8, swell=.45)),
    _scene("Storm", "35 m/s wind, 1,250 km fetch, 150 m depth",
           Wave_parameters(domain=1800, wind_speed=35, fetch_km=1250, depth=150, swell=.35)),
)


def scene_with_resolution(index, resolution):
    scene = SCENES[index]
    return replace(scene, parameters=replace(scene.parameters, resolution=resolution))


@dataclass(frozen=True)
class Presentation_view:
    name: str
    camera: Camera
    look: Look


def presentation_views(domain=512.0):
    """Camera/light compositions only; none change the ocean parameters."""
    scale=min(domain,512.0)/512.0
    heading=math.radians(-165)
    return (
        Presentation_view("Overview",frame_domain(min(domain,512.0)),Look()),
        Presentation_view("Storm horizon",look_at((425*scale,-425*scale,95*scale),fov=48),
                          Look(exposure=.3,sky_rotation=40,haze=1.3)),
        Presentation_view("Across the crests",
            look_at((-430*math.sin(heading)*scale,-430*math.cos(heading)*scale,75*scale),fov=44),
            Look(exposure=.45,sky_rotation=45,haze=.85)),
    )
