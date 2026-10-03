# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Presentation compositions on one continuous 1024-metre ocean patch.

These are illustrative presentation settings, not measured NOAA sea data:
https://www.weather.gov/mfl/beaufort
"""
from dataclasses import replace
import math

from .camera import Camera
from .foam import Foam_parameters
from .model import Wave_parameters
from .presets import Scene
from .render import Look


PREVIEW_RESOLUTION = 2048
FINAL_RESOLUTION = 4096
DOMAIN = 1024.0
KNOTS_TO_METERS_PER_SECOND = 1852.0 / 3600.0
# Rectilinear 20 mm lens on the ALEXA Mini 3.2K/UHD 16:9 image area.
# ARRI specifies 26.40 x 14.85 mm; camera.fov is VERTICAL.
# https://www.arri.com/en/cine-systems/cine-cameras/legacy-cine-cameras/alexa-mini
# Ocean-work reference: Boyd's 20-100 mm Cooke, often at its short end.
# https://theasc.com/article/hell-high-water-master-commander/
SCHOOL_FOCAL_LENGTH_MM = 20.0
SCHOOL_IMAGE_AREA_MM = (26.40, 14.85)
SCHOOL_VERTICAL_FOV = math.degrees(2 * math.atan(
    SCHOOL_IMAGE_AREA_MM[1] / (2 * SCHOOL_FOCAL_LENGTH_MM)))
SCHOOL_FOAM = Foam_parameters(resolution=1024)
SCHOOL_LOOK = Look(crest_crumble=True, crest_crumble_strength=.5)
NOAA_SEA_STATE_SOURCE = "https://www.weather.gov/mfl/beaufort"


def _camera(height, pitch, yaw, fov=SCHOOL_VERTICAL_FOV):
    """Retain the authored angles while placing the Maya pivot at sea origin."""
    camera = Camera(height=height, pitch=pitch, yaw=yaw, fov=fov)
    distance = -height / math.sin(math.radians(pitch))
    forward = camera.basis()[0]
    return replace(camera, x=-distance * float(forward[0]), y=-distance * float(forward[1]),
                   center_of_interest=distance)


def opening_scene(resolution=PREVIEW_RESOLUTION):
    return Scene(
        "Opening ocean", "17 m/s wind, 300 km fetch, 100 m depth",
        Wave_parameters(resolution=resolution, domain=DOMAIN, trough_damping=1.0),
        _camera(145.2, -18.1, -139.9), SCHOOL_LOOK,
    )


def school_examples(resolution=PREVIEW_RESOLUTION):
    base = opening_scene(resolution).parameters
    return (
        Scene(
            "NOAA severe sea state",
            "50 knots wind, 2000 km fetch, 1000 m depth, no swell",
            replace(base, wind_speed=50.0 * KNOTS_TO_METERS_PER_SECOND,
                    fetch_km=2000.0, depth=1000.0, swell=0.0),
            _camera(85.0, -13.0, -52.0), SCHOOL_LOOK,
        ),
        Scene(
            "Shallow choppy bay", "9 m/s wind, 15 km fetch, 3 m depth",
            replace(base, wind_speed=9.0, fetch_km=15.0, depth=3.0, swell=.05),
            _camera(24.0, -12.0, -122.0), SCHOOL_LOOK,
        ),
        Scene(
            "Calm deep seas", "4 knots wind, 300 km fetch, 1000 m depth, no swell",
            replace(base, wind_speed=4.0 * KNOTS_TO_METERS_PER_SECOND,
                    fetch_km=300.0, depth=1000.0, swell=0.0),
            _camera(45.1, -4.7, 55.9), SCHOOL_LOOK,
        ),
        Scene(
            "Choppy shallow water", "40 knots wind, 100 km fetch, 5 m depth, -0.9 swell",
            replace(base, wind_speed=40.0 * KNOTS_TO_METERS_PER_SECOND,
                    fetch_km=100.0, depth=5.0, swell=-.9),
            _camera(35.0, -11.0, -155.0), SCHOOL_LOOK,
        ),
    )
