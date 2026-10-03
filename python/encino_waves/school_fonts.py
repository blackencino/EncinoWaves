# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Font assets for School text overlays, independent of the viewer UI font."""
from pathlib import Path


def nimbus_regular() -> Path:
    """Use bundled Nimbus Sans Regular, falling back to system Helvetica."""
    candidates = (
        Path(__file__).parent / "assets/fonts/NimbusSans-Regular.otf",
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/Library/Fonts/Helvetica.ttf"),
    )
    for path in candidates:
        if path.is_file(): return path
    raise FileNotFoundError("Nimbus Sans Regular and system Helvetica are unavailable")
