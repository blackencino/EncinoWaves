# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Presentation artwork in fixed 1920 x 1080 output coordinates.

Call these helpers within the viewer's ImGui frame, after any ordinary UI.
Their caller owns animation timing so rehearsal and offline playback match.
"""
from imgui_bundle import imgui


WIDTH, HEIGHT = 1920, 1080
TITLE_HOLD = 4.0
TITLE_FADE = 0.45
TITLE_DURATION = TITLE_HOLD + TITLE_FADE
# A small Gaussian-like kernel produces a diffuse shadow, rather than a
# single crisp offset that makes the lettering look embossed.
_SHADOW_KERNEL = ((-4.0, 1), (-2.0, 4), (0.0, 6), (2.0, 4), (4.0, 1))
_SHADOW_TAPS = tuple((dx, dy + 5.0, 0.75 * wx * wy / 256.0)
                     for dx, wx in _SHADOW_KERNEL
                     for dy, wy in _SHADOW_KERNEL)


def format_control_value(mode, value):
    """Use the same presentation units in controls and sea telemetry."""
    mode = str(mode).lower()
    if mode in ("wind", "wind_speed", "w"):
        return f"{value * 3600.0 / 1852.0:.1f} knots"
    if mode in ("depth", "d"):
        return f"{value:.1f} meters"
    if mode in ("fetch", "fetch_km", "f"):
        return f"{value:.0f} kilometers"
    if mode in ("swell", "s"):
        return f"{value:.2f}"
    if mode in ("foam", "foam_amount", "m"):
        return f"{value * 100.0:.0f}%"
    return f"{value:g}"


def _unit(value):
    return max(0.0, min(1.0, float(value)))


def _smooth(value):
    value = _unit(value)
    return value * value * (3.0 - 2.0 * value)


def _color(red, green, blue, alpha):
    return imgui.get_color_u32(imgui.ImVec4(red, green, blue, _unit(alpha)))


def _text_size(font, size, text):
    imgui.push_font(font, size)
    try:
        return imgui.calc_text_size(text)
    finally:
        imgui.pop_font()


def _text(font, size, text, x, y, opacity, *, shadow=True):
    draw = imgui.get_foreground_draw_list()
    font = font or imgui.get_font()
    if shadow:
        for dx, dy, weight in _SHADOW_TAPS:
            draw.add_text(font, size, (x + dx, y + dy),
                          _color(0.0, 0.0, 0.0, opacity * weight), text)
    draw.add_text(font, size, (x, y), _color(1.0, 1.0, 1.0, opacity), text)


def _centered_text(font, size, text, y, opacity, *, shadow=True):
    extent = _text_size(font, size, text)
    _text(font, size, text, (WIDTH - extent.x) * 0.5, y, opacity,
          shadow=shadow)
    return extent


def title_opacity(elapsed):
    """Black/title opacity: four-second hold and a quick smooth reveal."""
    return 1.0 - _smooth((float(elapsed) - TITLE_HOLD) / TITLE_FADE)


def draw_title(font, elapsed):
    """Draw the opening card last; return whether it is still visible."""
    opacity = title_opacity(elapsed)
    if opacity <= 0.0:
        return False
    draw = imgui.get_foreground_draw_list()
    draw.add_rect_filled((0.0, 0.0), (WIDTH, HEIGHT),
                         _color(0.0, 0.0, 0.0, opacity))
    title_height = _text_size(font, 112.0, "Encino Waves").y
    subtitle_height = _text_size(font, 52.0, "Interactive Ocean Synthesis").y
    byline_height = _text_size(font, 44.0, "Christopher Jon Horvath").y
    group_height = title_height + 14.0 + subtitle_height + 52.0 + byline_height
    title_y = (HEIGHT - group_height) * 0.5
    subtitle_y = title_y + title_height + 14.0
    byline_y = subtitle_y + subtitle_height + 52.0
    _centered_text(font, 112.0, "Encino Waves", title_y, opacity, shadow=False)
    _centered_text(font, 52.0, "Interactive Ocean Synthesis", subtitle_y,
                   opacity * 0.90, shadow=False)
    _centered_text(font, 44.0, "Christopher Jon Horvath", byline_y,
                   opacity * 0.90, shadow=False)
    return True


def _arrow(x, y, direction, weight, opacity):
    """Thin gray at rest, brighter/heavier in proportion to actual motion."""
    weight = _unit(weight)
    thickness = 2.5 + 6.5 * weight
    brightness = 0.55 + 0.45 * weight
    alpha = opacity * (0.62 + 0.38 * weight)
    end = x + direction * 26.0
    start = x - direction * 26.0
    shoulder = end - direction * 16.0
    lines = (((start, y), (end, y)),
             ((shoulder, y - 14.0), (end, y)),
             ((shoulder, y + 14.0), (end, y)))
    draw = imgui.get_foreground_draw_list()
    for dx, dy, tap_weight in _SHADOW_TAPS:
        for a, b in lines:
            draw.add_line((a[0] + dx, a[1] + dy),
                          (b[0] + dx, b[1] + dy),
                          _color(0, 0, 0, alpha * tap_weight), thickness)
    for a, b in lines:
        draw.add_line(a, b, _color(brightness, brightness, brightness, alpha),
                      thickness)


def draw_control(font, overlay):
    """Draw ``label`` with arrows driven by ``left_weight/right_weight``.

    ``opacity`` and arrow weights are normalized floats. Optional ``value_text``
    (or ``value``) appears below the label, safely above the bottom clearance.
    """
    if not overlay:
        return
    opacity = _unit(overlay.get("opacity", 1.0))
    label = str(overlay.get("label", ""))
    if not label or opacity <= 0:
        return
    extent = _centered_text(font, 64.0, label, 734.0, opacity)
    half_width = extent.x * 0.5
    y = 734.0 + extent.y * 0.5
    _arrow(WIDTH * 0.5 - half_width - 70.0, y, -1,
           overlay.get("left_weight", 0.0), opacity)
    _arrow(WIDTH * 0.5 + half_width + 70.0, y, 1,
           overlay.get("right_weight", 0.0), opacity)
    value = overlay.get("value_text", overlay.get("value"))
    if value is not None and str(value):
        if isinstance(value, (float, int)):
            value = format_control_value(overlay.get("mode", ""), value)
        _centered_text(font, 44.0, str(value), 814.0, opacity * 0.90)


def draw_example(font, name, opacity):
    """Show an example's name without directional arrows."""
    if name and opacity > 0.0:
        _centered_text(font, 64.0, str(name), 756.0, _unit(opacity))


def draw_telemetry(font, parameters, opacity):
    """Show the currently displayed sea conditions in the title-safe area."""
    opacity = _unit(opacity)
    if opacity <= 0.0:
        return
    rows = (
        ("Wind Speed", format_control_value("wind", parameters.wind_speed)),
        ("Ocean Depth", format_control_value("depth", parameters.depth)),
        ("Fetch", format_control_value("fetch", parameters.fetch_km)),
        ("Swell", format_control_value("swell", parameters.swell)),
    )
    for row, (label, value) in enumerate(rows):
        y = 240.0 + row * 80.0
        _text(font, 52.0, label, 96.0, y, opacity * 0.90)
        _text(font, 56.0, value, 455.0, y - 2.0, opacity)


def draw_camera_telemetry(font, camera, opacity):
    """Draw optional live-camera information beside the sea telemetry."""
    opacity = _unit(opacity)
    if opacity <= 0.0:
        return
    _text(font, 52.0, "Camera", 1120.0, 210.0, opacity)
    rows = (
        ("Height", f"{camera.height:.1f} meters"),
        ("Pitch", f"{camera.pitch:.1f} degrees"),
        ("Yaw", f"{camera.yaw:.1f} degrees"),
        ("FOV", f"{camera.fov:.1f} degrees"),
    )
    for row, (label, value) in enumerate(rows):
        y = 290.0 + row * 80.0
        _text(font, 52.0, label, 1120.0, y, opacity * 0.90)
        _text(font, 56.0, value, 1410.0, y - 2.0, opacity)
