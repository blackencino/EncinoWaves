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
INFO_LEFT, INFO_TOP, INFO_WIDTH = 48.0, 48.0, 548.0
_CONTROL_KEYS = {"wind": "W", "depth": "D", "fetch": "F", "swell": "S",
                 "foam": "M", "gain": "G", "pinch": "P", "historical_pinch": "P"}
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
    if mode in ("gain", "phillips_amplitude", "g"):
        return f"{value:.3f}"
    if mode in ("pinch", "historical_pinch", "phillips_pinch", "p"):
        return f"{value:.2f}"
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


def _info_box(height, opacity):
    imgui.get_foreground_draw_list().add_rect_filled(
        (INFO_LEFT, INFO_TOP), (INFO_LEFT + INFO_WIDTH, INFO_TOP + height),
        _color(.025, .035, .04, opacity * .72), 10.0)


def draw_control(font, overlay):
    """Keep the selected control in the upper-left, with motion-weighted arrows."""
    if not overlay:
        return
    opacity = _unit(overlay.get("opacity", 1.0))
    label = str(overlay.get("label", ""))
    if not label or opacity <= 0:
        return
    _info_box(184.0, opacity)
    key = _CONTROL_KEYS.get(overlay.get("mode"), "")
    _text(font, 34.0, label, INFO_LEFT + 24, INFO_TOP + 18, opacity, shadow=False)
    _text(font, 24.0, key, INFO_LEFT + INFO_WIDTH - 44, INFO_TOP + 24,
          opacity * .65, shadow=False)
    y = INFO_TOP + 94
    _arrow(INFO_LEFT + 54, y, -1,
           overlay.get("left_weight", 0.0), opacity)
    _arrow(INFO_LEFT + INFO_WIDTH - 54, y, 1,
           overlay.get("right_weight", 0.0), opacity)
    value = overlay.get("value_text", overlay.get("value"))
    if value is not None and str(value):
        if isinstance(value, (float, int)):
            value = format_control_value(overlay.get("mode", ""), value)
        extent = _text_size(font, 42.0, str(value))
        _text(font, 42.0, str(value), INFO_LEFT + (INFO_WIDTH - extent.x) * .5,
              INFO_TOP + 72, opacity, shadow=False)
    _text(font, 24.0, f"Arrows adjust  ·  {key} closes", INFO_LEFT + 24,
          INFO_TOP + 146, opacity * .70, shadow=False)


def draw_example(font, name, opacity):
    """Show an example's name without directional arrows."""
    if name and opacity > 0.0:
        _centered_text(font, 64.0, str(name), 756.0, _unit(opacity))


def _telemetry_rows(parameters, foam):
    rows = [("wind", "Wind Speed", parameters.wind_speed),
            ("depth", "Ocean Depth", parameters.depth)]
    if parameters.spectrum == "phillips":
        return rows + [("gain", "Phillips Gain", parameters.phillips_amplitude),
                       ("historical_pinch", "Pinch", parameters.phillips_pinch)]
    return rows + [("fetch", "Fetch", parameters.fetch_km),
                   ("swell", "Swell", parameters.swell),
                   ("pinch", "Pinch", parameters.pinch),
                   ("foam", "Foam", foam)]


def info_bottom(parameters, telemetry_opacity, overlay):
    """Reserve room for comparison labels below either upper-left panel."""
    if telemetry_opacity > 0:
        return INFO_TOP + (298.0 if parameters.spectrum == "phillips" else 342.0)
    if overlay and overlay.get("mode") in _CONTROL_KEYS and overlay.get("opacity", 0) > 0:
        return INFO_TOP + 184.0
    return INFO_TOP


def draw_telemetry(font, parameters, opacity, *, foam=1.0, overlay=None):
    """Compact sea conditions with persistent highlighting of the active control."""
    opacity = _unit(opacity)
    if opacity <= 0.0:
        return
    rows = _telemetry_rows(parameters, foam)
    historical = parameters.spectrum == "phillips"
    height = info_bottom(parameters, opacity, overlay) - INFO_TOP
    _info_box(height, opacity)
    _text(font, 32.0, "Tessendorf baseline" if historical else "Encino Waves",
          INFO_LEFT + 24, INFO_TOP + 18, opacity, shadow=False)
    active = overlay.get("mode") if overlay and overlay.get("active") else None
    draw = imgui.get_foreground_draw_list()
    for row, (mode, label, value) in enumerate(rows):
        y = INFO_TOP + 64 + row * 38
        selected = mode == active
        if selected:
            draw.add_rect_filled((INFO_LEFT + 12, y - 3),
                                 (INFO_LEFT + INFO_WIDTH - 12, y + 33),
                                 _color(.50, .72, .82, opacity * .24), 4.0)
        _text(font, 23.0, _CONTROL_KEYS[mode], INFO_LEFT + 24, y + 3,
              opacity * (.95 if selected else .50), shadow=False)
        _text(font, 28.0, label, INFO_LEFT + 62, y,
              opacity * (1.0 if selected else .78), shadow=False)
        _text(font, 28.0, format_control_value(mode, value), INFO_LEFT + 310, y,
              opacity, shadow=False)
    footer_y = INFO_TOP + 64 + len(rows) * 38 + 8
    if historical:
        _text(font, 24.0, "Depth affects speed only", INFO_LEFT + 24, footer_y,
              opacity * .70, shadow=False)
        footer_y += 32
    key = _CONTROL_KEYS.get(active)
    hint = f"Arrows adjust  ·  {key} closes" if key else "Choose a control  ·  I hides info"
    _text(font, 24.0, hint, INFO_LEFT + 24, footer_y, opacity * .70, shadow=False)


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
