# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Fixed-format Academy presentation viewer, sharing the ordinary viewer's ocean.

The preview fills most of the desktop while output remains 1920 x 1080.
ImGui uses output coordinates; camera gestures retain native coordinates.
Rehearsals record decisions rather than screen frames. Offline playback uses
exact frame times, independently of the speed of the live preview.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
import time

from imgui_bundle import imgui, portable_file_dialogs as file_dialogs
from PIL import Image
from rendercanvas.glfw import RenderCanvas
from wgpu.utils.imgui import ImguiRenderer

from .camera import Camera, interpolate_camera
from .academy_fonts import nimbus_regular
from .academy_overlays import TITLE_DURATION, draw_title, draw_control, draw_example, draw_telemetry, format_control_value
from .academy_scenes import opening_scene, academy_examples, ACADEMY_FOAM
from .editing import make_wave_basis, same_wave_basis
from .foam import Foam_parameters, prepare_foam
from .model import Wave_parameters
from .render import Look, Ocean_renderer, read_rgba
from .viewer import Viewer


WIDTH, HEIGHT, FPS = 1920, 1080, 24
TITLE_SAFE_TOP, TITLE_SAFE_BOTTOM = 200, 880
FOAM_PREROLL = 6.0


class _GuiContext:
    """Scale fixed output coordinates onto the larger native preview."""
    logical_size = (WIDTH, HEIGHT)

    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        return getattr(self.context, name)

    @property
    def pixel_ratio(self):
        return self.context.physical_size[0] / WIDTH


class _GuiCanvas:
    def __init__(self, canvas):
        self.canvas = canvas
        self.context = _GuiContext(canvas.get_wgpu_context())

    def get_wgpu_context(self):
        return self.context

    def add_event_handler(self, callback, *types, **kwargs):
        def mapped(event):
            adjusted = dict(event)
            if event["event_type"] == "resize":
                adjusted.update(width=WIDTH, height=HEIGHT, pixel_ratio=self.context.pixel_ratio)
            elif "x" in event and "y" in event:
                width, height = self.canvas.get_logical_size()
                adjusted["x"] = event["x"] * WIDTH / max(1, width)
                adjusted["y"] = event["y"] * HEIGHT / max(1, height)
            callback(adjusted)
            if adjusted.get("stop_propagation"):
                event["stop_propagation"] = True
        self.canvas.add_event_handler(mapped, *types, **kwargs)


class AcademyViewer(Viewer):
    def __init__(self, device="auto", sky=None, preset=2, scene=None, *, canvas=None,
                 max_frames=0, replay=False, render_resolution=None):
        self._locking_size = False
        self.replay = replay
        self.render_resolution = render_resolution
        self.replay_elapsed = 0.0
        self.ui_state = {"headers": {}, "scroll": 0.0}
        self.seek_serial = 0
        self.seek_time = 10.0
        self.reset_foam_serial = 0
        self.rehearsal = None
        self.rehearsal_path = None
        self.rehearsal_started = 0.0
        self.last_take_duration = None
        self._last_autosave = 0.0
        self._applying_sample = False
        self._has_replay_sample = False
        self.performance = None
        self.performance_time = 0.0
        self.performance_overlay = None
        self.telemetry = {"visible": False, "from": 0.0, "start": 0.0}
        self._performance_replay = False
        self._held_keys = set()
        self._camera_take = None
        self._example_camera_move = None
        self._next_example_index = 0
        self._recording_initial_values = None
        self._performance_origin = time.perf_counter()
        canvas = canvas or RenderCanvas(size=(WIDTH, HEIGHT),
            title="Encino Waves — Academy | 1920 x 1080 | 24 fps",
            update_mode="continuous", max_fps=FPS)
        self._fit_canvas(canvas)
        if hasattr(canvas, "_window"):
            import glfw
            glfw.set_window_aspect_ratio(canvas._window, 16, 9)
            glfw.set_window_attrib(canvas._window, glfw.RESIZABLE, True)
        super().__init__(resolution=render_resolution or 2048, device=device, sky=sky, preset=preset, scene=scene,
                         canvas=canvas, max_frames=max_frames)
        if not scene:
            self.foam_parameters = ACADEMY_FOAM
        self.show_ui = False
        self.canvas.set_update_mode("continuous", max_fps=FPS)
        self.canvas.add_event_handler(self._on_resize, "resize", order=-100)
        self.canvas.remove_event_handler(self.on_key, "key_down")
        self.canvas.add_event_handler(self.on_key, "key_down", order=-100)
        self.canvas.add_event_handler(self._on_key_up, "key_up", order=-100)
        self._new_performance()
        if hasattr(self.canvas, "_window"):
            import glfw
            def focus_changed(window, focused):
                self.canvas._on_window_dirty(window, focused)
                self._on_focus({"focused": bool(focused)})
            self._focus_callback = focus_changed
            glfw.set_window_focus_callback(self.canvas._window, focus_changed)

    def load_scene(self, path):
        super().load_scene(path)
        self.look = replace(self.look, crest_crumble=True, crest_crumble_strength=.5)
        if self.foam_parameters.resolution < 1024:
            self.foam_parameters = replace(self.foam_parameters, resolution=1024)
        if self.performance is not None:
            self._new_performance()

    def _initial_scene(self, preset, resolution):
        return opening_scene(resolution)

    def _style(self):
        super()._style()
        self.overlay_font = imgui.get_io().fonts.add_font_from_file_ttf(str(nimbus_regular()), 42)

    def _new_performance(self):
        from .academy_performance import Performance
        self.performance = Performance(asdict(self.parameters), initial_look=asdict(self.look))
        self._performance_origin = time.perf_counter()
        self.performance_time = 0.0
        self.performance_overlay = None
        # Preparation choices carry into a take without retaining timestamps
        # from the preparation clock, which resets after the foam pre-roll.
        self.telemetry = {"visible": self.telemetry["visible"],
                          "from": float(self.telemetry["visible"]), "start": 0.0}
        self._held_keys.clear()
        self._camera_take = None
        self._example_camera_move = None

    def _wall_time(self):
        return max(0.0, time.perf_counter() - self._performance_origin)

    @staticmethod
    def _fit_canvas(canvas):
        if hasattr(canvas, "_window"):
            import glfw
            x, y, available_width, available_height = glfw.get_monitor_workarea(glfw.get_primary_monitor())
            width = int(min(available_width * .94, (available_height - 60) * .94 * WIDTH / HEIGHT))
            height = round(width * HEIGHT / WIDTH)
            canvas.set_logical_size(width, height)
            glfw.set_window_pos(canvas._window, x + (available_width - width) // 2,
                                y + (available_height - height) // 2)
        else:
            ratio = canvas.get_pixel_ratio()
            canvas.set_logical_size(WIDTH / ratio, HEIGHT / ratio)
        # GLFW caches dimensions; synchronize them before ImGui initialization.
        if hasattr(canvas, "_determine_size"):
            canvas._determine_size()

    def _on_resize(self, event):
        if hasattr(self.canvas, "_window"):
            return  # GLFW retains 16:9; preview size does not affect the export.
        if self._locking_size or self.canvas.get_physical_size() == (WIDTH, HEIGHT):
            return
        if min(self.canvas.get_physical_size()) < 1:
            return
        self._locking_size = True
        try:
            self._fit_canvas(self.canvas)
        finally:
            self._locking_size = False

    def _create_gui(self):
        gui = ImguiRenderer(self.device, _GuiCanvas(self.canvas), self.format)
        backend = gui._backend
        update_texture = backend._update_texture

        def upload_texture(tex):
            creating = tex.status == imgui.ImTextureStatus.want_create
            update_texture(tex)
            if creating:
                # The installed backend uploads only the dirty rectangle when
                # creating a larger font atlas, leaving copied glyphs blank.
                # A new texture needs the full atlas; later dirty updates work.
                backend._device.queue.write_texture(
                    {"texture": backend._textures[tex.tex_id], "mip_level": 0,
                     "origin": (0, 0, 0)},
                    tex.get_pixels_array(),
                    {"offset": 0, "bytes_per_row": tex.width * tex.bytes_per_pixel},
                    (tex.width, tex.height, 1))

        backend._update_texture = upload_texture
        return gui

    def _create_renderer(self, sky):
        return Ocean_renderer(self.device, sky, target_format=self.format,
                              mesh_resolution=(960, 576) if self.replay else (640, 384))

    def _parameters_from_scene(self, saved):
        parameters = super()._parameters_from_scene(saved)
        return replace(parameters, domain=1024, trough_damping=1.0,
                       resolution=self.render_resolution or 2048)

    def _follow_parameters(self, current, desired, elapsed):
        if (not self.replay and self.performance is not None) or self._performance_replay:
            return desired
        return super()._follow_parameters(current, desired, elapsed)

    def _frame_delta(self, now):
        return self.replay_elapsed if self.replay else super()._frame_delta(now)

    def _update_parameters(self, elapsed):
        if self.replay and not same_wave_basis(self.basis.parameters, self.parameters):
            self.basis = make_wave_basis(self.parameters, self.device_name)
            self.reset_requested = True
        super()._update_parameters(elapsed)

    def _header(self, label):
        imgui.set_next_item_open(self.ui_state["headers"].get(label, False))
        expanded = imgui.collapsing_header(label)
        self.ui_state["headers"][label] = expanded
        return expanded

    def _tree(self, label):
        imgui.set_next_item_open(self.ui_state["headers"].get(label, False))
        expanded = imgui.tree_node(label)
        self.ui_state["headers"][label] = expanded
        return expanded

    def _control_scroll(self):
        if self.replay:
            imgui.set_scroll_y(self.ui_state["scroll"])
        else:
            self.ui_state["scroll"] = imgui.get_scroll_y()

    def _seek_time(self, value):
        self.seek_serial += 1
        self.seek_time = value
        super()._seek_time(value)

    def reset_foam(self):
        if not self._applying_sample:
            self.reset_foam_serial += 1
        super().reset_foam()

    def _control_panel_rect(self, width, height):
        # The Academy artwork diagram reserves 400px top/bottom at UHD.
        # Apply its proportional 200px clearance to presentation UI at HD.
        return WIDTH - 370 - 48, TITLE_SAFE_TOP, 370, TITLE_SAFE_BOTTOM - TITLE_SAFE_TOP

    def _draw_navigation(self, width, height, fixed):
        # Keep presentation frames free of the ordinary viewer's bottom hint.
        pass

    def _draw_comparison_labels(self, width, height, fixed):
        if not self.comparing:
            return
        for x, label in ((96, "EARLIER MODEL"), (WIDTH // 2 + 48, "ENCINO WAVES")):
            imgui.set_next_window_pos((x, TITLE_SAFE_TOP + 16))
            imgui.set_next_window_bg_alpha(.65)
            imgui.begin(label, flags=fixed | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_inputs)
            self._label(label, 16, (.92, .94, .89, 1))
            imgui.end()

    def _draw_files_gui(self):
        if self._header("Files"):
            imgui.begin_disabled(self.rehearsal is not None or self.replay)
            if imgui.button("Open scene..."):
                self._open_file("scene")
            if imgui.button("Open rehearsal..."):
                self._open_file("rehearsal")
            imgui.end_disabled()
            imgui.text_wrapped("Save still captures this 1920 x 1080 view, including controls, and saves its scene.")

    def _open_file(self, action):
        if self.rehearsal is not None:
            self.message = "Stop the rehearsal before opening a scene or sky."
            return
        if action in ("rehearsal", "save_rehearsal_as"):
            if self.dialog:
                return
            self.dialog_action = action
            if action == "save_rehearsal_as":
                if self.rehearsal_path is None:
                    return
                default = Path("renders") / (self.rehearsal_path.parent.name + ".json")
                self.dialog = file_dialogs.save_file("Save Academy recording as", str(default.resolve()),
                                                    ["Academy recording", "*.json"])
            else:
                self.dialog = file_dialogs.open_file("Open Academy rehearsal", str(Path("renders").resolve()),
                                                    ["Academy rehearsal", "*.json"])
            return
        super()._open_file(action)

    def _poll_files(self):
        academy_dialogs = ("rehearsal", "save_rehearsal_as")
        if self.dialog and self.dialog_action in academy_dialogs and self.dialog.ready(0):
            result = self.dialog.result()
            action = self.dialog_action
            self.dialog = None
            if result:
                from .academy_timeline import load_timeline, save_rehearsal_as
                try:
                    path = Path(result[0] if isinstance(result, list) else result)
                    if action == "save_rehearsal_as":
                        path = save_rehearsal_as(self.rehearsal_path, path)
                    timeline = load_timeline(path)
                    self.rehearsal_path = path
                    self.last_take_duration = timeline.duration
                    self.message = f"{'Saved' if action == 'save_rehearsal_as' else 'Loaded'} recording: {path}"
                except Exception as error:
                    self.message = str(error)
        exporting = self.movie_process is not None
        # The base poller only handles sky/scene dialogs and its movie process.
        dialog = self.dialog if self.dialog_action in academy_dialogs else None
        if dialog is not None:
            self.dialog = None
        try:
            super()._poll_files()
        finally:
            if dialog is not None:
                self.dialog = dialog
        if exporting and self.movie_process is None and "failed" in self.message.lower():
            self.message = "Academy render failed; see renders/academy_export.log"

    def draw_gui(self):
        if self.show_ui:
            fixed = imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
            imgui.set_next_window_pos((WIDTH - 420, TITLE_SAFE_TOP))
            imgui.set_next_window_size((372, 460))
            imgui.begin("academy_setup", flags=fixed)
            self._label("Academy setup", 23, (.92, .94, .92, 1))
            imgui.text("1024 m patch | 2048 waves | 1024 foam")
            imgui.text("Output: 1080p24 | 4096 waves")
            imgui.text_wrapped("W wind speed, D depth, F fetch, S swell, M foam. Left/right arrows drive the selected control. E advances to the next example. I toggles telemetry.")
            imgui.text_wrapped("Maya camera controls are unchanged. Playback smooths between cameras after four seconds without an adjustment.")
            imgui.begin_disabled(self.rehearsal is not None or self.replay)
            if imgui.button("Reset opening ocean"):
                self._reset_opening()
            if imgui.button("Open scene..."):
                self._open_file("scene")
            if imgui.button("Open rehearsal..."):
                self._open_file("rehearsal")
            imgui.begin_disabled(self.rehearsal_path is None)
            if imgui.button("Save recording as..."):
                self._open_file("save_rehearsal_as")
            imgui.end_disabled()
            imgui.end_disabled()
            if self.message:
                imgui.text_wrapped(self.message)
            imgui.end()
        if self.comparing:
            fixed = imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
            self._draw_comparison_labels(WIDTH, HEIGHT, fixed)
        draw_telemetry(self.overlay_font, self.state.parameters, self._telemetry_opacity(self.performance_time))
        if self.performance_overlay:
            overlay = dict(self.performance_overlay)
            if overlay.get("mode") == "examples":
                draw_example(self.overlay_font, overlay.get("label", "Examples"), overlay.get("opacity", 0))
            else:
                mode, value = overlay.get("mode"), overlay.get("value")
                if isinstance(value, (float, int)):
                    overlay["value_text"] = format_control_value(mode, value)
                draw_control(self.overlay_font, overlay)
        if self._show_title():
            draw_title(self.overlay_font, self.performance_time)
        if self.replay:
            return
        if self.rehearsal is not None:
            self._draw_recording_clock()
        # Session controls are present while rehearsing and omitted in the movie.
        fixed = imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
        imgui.set_next_window_pos((48, TITLE_SAFE_BOTTOM - 128))
        imgui.set_next_window_size((540, 128))
        imgui.begin("academy_session", flags=fixed)
        if self.rehearsal is not None:
            elapsed = min(298.0, time.perf_counter() - self.rehearsal_started)
            imgui.text(f"Recording decisions   {int(elapsed)//60}:{int(elapsed)%60:02d} / 4:58")
            if imgui.button("Stop rehearsal (R)"):
                self.stop_rehearsal()
        else:
            self._label("W wind  D depth  F fetch  S swell  M foam  E next example  I telemetry", 15, (.91, .93, .92, 1))
            imgui.begin_disabled(self.changed or self.future is not None or self.movie_process is not None)
            if imgui.button("Record rehearsal (R)"):
                self.start_rehearsal()
            imgui.end_disabled()
            imgui.same_line()
            imgui.begin_disabled(self.rehearsal_path is None or self.movie_process is not None)
            if imgui.button("Render rehearsal"):
                self.render_rehearsal()
            imgui.end_disabled()
            imgui.same_line()
            if imgui.button("Files / setup (Tab)"):
                self.show_ui = not self.show_ui
        if self.movie_process is not None:
            imgui.text("Rendering offline. Progress: renders/academy_export.log")
        elif self.rehearsal_path:
            duration = self.last_take_duration
            label = f"CUT {self._clock_text(duration)}  ·  Saved: " if duration is not None else "Saved: "
            self._label(label + self.rehearsal_path.parent.name + "/" + self.rehearsal_path.name, 12)
        imgui.end()

    def _show_title(self):
        return (self.replay and self._performance_replay or self.rehearsal is not None) and self.performance_time < TITLE_DURATION

    def _telemetry_opacity(self, now):
        amount = min(1.0, max(0.0, (now - self.telemetry["start"]) / .22))
        eased = amount ** 3 * (10 + amount * (-15 + 6 * amount))
        return self.telemetry["from"] + (float(self.telemetry["visible"]) - self.telemetry["from"]) * eased

    def _toggle_telemetry(self, now):
        opacity = self._telemetry_opacity(now)
        self.telemetry = {"visible": not self.telemetry["visible"], "from": opacity, "start": now}
        if self.rehearsal is not None:
            self.rehearsal.append(min(298.0, now), self.rehearsal_values())

    @staticmethod
    def _clock_text(seconds):
        tenths = max(0, int(seconds * 10))
        return f"{tenths // 600:02d}:{tenths // 10 % 60:02d}.{tenths % 10}"

    def _draw_recording_clock(self):
        elapsed = min(298.0, max(0.0, time.perf_counter() - self.rehearsal_started))
        remaining = max(0.0, 298.0 - elapsed)
        # Deliberately outside the picture's title area. This operator clock is
        # drawn only in the interactive viewer, including over the title card.
        draw = imgui.get_foreground_draw_list()
        color = imgui.get_color_u32(imgui.ImVec4(1, .35 if remaining < 15 else .85, .3 if remaining < 15 else .85, 1))
        draw.add_rect_filled((32, 32), (575, 117), imgui.get_color_u32(imgui.ImVec4(0, 0, 0, .8)), 8)
        draw.add_text(self.overlay_font, 30, (50, 45), color,
                      f"REC  {self._clock_text(elapsed)}  /  04:58.0")
        draw.add_text(self.overlay_font, 19, (50, 84), color,
                      f"{self._clock_text(remaining)} remaining     R to cut")

    def rehearsal_values(self):
        values = {"parameters": asdict(self.parameters), "camera": asdict(self.camera),
                "look": asdict(self.look), "foam": asdict(self.foam_parameters),
                "selected_scene": self.selected_scene, "comparing": self.comparing,
                "tessendorf_only": self.tessendorf_only, "playing": self.playing,
                "speed": self.speed, "show_ui": self.show_ui, "ui_state": self.ui_state,
                "seek_serial": self.seek_serial, "seek_time": self.seek_time,
                "reset_foam_serial": self.reset_foam_serial}
        values["telemetry"] = deepcopy(getattr(self, "telemetry", {"visible": False, "from": 0.0, "start": 0.0}))
        if self.rehearsal is not None and self._recording_initial_values is not None:
            # Physical motion is reconstructed from timed impulses, and camera
            # tweaks are replaced with deliberate endpoint moves on the tape.
            for name in ("parameters", "camera"):
                values[name] = deepcopy(self._recording_initial_values[name])
            values["look"]["foam"] = self._recording_initial_values["look"]["foam"]
        return values

    def start_rehearsal(self):
        from .academy_timeline import Rehearsal
        if self.rehearsal is not None or self.changed or self.future is not None:
            return
        # Build history ending at the configured simulation time. Neither this
        # work nor checkpoint compression belongs on the performance clock.
        self.foam_state = prepare_foam(self.state, self.time, self.foam_parameters,
                                       preroll=FOAM_PREROLL)
        if self.comparing and self.comparison_state is not None:
            self.comparison_foam_state = prepare_foam(
                self.comparison_state, self.time, self.foam_parameters, preroll=FOAM_PREROLL)
        directory = Path("renders") / (time.strftime("academy_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}")
        directory.mkdir(parents=True, exist_ok=True)
        initial = directory / "initial_scene.json"
        self.save_scene(initial)
        self._new_performance()
        self.last_time = self._performance_origin
        self.message = ""
        self.rehearsal_path = directory / "rehearsal.json"
        self.rehearsal_started = self._performance_origin
        self._last_autosave = 0.0
        self._recording_initial_values = deepcopy(self.rehearsal_values())
        self.rehearsal = Rehearsal(initial.name, self._recording_initial_values)
        self.rehearsal.set_performance(self.performance.to_dict())
        self.last_take_duration = None

    def stop_rehearsal(self):
        if self.rehearsal is None:
            return
        elapsed = min(298.0, time.perf_counter() - self.rehearsal_started)
        self._commit_camera(elapsed, force=True)
        self.rehearsal.append(elapsed, self.rehearsal_values())
        self.rehearsal.set_performance(self.performance.to_dict())
        self.rehearsal.save(self.rehearsal_path, max(1 / FPS, elapsed))
        self.last_take_duration = elapsed
        self.rehearsal = None
        self.message = f"Saved rehearsal: {self.rehearsal_path}"

    def render_rehearsal(self):
        if self.rehearsal_path is None or self.movie_process is not None:
            return
        output = self.rehearsal_path.with_name(f"academy_{time.time_ns()}.mov")
        command = [sys.executable, "-m", "encino_waves", "academy-render",
                   str(self.rehearsal_path), str(output), "--device", self.device_name]
        self.movie_log = Path("renders/academy_export.log").open("w")
        self.movie_process = subprocess.Popen(command, stdout=self.movie_log, stderr=subprocess.STDOUT)
        self.playing = False
        self.message = f"Rendering {output}"

    def apply_rehearsal_sample(self, values):
        self._applying_sample = True
        try:
            parameters = Wave_parameters.from_dict(values["parameters"])
            if self.render_resolution:
                parameters = replace(parameters, resolution=self.render_resolution)
            self.parameters = parameters
            self.camera = Camera(**values["camera"])
            self.look = Look.from_dict(values["look"])
            self.foam_parameters = Foam_parameters.from_dict(values["foam"])
            for name in ("selected_scene", "comparing", "tessendorf_only", "playing", "speed", "show_ui"):
                setattr(self, name, values[name])
            self.ui_state = values.get("ui_state", {"headers": {}, "scroll": 0.0})
            if self._has_replay_sample and values.get("seek_serial", 0) != self.seek_serial:
                self.time = values["seek_time"]
                self.reset_foam()
            if self._has_replay_sample and values.get("reset_foam_serial", 0) != self.reset_foam_serial:
                self.reset_foam()
            self.seek_serial = values.get("seek_serial", 0)
            self.reset_foam_serial = values.get("reset_foam_serial", 0)
            self._has_replay_sample = True
            self._performance_replay = "performance_time" in values
            self.performance_time = values.get("performance_time", TITLE_DURATION)
            self.performance_overlay = values.get("performance_overlay")
            self.telemetry = deepcopy(values.get("telemetry", {"visible": False, "from": 0.0, "start": 0.0}))
            imgui.get_io().add_mouse_pos_event(-100, -100)
            imgui.get_io().mouse_draw_cursor = False
        finally:
            self._applying_sample = False

    def draw(self):
        if not self.replay:
            self.performance_time = self._wall_time()
            sampled = self.performance.sample(self.performance_time)
            parameters = Wave_parameters.from_dict(sampled["parameters"])
            if parameters != self.parameters:
                self.edit_parameters(**asdict(parameters))
            if "look" in sampled:
                self.look = replace(self.look, **sampled["look"])
            self.performance_overlay = sampled["overlay"]
            if self._example_camera_move is not None:
                start, end, before, after = self._example_camera_move
                amount = max(0.0, min(1.0, (self.performance_time - start) / (end - start)))
                eased = amount ** 3 * (10 + amount * (-15 + 6 * amount))
                self.camera = interpolate_camera(before, after, eased)
                if amount >= 1:
                    self._example_camera_move = None
            self._commit_camera(self.performance_time)
        super().draw()
        if self.rehearsal is not None:
            elapsed = min(298.0, time.perf_counter() - self.rehearsal_started)
            self.rehearsal.append(elapsed, self.rehearsal_values())
            if elapsed - self._last_autosave >= 1:
                self.rehearsal.set_performance(self.performance.to_dict())
                self.rehearsal.save(self.rehearsal_path, elapsed)
                self._last_autosave = elapsed
            if elapsed >= 298:
                self.stop_rehearsal()

    def run(self):
        try:
            super().run()
        finally:
            self.stop_rehearsal()

    def on_key(self, event):
        if self.replay or imgui.get_io().want_text_input:
            return
        key = event.get("key", "").lower()
        if key in self._held_keys:
            return
        self._held_keys.add(key)
        now = self._wall_time()
        modes = {"w": "wind_speed", "d": "depth", "f": "fetch_km", "s": "swell", "m": "foam"}
        if key in modes:
            self.performance.mode_key(modes[key], now)
        elif key in ("arrowleft", "arrowright"):
            self.performance.direction_key("left" if key == "arrowleft" else "right", True, now)
        elif key == "e":
            self._next_example(now)
        elif key == "i":
            self._toggle_telemetry(now)
        elif key == "r":
            if self.rehearsal is None:
                self.start_rehearsal()
            else:
                self.stop_rehearsal()
        elif key in ("f11", "1", "2", "3", "4", "5", "6"):
            return
        elif key == "escape":
            self.show_ui = False
        else:
            super().on_key(event)

    def _on_key_up(self, event):
        key = event.get("key", "").lower()
        self._held_keys.discard(key)
        if not self.replay and key in ("arrowleft", "arrowright"):
            self.performance.direction_key("left" if key == "arrowleft" else "right", False, self._wall_time())

    def _on_focus(self, event):
        if event.get("event_type") == "blur" or event.get("focused") is False:
            self._held_keys.clear()
            for direction in ("left", "right"):
                self.performance.direction_key(direction, False, self._wall_time())

    def _next_example(self, now):
        self._commit_camera(now, force=True)
        examples = academy_examples(self.parameters.resolution)
        scene = examples[self._next_example_index]
        self._next_example_index = (self._next_example_index + 1) % len(examples)
        self.performance.transition(asdict(scene.parameters), now, duration=3.0, name=scene.name)
        self.performance.add_camera_move(now, now + 4.0, self.camera, scene.camera)
        self._example_camera_move = (now, now + 4.0, self.camera, scene.camera)

    def _reset_opening(self):
        from .editing import state_from_basis
        self._next_example_index = 0
        scene = opening_scene(self.render_resolution or 2048)
        self.parameters, self.camera, self.look = scene.parameters, scene.camera, scene.look
        if not same_wave_basis(self.basis.parameters, self.parameters):
            self.basis = make_wave_basis(self.parameters, self.device_name)
        self.state = state_from_basis(self.basis, self.parameters)
        self.time = 10.0
        self.foam_parameters = ACADEMY_FOAM
        self.reset_foam()
        self.comparing = self.tessendorf_only = False
        self.comparison_state = None
        self.future = None
        self.reset_requested = self.changed = False
        self.playing = True
        self.show_ui = False
        self.message = ""
        self._new_performance()

    def on_pointer(self, event):
        if self.replay:
            return
        before = self.camera
        now = self._wall_time()
        self._commit_camera(now)
        # Preserve the familiar Maya interaction exactly. Only the saved take
        # replaces these exploratory tweaks with a smooth endpoint movement.
        super().on_pointer(event)
        if self.camera != before:
            self._example_camera_move = None
            if self._camera_take is None:
                self._camera_take = {"start": now, "camera": before, "last": now}
            self._camera_take["last"] = now

    def _commit_camera(self, now, force=False):
        take = self._camera_take
        if take is None or (not force and now - take["last"] < 4.0):
            return
        if now > take["start"]:
            self.performance.add_camera_move(take["start"], now, take["camera"], self.camera)
        self._camera_take = None

    def save_still(self):
        directory = Path("renders")
        directory.mkdir(exist_ok=True)
        name = time.strftime("academy_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        width, height = self.canvas.get_physical_size()
        if (width, height) != (WIDTH, HEIGHT):
            self.message = "The Academy view must be 1920 x 1080 before saving."
            return
        rgba = read_rgba(self.device, self.context.get_current_texture(), WIDTH, HEIGHT)
        Image.fromarray(rgba).save(directory / f"{name}.png")
        self.save_scene(directory / f"{name}.json")
        self.message = f"Saved renders/{name}.png"
