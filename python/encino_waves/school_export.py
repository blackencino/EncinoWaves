# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Render a recorded School UI rehearsal at the fixed presentation format."""
from pathlib import Path
import json
import math
import subprocess
import tempfile
import time
import uuid

import imageio_ffmpeg
import numpy as np


WIDTH = 1920
HEIGHT = 1080
FPS = 24
RENDER_RESOLUTION = 4096
BLACK_SECONDS = 1
MAX_CONTENT_SECONDS = 298
BITRATE_MBPS = 80

# The viewer composites display-referred RGB into an unorm surface. Convert
# both its sRGB transfer and RGB matrix, rather than merely tagging those bytes
# as Rec.709. zscale keeps the intermediate conversion above 8-bit precision.
REC709_FILTER = (
    "format=gbrp,"
    "zscale=matrixin=gbr:transferin=iec61966-2-1:primariesin=bt709:rangein=full:"
    "matrix=bt709:transfer=bt709:primaries=bt709:range=limited:"
    "dither=error_diffusion:chromal=left,format=yuv420p,setsar=1"
)


class School_movie_writer:
    """Fixed 1080p24 H.264 MOV; replace the destination only on success."""

    def __init__(self, path, *, overwrite=False):
        self.path = Path(path)
        if self.path.suffix.lower() != ".mov":
            raise ValueError("School output must be a .mov file")
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {self.path}; use --overwrite explicitly")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex}.partial.mov")
        self.overwrite = overwrite
        self.closed = False
        self.frames = 0
        self.error_log = tempfile.TemporaryFile()
        self.command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(FPS), "-i", "-", "-an",
            "-vf", REC709_FILTER,
            "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-level:v", "5.0",
            "-pix_fmt", "yuv420p", "-b:v", f"{BITRATE_MBPS}M",
            "-maxrate", "120M", "-bufsize", "240M", "-x264-params", "force-cfr=1",
            "-r", str(FPS), "-fps_mode", "cfr",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-color_range", "tv", "-chroma_sample_location", "left",
            "-movflags", "+faststart+write_colr", "-f", "mov", str(self.partial),
        ]
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=self.error_log)
        except BaseException:
            self.error_log.close()
            raise

    def _error(self):
        self.error_log.seek(0)
        details = self.error_log.read().decode("utf-8", errors="replace").strip()
        return RuntimeError("School video encoder failed" + (f": {details[-4000:]}" if details else ""))

    def write(self, pixels):
        if self.closed:
            raise RuntimeError("Movie writer is closed")
        pixels = np.asarray(pixels)
        if pixels.dtype != np.uint8 or pixels.shape not in ((HEIGHT, WIDTH, 3), (HEIGHT, WIDTH, 4)):
            raise ValueError(f"School frames must be {WIDTH}x{HEIGHT} uint8 RGB or RGBA")
        try:
            self.process.stdin.write(np.ascontiguousarray(pixels[..., :3]).tobytes())
        except BrokenPipeError as error:
            self.process.wait()
            raise self._error() from error
        self.frames += 1

    def close(self, commit=True):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            if not commit or code or not self.frames:
                self.partial.unlink(missing_ok=True)
                if commit:
                    raise self._error()
            elif self.path.exists() and not self.overwrite:
                self.partial.unlink(missing_ok=True)
                raise FileExistsError(f"Output appeared during rendering: {self.path}")
            else:
                self.partial.replace(self.path)
        finally:
            self.error_log.close()


def render_rehearsal(tape_path, output_path, *, device="auto", overwrite=False, progress=print):
    """Replay real viewer controls offline, including the visible ImGui UI.

    The output contains one second of black at each end. Its audio and caption
    tracks are intentionally absent: narration and matching captions must come
    from the presenter. Every content frame samples the tape at an exact 1/24 s.
    """
    from rendercanvas.offscreen import RenderCanvas
    from .school_timeline import load_timeline
    from .school_viewer import SchoolViewer

    tape_path = Path(tape_path)
    output_path = Path(output_path)
    timeline = load_timeline(tape_path)
    if not math.isfinite(timeline.duration) or not 0 < timeline.duration <= MAX_CONTENT_SECONDS:
        raise ValueError("Rehearsal must contain more than zero and at most 298 seconds")
    if output_path.suffix.lower() != ".mov":
        raise ValueError("School output must be a .mov file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite explicitly")
    initial_scene = Path(timeline.initial_scene)
    if not initial_scene.is_absolute():
        initial_scene = tape_path.parent / initial_scene
    if not initial_scene.is_file():
        raise FileNotFoundError(f"Rehearsal scene checkpoint is missing: {initial_scene}")

    content_frames = math.ceil(timeline.duration * FPS - 1e-9)
    black_frames = BLACK_SECONDS * FPS
    total_frames = content_frames + 2 * black_frames
    if total_frames > 300 * FPS:
        raise ValueError("School movie would exceed five minutes")
    canvas = RenderCanvas(size=(WIDTH, HEIGHT), pixel_ratio=1)
    viewer = None
    writer = None
    complete = False
    started = time.perf_counter()
    try:
        viewer = SchoolViewer(
            device=device, canvas=canvas, scene=initial_scene,
            replay=True, render_resolution=RENDER_RESOLUTION,
        )
        writer = School_movie_writer(output_path, overwrite=overwrite)
        black = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for _ in range(black_frames):
            writer.write(black)
        progress(f"Rendering School rehearsal: {content_frames} content frames, 1920x1080 at 24 fps")
        for index in range(content_frames):
            elapsed = index / FPS
            viewer.apply_rehearsal_sample(timeline.sample(elapsed))
            viewer.replay_elapsed = 0.0 if index == 0 else 1 / FPS
            previous_frames = viewer.frames
            pixels = canvas.draw()
            # Rendercanvas logs draw callback failures rather than propagating
            # them; never silently encode its cached image as a fresh frame.
            if pixels is None or viewer.frames != previous_frames + 1:
                raise RuntimeError(f"School viewer did not render frame {index}")
            writer.write(pixels)
            if index and index % (FPS * 5) == 0:
                progress(f"  {elapsed:.0f}/{timeline.duration:g} s rendered")
        for _ in range(black_frames):
            writer.write(black)
        complete = True
    finally:
        try:
            if writer is not None:
                writer.close(commit=complete)
        finally:
            if viewer is not None:
                viewer.executor.shutdown(wait=True, cancel_futures=True)
            canvas.close()

    manifest = {
        "type": "school_rehearsal", "source": str(tape_path.resolve()),
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": total_frames,
        "content_frames": content_frames, "content_seconds": content_frames / FPS,
        "black_head_seconds": BLACK_SECONDS, "black_tail_seconds": BLACK_SECONDS,
        "duration_seconds": total_frames / FPS, "wave_resolution": RENDER_RESOLUTION,
        "codec": "h264", "container": "mov", "bitrate_mbps": BITRATE_MBPS,
        "pixel_format": "yuv420p", "color_primaries": "bt709", "color_transfer": "bt709",
        "color_matrix": "bt709", "color_range": "limited",
        "audio": None, "captions": None, "render_seconds": time.perf_counter() - started,
    }
    manifest_path = output_path.with_name(output_path.name + ".json")
    partial_manifest = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.partial")
    try:
        partial_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        partial_manifest.replace(manifest_path)
    finally:
        partial_manifest.unlink(missing_ok=True)
    progress(f"Saved {output_path} ({manifest['duration_seconds']:.3f} s, including black head and tail)")
    return manifest
