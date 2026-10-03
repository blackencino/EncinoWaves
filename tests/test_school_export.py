"""Fixed-rate School playback, safe publication and real encoder color output."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

from encino_waves import school_export as export
from encino_waves.school_timeline import Rehearsal


def test_encoder_writes_real_1080p24_rec709_mov_and_transforms_rgb(tmp_path):
    path = tmp_path / "school.mov"
    writer = export.School_movie_writer(path)
    for value in (0, 128, 255):
        writer.write(np.full((1080, 1920, 3), value, np.uint8))
    writer.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    assert "major_brand     : qt" in result.stderr
    assert "h264 (High)" in result.stderr
    assert "yuv420p(tv, bt709, progressive)" in result.stderr
    assert "1920x1080 [SAR 1:1 DAR 16:9]" in result.stderr
    assert "24 fps" in result.stderr
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True, check=True,
    )
    planes = np.frombuffer(decoded.stdout, np.uint8).reshape(3, 1080 * 1920 * 3 // 2)
    luma = planes[:, :1080 * 1920]
    assert np.median(luma[0]) == 16
    assert np.median(luma[2]) == 235
    # zscale's display-referred Rec.709 conversion changes the sRGB midtone;
    # a matrix-only conversion with misleading tags would produce Y=126.
    assert 130 <= np.median(luma[1]) <= 133
    assert np.max(np.abs(planes[:, 1080 * 1920:].astype(int) - 128)) <= 1
    assert not list(tmp_path.glob("*.partial.mov"))


def test_aborted_encode_preserves_existing_destination(tmp_path):
    path = tmp_path / "school.mov"
    path.write_bytes(b"previous approved movie")
    with pytest.raises(FileExistsError):
        export.School_movie_writer(path)
    writer = export.School_movie_writer(path, overwrite=True)
    writer.write(np.zeros((1080, 1920, 3), np.uint8))
    writer.close(commit=False)
    assert path.read_bytes() == b"previous approved movie"
    assert not writer.partial.exists()


@pytest.fixture
def replay_stub(monkeypatch):
    import rendercanvas.offscreen
    from encino_waves import school_viewer
    result = SimpleNamespace(viewers=[], canvases=[], writers=[], fail_at=None)

    class Canvas:
        def __init__(self, **kwargs):
            assert kwargs == {"size": (1920, 1080), "pixel_ratio": 1}
            self.closed = False
            result.canvases.append(self)
        def draw(self):
            if result.fail_at != self.viewer.frames:
                self.viewer.frames += 1
            return np.full((1080, 1920, 4), 128, np.uint8)
        def close(self): self.closed = True

    class Viewer:
        def __init__(self, **kwargs):
            self.options = kwargs
            kwargs["canvas"].viewer = self
            self.frames = 0
            self.samples = []
            self.deltas = []
            self.shutdown = False
            self.executor = SimpleNamespace(shutdown=self._shutdown)
            result.viewers.append(self)
        def _shutdown(self, **kwargs): self.shutdown = True
        def apply_rehearsal_sample(self, values): self.samples.append(values)
        @property
        def replay_elapsed(self): return self.deltas[-1]
        @replay_elapsed.setter
        def replay_elapsed(self, value): self.deltas.append(value)

    class Writer:
        def __init__(self, path, **kwargs):
            self.frames = []
            self.committed = None
            result.writers.append(self)
        def write(self, pixels): self.frames.append(int(pixels[0, 0, 0]))
        def close(self, commit=True): self.committed = commit

    monkeypatch.setattr(rendercanvas.offscreen, "RenderCanvas", Canvas)
    monkeypatch.setattr(school_viewer, "SchoolViewer", Viewer)
    monkeypatch.setattr(export, "School_movie_writer", Writer)
    return result


def _tape(tmp_path):
    (tmp_path / "initial.json").write_text("{}")
    tape = Rehearsal("initial.json", {"playing": True, "speed": 1.0})
    tape.append(1 / 24, {"playing": False, "speed": 1.0})
    path = tmp_path / "rehearsal.json"
    tape.save(path, .08)
    return path


def test_replay_uses_exact_frame_times_real_ui_and_black_head_tail(replay_stub, tmp_path):
    result = export.render_rehearsal(_tape(tmp_path), tmp_path / "out.mov", progress=lambda _: None)
    viewer = replay_stub.viewers[0]
    assert viewer.options["replay"] is True
    assert viewer.options["render_resolution"] == 4096
    assert viewer.options["scene"] == tmp_path / "initial.json"
    assert viewer.deltas == [0, 1 / 24]
    assert [sample["playing"] for sample in viewer.samples] == [True, False]
    assert replay_stub.writers[0].frames == [0] * 24 + [128] * 2 + [0] * 24
    assert replay_stub.writers[0].committed is True
    assert viewer.shutdown and replay_stub.canvases[0].closed
    assert result["frames"] == 50 and result["duration_seconds"] == 50 / 24
    assert result["fps"] == 24
    assert result["audio"] is None and result["captions"] is None


def test_callback_failure_does_not_encode_stale_frame(replay_stub, tmp_path):
    replay_stub.fail_at = 1
    with pytest.raises(RuntimeError, match="did not render frame 1"):
        export.render_rehearsal(_tape(tmp_path), tmp_path / "out.mov", progress=lambda _: None)
    assert replay_stub.writers[0].committed is False
    assert replay_stub.viewers[0].shutdown and replay_stub.canvases[0].closed
    assert not (tmp_path / "out.mov.json").exists()
