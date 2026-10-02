# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""GPU-only conversion of RGB foam history into filtered material coverage.

Prepend ``FOAM_SURFACE_WGSL`` to the ocean shader to use the same
``foam_surface(density, grain, strength) -> vec2(coverage, coverage*freshness)``
for close fragments. B stores coverage*grain; A stores optional crest area
outside that history coverage, integrated before filtering.

``update`` always encodes a new base pass and its linear mips: a texture object
can retain its identity while its contents change. The caller owns dirtiness,
including wave/source changes while paused. Bind groups, views and allocations
are reused until an input texture or its dimensions change. The caller submits
the supplied encoder and must rebind the returned texture after a size change.
One material instance represents one currently displayed foam history.
"""
from pathlib import Path
import math
import struct

import wgpu


_SHADERS = Path(__file__).parent / "shaders"
FOAM_SURFACE_WGSL = (_SHADERS / "foam_surface.wgsl").read_text()


class Foam_material:
    """Own the prefiltered RGBA16F coverage texture, without CPU field access."""

    def __init__(self, device):
        self.device = device
        shader = device.create_shader_module(
            label="Foam coverage and mip filtering",
            code=FOAM_SURFACE_WGSL + "\n" + (_SHADERS / "foam_material.wgsl").read_text())
        self.base_pipeline = device.create_render_pipeline(
            label="Foam subpixel coverage",
            layout="auto",
            vertex={"module": shader, "entry_point": "foam_fullscreen"},
            fragment={"module": shader, "entry_point": "foam_integrate",
                      "targets": [{"format": "rgba16float"}]},
            primitive={"topology": "triangle-list"})
        self.mip_pipeline = device.create_render_pipeline(
            label="Foam coverage mip",
            layout="auto",
            vertex={"module": shader, "entry_point": "foam_fullscreen"},
            fragment={"module": shader, "entry_point": "foam_mip",
                      "targets": [{"format": "rgba16float"}]},
            primitive={"topology": "triangle-list"})
        self.sampler = device.create_sampler(
            address_mode_u="repeat", address_mode_v="repeat",
            mag_filter="linear", min_filter="linear", mipmap_filter="nearest")
        self.uniform = device.create_buffer(
            label="Foam material strength", size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.texture = None
        self._raw_texture = None
        self._crest_texture = None
        self._base_group = None
        self._base_view = None
        self._mips = []
        self._settings = None

    def _allocate(self, width, height):
        if self.texture is not None:
            self.texture.destroy()
        self.texture = self.device.create_texture(
            label="Prefiltered foam material", size=(width, height, 1),
            format="rgba16float", mip_level_count=max(width, height).bit_length(),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.RENDER_ATTACHMENT
                  | wgpu.TextureUsage.COPY_SRC)
        self._base_view = self.texture.create_view(base_mip_level=0, mip_level_count=1)
        self._mips = []
        for level in range(1, self.texture.mip_level_count):
            source = self.texture.create_view(base_mip_level=level-1, mip_level_count=1)
            target = self.texture.create_view(base_mip_level=level, mip_level_count=1)
            group = self.device.create_bind_group(
                layout=self.mip_pipeline.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": source},
                    {"binding": 1, "resource": self.sampler}])
            self._mips.append((group, target))

    @staticmethod
    def _draw(encoder, pipeline, group, target):
        render_pass = encoder.begin_render_pass(color_attachments=[{
            "view": target, "load_op": "clear", "store_op": "store",
            "clear_value": (0, 0, 0, 1)}])
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, group)
        render_pass.draw(3)
        render_pass.end()

    def update(self, encoder, raw_texture, strength=1.0, *, crest_texture=None,
               crest_threshold=.75, crest_width=.15):
        """Encode coverage and all its mips, returning this instance's texture.

        Raw RGB must contain finite aeration density; alpha supplies periodic
        grain. A 1x1 zero raw texture naturally produces disabled coverage.
        Optional wave displacement alpha is negative minimum horizontal stretch.
        No texture readback, queue submission, or synchronization occurs here.
        """
        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 3.4028234663852886e38:
            raise ValueError("Foam strength must be finite, nonnegative and fit float32")
        if not math.isfinite(crest_threshold) or not 0 <= crest_threshold <= 1:
            raise ValueError("Crest threshold must be in [0, 1]")
        if not math.isfinite(crest_width) or not 0 < crest_width <= 1:
            raise ValueError("Crest width must be in (0, 1]")
        width, height, layers = raw_texture.size
        if raw_texture.format != "rgba16float" or layers != 1:
            raise ValueError("Foam material requires a single RGBA16F texture")
        if width & (width-1) or height & (height-1):
            raise ValueError("Foam material dimensions must be powers of two, including 1")
        settings = (strength, crest_threshold, crest_width, float(crest_texture is not None))
        crest_input = crest_texture if crest_texture is not None else raw_texture
        if self.texture is None or self.texture.size[:2] != (width, height):
            self._allocate(width, height)
        if self._raw_texture is not raw_texture or self._crest_texture is not crest_input:
            self._raw_texture = raw_texture
            self._crest_texture = crest_input
            self._base_group = self.device.create_bind_group(
                layout=self.base_pipeline.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": raw_texture.create_view(
                        base_mip_level=0, mip_level_count=1)},
                    {"binding": 1, "resource": self.sampler},
                    {"binding": 2, "resource": {"buffer": self.uniform}},
                    {"binding": 3, "resource": crest_input.create_view(
                        base_mip_level=0, mip_level_count=1)}])
        if self._settings != settings:
            self.device.queue.write_buffer(self.uniform, 0, struct.pack("<4f", *settings))
            self._settings = settings
        self._draw(encoder, self.base_pipeline, self._base_group, self._base_view)
        for group, target in self._mips:
            self._draw(encoder, self.mip_pipeline, group, target)
        return self.texture

    def destroy(self):
        """Release owned textures and uniforms; raw history belongs to the caller."""
        if self.texture is not None:
            self.texture.destroy()
            self.texture = None
        self.uniform.destroy()
        self._raw_texture = None
        self._crest_texture = None
        self._base_group = None
        self._base_view = None
        self._mips = []
