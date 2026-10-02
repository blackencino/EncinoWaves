# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Startup-only GGX filtering of a linear latitude/longitude environment.

This is the normalized, ``N = V = reflection`` half of split-sum specular IBL.
It does not contain Fresnel or the integrated visibility term: the ocean shader
must multiply its lookup by an environment BRDF response exactly once.

Source: Brian Karis, Real Shading in Unreal Engine 4 (2013),
https://blog.selfshadow.com/publications/s2013-shading-course/ ; prefiltered
importance sampling and its PDF in Filament,
https://google.github.io/filament/main/filament.html#annex/importancesamplingfortheibl .

Output mip ``i`` represents PERCEPTUAL roughness ``i/(mip_count-1)``; the GGX
microfacet parameter is its square. Ordinary source mips are only used to filter
Monte Carlo sample footprints. They are not mistaken for roughness levels.
"""

import math
from pathlib import Path

import numpy as np
import wgpu


def prefilter_environment(device, source_texture, *, sample_count=128, max_width=1024):
    """Return an independent RGBA16F GGX latlong mip chain, at most 1024x512.

    ``source_texture`` must be a filterable 2D RGBA16F radiance texture with its
    ordinary mip chain already initialized. The source is sampled unchanged.
    Longitude repeats, latitude clamps, and source LOD uses each sample's PDF
    and local spherical texel area rather than a cubemap's uniform-area formula.

    ``max_width`` may reduce the default limit for thumbnails/tests. Smaller
    sources are never enlarged. A separate immutable uniform buffer is used per
    output mip; all passes may safely be encoded in a single queue submission.
    No GPU readback or wave/simulation resources are involved.
    """
    if not isinstance(sample_count, int) or not 16 <= sample_count <= 1024:
        raise ValueError("Environment sample count must be an integer from 16 to 1024")
    if not isinstance(max_width, int) or not 2 <= max_width <= 1024:
        raise ValueError("Environment maximum width must be an integer from 2 to 1024")
    source_width, source_height, layers = source_texture.size
    if layers != 1 or source_texture.format != "rgba16float":
        raise ValueError("Environment source must be a 2D RGBA16F texture")
    scale = min(1., max_width/source_width, max(1, max_width//2)/source_height)
    width, height = max(1, int(source_width*scale)), max(1, int(source_height*scale))
    levels = int(math.log2(max(width, height)))+1
    result = device.create_texture(
        label="GGX environment", size=(width, height, 1), format="rgba16float",
        mip_level_count=levels,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.RENDER_ATTACHMENT
              | wgpu.TextureUsage.COPY_SRC)
    module = device.create_shader_module(
        label="GGX environment filter",
        code=(Path(__file__).parent/"shaders/environment_filter.wgsl").read_text())
    pipeline = device.create_render_pipeline(
        label="GGX environment filter", layout="auto",
        vertex={"module":module, "entry_point":"vertex"},
        fragment={"module":module, "entry_point":"fragment",
                  "targets":[{"format":"rgba16float"}]},
        primitive={"topology":"triangle-list"})
    sampler = device.create_sampler(
        label="Spherical environment sampler", address_mode_u="repeat",
        address_mode_v="clamp-to-edge", mag_filter="linear", min_filter="linear",
        mipmap_filter="linear")
    source_view = source_texture.create_view()
    layout = pipeline.get_bind_group_layout(0)
    encoder = device.create_command_encoder(label="Prefilter HDR environment")
    # Retain Python references until submission as well as native bind references.
    buffers, groups = [], []
    for level in range(levels):
        roughness = level/max(levels-1, 1)
        parameters = np.array([
            source_width, source_height, source_texture.mip_level_count-1,
            max(0., math.log2(source_width/width), math.log2(source_height/height)),
            roughness, sample_count, 0, 0,
        ], np.float32)
        uniform = device.create_buffer_with_data(
            label=f"GGX mip {level} parameters", data=parameters, usage=wgpu.BufferUsage.UNIFORM)
        group = device.create_bind_group(layout=layout, entries=[
            {"binding":0, "resource":{"buffer":uniform}},
            {"binding":1, "resource":sampler},
            {"binding":2, "resource":source_view},
        ])
        buffers.append(uniform)
        groups.append(group)
        target = result.create_view(base_mip_level=level, mip_level_count=1)
        render_pass = encoder.begin_render_pass(color_attachments=[{
            "view":target, "load_op":"clear", "store_op":"store",
            "clear_value":(0, 0, 0, 1),
        }])
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, group)
        render_pass.draw(3)
        render_pass.end()
    device.queue.submit([encoder.finish()])
    return result
