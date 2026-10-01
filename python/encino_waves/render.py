# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Native Metal / Vulkan / D3D12 ocean renderer, also usable without a window."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import wgpu
from .model import texture_arrays
from .sky import load_sky
from .camera import Camera


@dataclass(frozen=True)
class Look:
    exposure: float = 0.0
    sky_rotation: float = 0.0
    sky_gain: float = 2.0
    haze: float = 1.0
    foam: float = 1.0
    crest_threshold: float = 0.5
    crest_maximum: float = 1.1


@dataclass(frozen=True)
class Shading_statistics:
    big_height: float
    crest_gain: float
    crest_bias: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.big_height,self.crest_gain,self.crest_bias)):
            raise ValueError("Shading statistics must be finite")
        if self.big_height<=0 or self.crest_gain<=0:
            raise ValueError("Shading height and crest gain must be positive")


def make_device(canvas=None):
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance", canvas=canvas)
    if adapter is None or adapter.info["adapter_type"] == "CPU":
        raise RuntimeError("No hardware graphics adapter found; a Metal/Vulkan/DX12 GPU is required.")
    return adapter.request_device_sync()


class Ocean_renderer:
    """Owns graphics resources only; the core model has no renderer dependency."""
    def __init__(self, device, sky=None, mesh_resolution=(640,384), target_format="rgba8unorm"):
        self.device = device
        self.format = target_format
        self.mesh_resolution = mesh_resolution
        self.uniform = device.create_buffer(size=176, usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)
        self.wave_sampler = device.create_sampler(address_mode_u="repeat", address_mode_v="repeat",
                                                 mag_filter="linear", min_filter="linear", mipmap_filter="linear")
        self.sky_sampler = device.create_sampler(address_mode_u="repeat", address_mode_v="clamp-to-edge",
                                                mag_filter="linear", min_filter="linear", mipmap_filter="linear")
        shader = device.create_shader_module(code=(Path(__file__).parent/"shaders/ocean.wgsl").read_text())
        # Explicit layout is shared by both pipelines, including unused entries.
        self.layout = device.create_bind_group_layout(entries=[
            {"binding":0,"visibility":3,"buffer":{"type":"uniform"}},
            {"binding":1,"visibility":3,"sampler":{"type":"filtering"}},
            {"binding":2,"visibility":3,"texture":{"sample_type":"float"}},
            {"binding":3,"visibility":2,"texture":{"sample_type":"float"}},
            {"binding":4,"visibility":2,"sampler":{"type":"filtering"}},
            {"binding":5,"visibility":2,"texture":{"sample_type":"float"}},
        ])
        layout = device.create_pipeline_layout(bind_group_layouts=[self.layout])
        self.sky_pipeline = device.create_render_pipeline(layout=layout,
            vertex={"module":shader,"entry_point":"sky_vertex"},
            fragment={"module":shader,"entry_point":"sky_fragment","targets":[{"format":target_format}]},
            primitive={"topology":"triangle-list"},
            depth_stencil={"format":"depth32float","depth_write_enabled":False,"depth_compare":"always"})
        self.ocean_pipeline = device.create_render_pipeline(layout=layout,
            vertex={"module":shader,"entry_point":"ocean_vertex"},
            fragment={"module":shader,"entry_point":"ocean_fragment","targets":[{"format":target_format}]},
            primitive={"topology":"triangle-list","cull_mode":"none"},
            depth_stencil={"format":"depth32float","depth_write_enabled":True,"depth_compare":"less"})
        mip_shader = device.create_shader_module(code=(Path(__file__).parent/"shaders/mipmap.wgsl").read_text())
        self.mip_pipeline = device.create_render_pipeline(layout="auto",
            vertex={"module":mip_shader,"entry_point":"vertex"},
            fragment={"module":mip_shader,"entry_point":"fragment","targets":[{"format":"rgba16float"}]})
        nx,ny = mesh_resolution
        cells = (np.arange(ny,dtype=np.uint32)[:,None]*(nx+1) + np.arange(nx,dtype=np.uint32)[None,:]).ravel()
        indices = np.stack((cells,cells+1,cells+nx+1,cells+1,cells+nx+2,cells+nx+1),axis=1).ravel()
        self.index_count = indices.size
        self.indices = device.create_buffer_with_data(data=indices,usage=wgpu.BufferUsage.INDEX)
        self.depth = None
        self.wave_size = 0
        self.wave_textures = []
        self.wave_mips = []
        sky_pixels,self.sky_name = load_sky(sky)
        self.sky = self._make_texture(sky_pixels.shape[1],sky_pixels.shape[0])
        self.sky_storage_scale=max(1.0,float(np.max(sky_pixels[...,:3]))/60000)
        self._write_texture(self.sky,(sky_pixels/self.sky_storage_scale).astype(np.float16))
        encoder = device.create_command_encoder()
        self._mipmaps(encoder,self._make_mip_groups(self.sky))
        device.queue.submit([encoder.finish()])
        # Dominant upper-hemisphere light and broad light color, in linear HDR.
        rgb = sky_pixels[...,:3]
        luminance = rgb@np.array([.2126,.7152,.0722])
        luminance[luminance.shape[0]//2:] = 0
        yy,xx = np.unravel_index(np.argmax(luminance),luminance.shape)
        lat = math.pi/2-(yy+.5)/rgb.shape[0]*math.pi
        lon = (xx+.5)/rgb.shape[1]*2*math.pi-math.pi
        self.sun_direction = np.array([math.cos(lon)*math.cos(lat),math.sin(lon)*math.cos(lat),math.sin(lat)])
        self.sun_color = rgb[yy,xx].copy()
        self.moon_color = rgb[yy,(xx+rgb.shape[1]//2)%rgb.shape[1]].copy()
        self.statistics_parameters = None
        self.frame = None

    def _make_texture(self,width,height):
        return self.device.create_texture(size=(width,height,1),format="rgba16float",
            mip_level_count=int(math.log2(max(width,height)))+1,
            usage=wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST|wgpu.TextureUsage.RENDER_ATTACHMENT)

    def _write_texture(self,texture,data):
        self.device.queue.write_texture({"texture":texture},data,
            {"bytes_per_row":data.shape[1]*8,"rows_per_image":data.shape[0]},
            (data.shape[1],data.shape[0],1))

    def _make_mip_groups(self,texture):
        groups = []
        for level in range(1,texture.mip_level_count):
            source = texture.create_view(base_mip_level=level-1,mip_level_count=1)
            target = texture.create_view(base_mip_level=level,mip_level_count=1)
            group = self.device.create_bind_group(layout=self.mip_pipeline.get_bind_group_layout(0),entries=[
                {"binding":0,"resource":source}, {"binding":1,"resource":self.wave_sampler}])
            groups.append((group,target))
        return groups

    def _mipmaps(self,encoder,groups):
        for group,target in groups:
            render_pass = encoder.begin_render_pass(color_attachments=[
                {"view":target,"load_op":"clear","store_op":"store","clear_value":(0,0,0,0)}])
            render_pass.set_pipeline(self.mip_pipeline)
            render_pass.set_bind_group(0,group)
            render_pass.draw(3)
            render_pass.end()

    def upload(self,frame):
        maps = texture_arrays(frame)
        n = maps[0].shape[0]
        if n != self.wave_size:
            for texture in self.wave_textures:
                texture.destroy()
            self.wave_size = n
            self.wave_textures = [self._make_texture(n,n) for _ in range(2)]
            self.wave_mips = [self._make_mip_groups(texture) for texture in self.wave_textures]
            self.bind_group = self.device.create_bind_group(layout=self.layout,entries=[
                {"binding":0,"resource":{"buffer":self.uniform}},
                {"binding":1,"resource":self.wave_sampler},
                {"binding":2,"resource":self.wave_textures[0].create_view()},
                {"binding":3,"resource":self.wave_textures[1].create_view()},
                {"binding":4,"resource":self.sky_sampler},
                {"binding":5,"resource":self.sky.create_view()},
            ])
        for texture,data in zip(self.wave_textures,maps):
            self._write_texture(texture,data)
        if self.statistics_parameters != frame.parameters:
            # Original viewer takes these statistics once after initialization.
            height=maps[0][...,2].astype(np.float32)
            crest=maps[0][...,3].astype(np.float32)
            self.big_height=max(.001,1.5*float(np.max(np.abs(height))))
            self.crest_gain=1/max(1e-8,2*float(np.std(crest)))
            self.crest_bias=-float(np.mean(crest))*self.crest_gain
            self.statistics_parameters=frame.parameters
        self.frame = frame
        self.mips_dirty = True

    @property
    def shading_statistics(self):
        return Shading_statistics(self.big_height,self.crest_gain,self.crest_bias)

    def restore_shading_statistics(self,statistics):
        self.big_height=statistics.big_height
        self.crest_gain=statistics.crest_gain
        self.crest_bias=statistics.crest_bias

    def draw(self,target_view,width,height,camera=Camera(),look=Look(),viewport=None,clear=True):
        if self.frame is None:
            raise RuntimeError("Upload a wave frame before drawing")
        if self.depth is None or self.depth.size[:2] != (width,height):
            if self.depth:
                self.depth.destroy()
            self.depth = self.device.create_texture(size=(width,height,1),format="depth32float",usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        forward,right,up = camera.basis()
        aspect = (viewport[2]/viewport[3]) if viewport else width/height
        rotation = math.radians(look.sky_rotation)
        sx,sy,sz = self.sun_direction
        sun = (sx*math.cos(rotation)+sy*math.sin(rotation),sy*math.cos(rotation)-sx*math.sin(rotation),sz)
        uniform = np.array([
            [camera.x,camera.y,camera.height,self.frame.time],
            [*forward,math.tan(math.radians(camera.fov)/2)],
            [*right,aspect],[*up,look.exposure],
            [self.frame.parameters.domain,self.wave_size,look.crest_threshold,look.foam],
            [rotation,look.haze,look.sky_gain*self.sky_storage_scale,0],[*sun,0],[*(self.sun_color*look.sky_gain),0],
            [*self.mesh_resolution,.5,30000],
            [self.big_height,self.crest_gain,self.crest_bias,look.crest_maximum],
            [*(self.moon_color*look.sky_gain),0],
        ],np.float32)
        self.device.queue.write_buffer(self.uniform,0,uniform)
        encoder = self.device.create_command_encoder()
        if self.mips_dirty:
            for groups in self.wave_mips:
                self._mipmaps(encoder,groups)
            self.mips_dirty = False
        render_pass = encoder.begin_render_pass(color_attachments=[
            {"view":target_view,"clear_value":(0,0,0,1),"load_op":"clear" if clear else "load","store_op":"store"}],
            depth_stencil_attachment={"view":self.depth.create_view(),"depth_clear_value":1.0,
                                      "depth_load_op":"clear","depth_store_op":"store"})
        if viewport:
            render_pass.set_viewport(*viewport,0,1)
            render_pass.set_scissor_rect(*(int(v) for v in viewport))
        render_pass.set_bind_group(0,self.bind_group)
        render_pass.set_pipeline(self.sky_pipeline)
        render_pass.draw(3)
        render_pass.set_pipeline(self.ocean_pipeline)
        render_pass.set_index_buffer(self.indices,"uint32")
        render_pass.draw_indexed(self.index_count)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

    def render_image(self,width=1920,height=1080,camera=Camera(),look=Look(),left_renderer=None):
        target = self.device.create_texture(size=(width,height,1),format=self.format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
        if left_renderer is not None:
            half=width//2
            left_renderer.draw(target.create_view(),width,height,camera,look,viewport=(0,0,half,height))
            self.draw(target.create_view(),width,height,camera,look,viewport=(half,0,width-half,height),clear=False)
        else:
            self.draw(target.create_view(),width,height,camera,look)
        result = read_rgba(self.device,target,width,height)
        target.destroy()
        return result


def read_rgba(device,texture,width,height):
    stride = (width*4+255)//256*256
    data = device.queue.read_texture({"texture":texture},{"bytes_per_row":stride,"rows_per_image":height},(width,height,1))
    rgba = np.frombuffer(data,np.uint8).reshape(height,stride)[:,:width*4].reshape(height,width,4).copy()
    if texture.format.startswith("bgra"):
        rgba = rgba[:,:,[2,1,0,3]]
    return rgba
