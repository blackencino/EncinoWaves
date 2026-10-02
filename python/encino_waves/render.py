# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Native Metal / Vulkan / D3D12 ocean renderer, also usable without a window."""
from dataclasses import dataclass
from pathlib import Path
import math
import os
import warnings
import numpy as np
import torch
import wgpu
from .model import texture_arrays
from .sky import load_sky, ocean_dome
from .lighting import analyze_sky
from .environment import prefilter_environment
from .foam_material import Foam_material, FOAM_SURFACE_WGSL
from .foam_detail import Foam_detail, FOAM_DETAIL_WGSL
from .camera import Camera


@dataclass(frozen=True)
class Look:
    exposure: float = 0.55
    sky_rotation: float = 25.0
    sky_gain: float = 1.0
    haze: float = 0.9
    foam: float = 1.0
    crest_threshold: float = 0.5
    crest_maximum: float = 1.1
    aeration: float = 1.25
    material: str = "physical"

    @classmethod
    def from_dict(cls, values):
        # The immediate crest overlay was added with the wind-streak experiment.
        # Older scenes retain lighting settings while using the restored foam.
        return cls(**{name:value for name,value in values.items()
                      if name not in {"crest_foam", "crest_breakup"}})


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
    def __init__(self, device, sky=None, mesh_resolution=(640,384), target_format="rgba8unorm", transfer=None, sample_count=4):
        self.device = device
        self.transfer_mode = transfer or os.environ.get("ENCINO_WAVES_TRANSFER","auto")
        if self.transfer_mode not in ("auto","metal","host"):
            raise ValueError("Texture transfer must be auto, metal, or host")
        self.gpu_transfer = None
        self.transfer_device = None
        self.format = target_format
        if sample_count not in (1,4):
            raise ValueError("Antialiasing sample count must be 1 or 4")
        self.sample_count = sample_count
        self._physical_ready = device.adapter.info.get("backend_type") != "Metal"
        self.mesh_resolution = mesh_resolution
        self.uniform = device.create_buffer(size=496, usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)
        self.wave_sampler = device.create_sampler(address_mode_u="repeat", address_mode_v="repeat",
                                                 mag_filter="linear", min_filter="linear", mipmap_filter="linear",
                                                 max_anisotropy=8)
        self.legacy_wave_sampler = device.create_sampler(address_mode_u="repeat",address_mode_v="repeat",
            mag_filter="linear",min_filter="linear",mipmap_filter="linear")
        self.sky_sampler = device.create_sampler(address_mode_u="repeat", address_mode_v="clamp-to-edge",
                                                mag_filter="linear", min_filter="linear", mipmap_filter="linear")
        shader = device.create_shader_module(code=FOAM_SURFACE_WGSL+"\n"+FOAM_DETAIL_WGSL+"\n"+
            (Path(__file__).parent/"shaders/ocean.wgsl").read_text())
        # Explicit layout is shared by both pipelines, including unused entries.
        self.layout = device.create_bind_group_layout(entries=[
            {"binding":0,"visibility":3,"buffer":{"type":"uniform"}},
            {"binding":1,"visibility":3,"sampler":{"type":"filtering"}},
            {"binding":2,"visibility":3,"texture":{"sample_type":"float"}},
            {"binding":3,"visibility":2,"texture":{"sample_type":"float"}},
            {"binding":4,"visibility":2,"sampler":{"type":"filtering"}},
            {"binding":5,"visibility":2,"texture":{"sample_type":"float"}},
            {"binding":6,"visibility":2,"texture":{"sample_type":"float"}},
            {"binding":7,"visibility":2,"texture":{"sample_type":"float"}},
            {"binding":8,"visibility":2,"texture":{"sample_type":"float"}},
            {"binding":9,"visibility":2,"texture":{"sample_type":"float"}},
        ])
        layout = device.create_pipeline_layout(bind_group_layouts=[self.layout])
        self.sky_pipeline = device.create_render_pipeline(layout=layout,
            vertex={"module":shader,"entry_point":"sky_vertex"},
            fragment={"module":shader,"entry_point":"sky_fragment","targets":[{"format":"rgba16float"}]},
            primitive={"topology":"triangle-list"},
            multisample={"count":sample_count},
            depth_stencil={"format":"depth32float","depth_write_enabled":False,"depth_compare":"always"})
        self.ocean_pipeline = device.create_render_pipeline(layout=layout,
            vertex={"module":shader,"entry_point":"ocean_vertex"},
            fragment={"module":shader,"entry_point":"ocean_fragment","targets":[{"format":"rgba16float"}]},
            primitive={"topology":"triangle-list","cull_mode":"none"},
            multisample={"count":sample_count},
            depth_stencil={"format":"depth32float","depth_write_enabled":True,"depth_compare":"less"})
        self.pipeline_layout = layout
        self.legacy_pipelines = None
        presentation = device.create_shader_module(code=(Path(__file__).parent/"shaders/presentation.wgsl").read_text())
        self.presentation_pipeline = device.create_render_pipeline(layout="auto",
            vertex={"module":presentation,"entry_point":"vertex"},
            fragment={"module":presentation,"entry_point":"fragment","targets":[{"format":target_format}]})
        self.presentation_sampler = device.create_sampler(mag_filter="linear",min_filter="linear")
        self.hdr = None
        self.multisample = None
        self.presentation_group = None
        self.depth_sample_count = 0
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
        self.foam_state = None
        self.foam_texture = self._make_texture(1,1)
        self._write_texture(self.foam_texture,np.zeros((1,1,4),np.float16))
        self.foam_mips = []
        self.foam_dirty = False
        self.foam_material = Foam_material(device)
        self.foam_material_texture = None
        self.foam_material_strength = None
        self.foam_material_dirty = True
        self.foam_detail = Foam_detail(device)
        self.foam_grain_basis = None
        self.foam_grain = None
        raw_sky,self.sky_name = load_sky(sky)
        self.raw_sky = raw_sky
        self.sky_peak = float(np.max(raw_sky[...,:3]))
        sky_pixels = ocean_dome(raw_sky,horizon_trim_degrees=10.0)
        self.lighting = analyze_sky(sky_pixels)
        self.sky = self._make_texture(sky_pixels.shape[1],sky_pixels.shape[0])
        self.sky_storage_scale=max(1.0,float(np.max(raw_sky[...,:3]))/60000,
                                   float(np.max(sky_pixels[...,:3]))/60000)
        self._write_texture(self.sky,(sky_pixels/self.sky_storage_scale).astype(np.float16))
        encoder = device.create_command_encoder()
        self._mipmaps(encoder,self._make_mip_groups(self.sky))
        device.queue.submit([encoder.finish()])
        self.reflection_texture = prefilter_environment(device,self.sky)
        # Dominant upper-hemisphere light and broad light color, in linear HDR.
        rgb = raw_sky[...,:3]
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
        self.active_material = "physical"
        self.legacy_sky = None
        self.presentation_sky = self.sky

    def _select_material(self,material):
        if material not in ("physical","2015"):
            raise ValueError("Material must be physical or 2015")
        if material == "2015" and self.legacy_pipelines is None:
            shader = self.device.create_shader_module(code=(Path(__file__).parent/"shaders/ocean_2015.wgsl").read_text())
            sky = self.device.create_render_pipeline(layout=self.pipeline_layout,
                vertex={"module":shader,"entry_point":"sky_vertex"},
                fragment={"module":shader,"entry_point":"sky_fragment","targets":[{"format":self.format}]},
                primitive={"topology":"triangle-list"},
                depth_stencil={"format":"depth32float","depth_write_enabled":False,"depth_compare":"always"})
            ocean = self.device.create_render_pipeline(layout=self.pipeline_layout,
                vertex={"module":shader,"entry_point":"ocean_vertex"},
                fragment={"module":shader,"entry_point":"ocean_fragment","targets":[{"format":self.format}]},
                primitive={"topology":"triangle-list","cull_mode":"none"},
                depth_stencil={"format":"depth32float","depth_write_enabled":True,"depth_compare":"less"})
            self.legacy_pipelines = sky,ocean
            self.legacy_sky = self._make_texture(self.raw_sky.shape[1],self.raw_sky.shape[0])
            self._write_texture(self.legacy_sky,(self.raw_sky/self.sky_storage_scale).astype(np.float16))
            encoder = self.device.create_command_encoder()
            self._mipmaps(encoder,self._make_mip_groups(self.legacy_sky))
            self.device.queue.submit([encoder.finish()])
        if material != self.active_material:
            self.active_material = material
            self.sky = self.presentation_sky if material=="physical" else self.legacy_sky
            self._bind_textures()
        return (self.sky_pipeline,self.ocean_pipeline) if material=="physical" else self.legacy_pipelines

    def _make_texture(self,width,height):
        return self.device.create_texture(size=(width,height,1),format="rgba16float",
            mip_level_count=int(math.log2(max(width,height)))+1,
            usage=wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST|wgpu.TextureUsage.RENDER_ATTACHMENT
                  |wgpu.TextureUsage.STORAGE_BINDING|wgpu.TextureUsage.COPY_SRC)

    @property
    def transfer_name(self):
        return self.gpu_transfer.name if self.gpu_transfer else "Host texture upload"

    def _select_transfer(self,compute_device):
        if self.transfer_device == compute_device: return
        self.gpu_transfer = None
        if self.transfer_mode != "host" and compute_device.type == "mps":
            try:
                from .metal_transfer import Metal_transfer
                self.gpu_transfer = Metal_transfer(self.device)
            except (RuntimeError,OSError,AttributeError) as error:
                if self.transfer_mode == "metal": raise
                warnings.warn(f"Metal GPU transfer unavailable; using host texture uploads: {error}",RuntimeWarning)
        elif self.transfer_mode == "metal":
            raise ValueError("Metal texture transfer requires Torch MPS compute")
        self.transfer_device = compute_device

    def _initialize_textures(self,textures):
        # Inform wgpu that level zero has been initialized before external Metal
        # writes. Otherwise WebGPU's first-use safety clear would erase them.
        encoder = self.device.create_command_encoder()
        for texture in textures:
            render_pass = encoder.begin_render_pass(color_attachments=[{
                "view":texture.create_view(base_mip_level=0,mip_level_count=1),
                "load_op":"clear","store_op":"store","clear_value":(0,0,0,0)}])
            render_pass.end()
        self.device.queue.submit([encoder.finish()])

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
        self._select_transfer(frame.displacement.device)
        n = frame.parameters.resolution
        if n != self.wave_size:
            for texture in self.wave_textures:
                texture.destroy()
            self.wave_size = n
            self.wave_textures = [self._make_texture(n,n) for _ in range(2)]
            if self.gpu_transfer: self._initialize_textures(self.wave_textures)
            self.wave_mips = [self._make_mip_groups(texture) for texture in self.wave_textures]
            self._bind_textures()
        if self.gpu_transfer:
            self.gpu_transfer.upload_waves(frame,self.wave_textures)
        else:
            maps = texture_arrays(frame)
            for texture,data in zip(self.wave_textures,maps):
                self._write_texture(texture,data)
        parameters = frame.parameters.in_ocean_space()
        if self.statistics_parameters != parameters:
            # Original viewer takes these statistics once after initialization.
            if self.gpu_transfer:
                # Match the display's half quantization, particularly for flat
                # crests. Read only three reduced scalars on parameter changes.
                display = frame.displacement[...,2:].to(torch.float16).float()
                variance,mean = torch.var_mean(display[...,1],correction=0)
                maximum,std,mean = torch.stack((display[...,0].abs().amax(),variance.sqrt(),mean)).cpu().tolist()
            else:
                height=maps[0][...,2].astype(np.float32)
                crest=maps[0][...,3].astype(np.float32)
                maximum,std,mean = float(np.max(np.abs(height))),float(np.std(crest)),float(np.mean(crest))
            self.big_height=max(.001,1.5*maximum)
            self.crest_gain=1/max(1e-8,2*std)
            self.crest_bias=-mean*self.crest_gain
            self.statistics_parameters=parameters
        self.frame = frame
        self.mips_dirty = True

    def _bind_textures(self):
        if not self.wave_textures: return
        self.bind_group = self.device.create_bind_group(layout=self.layout,entries=[
            {"binding":0,"resource":{"buffer":self.uniform}},
            {"binding":1,"resource":self.legacy_wave_sampler if self.active_material=="2015" else self.wave_sampler},
            {"binding":2,"resource":self.wave_textures[0].create_view()},
            {"binding":3,"resource":self.wave_textures[1].create_view()},
            {"binding":4,"resource":self.sky_sampler},
            {"binding":5,"resource":self.sky.create_view()},
            {"binding":6,"resource":self.foam_texture.create_view()},
            {"binding":7,"resource":self.reflection_texture.create_view()},
            {"binding":8,"resource":(self.foam_material_texture or self.foam_texture).create_view()},
            {"binding":9,"resource":self.foam_detail.view},
        ])

    def upload_foam(self,state):
        if state is self.foam_state: return
        previous = self.foam_state
        self.foam_state = state
        if state is None: return
        if previous is not None and previous.density is state.density: return
        n = state.parameters.resolution
        if self.foam_texture.size[0] != n:
            self.foam_texture.destroy()
            self.foam_texture = self._make_texture(n,n)
            if self.gpu_transfer: self._initialize_textures([self.foam_texture])
            self.foam_mips = self._make_mip_groups(self.foam_texture)
            self._bind_textures()
        if self.gpu_transfer:
            self.gpu_transfer.upload_foam(state,self.foam_texture)
            self.foam_dirty = True
            self.foam_material_dirty = True
            return
        rgb = state.density.permute(1,2,0).cpu().numpy()
        data = np.zeros((*rgb.shape[:2],4),np.float16)
        data[...,:3] = np.minimum(rgb,60000)
        if self.foam_grain_basis is not state.basis:
            # Stable fine bubble breakup, carried in the otherwise unused alpha
            # channel. Density history remains exactly the three RGB fields.
            grain=state.basis.noise_cos[-1].cpu().numpy()
            self.foam_grain=np.clip(.5+grain*(.5/(.5**3/1.875)),0,1).astype(np.float16)
            self.foam_grain_basis=state.basis
        data[...,3] = self.foam_grain
        self._write_texture(self.foam_texture,data)
        self.foam_dirty = True
        self.foam_material_dirty = True

    @property
    def shading_statistics(self):
        return Shading_statistics(self.big_height,self.crest_gain,self.crest_bias)

    def restore_shading_statistics(self,statistics):
        self.big_height=statistics.big_height
        self.crest_gain=statistics.crest_gain
        self.crest_bias=statistics.crest_bias

    def _render_targets(self,width,height,sample_count,physical):
        if (self.depth is None or self.depth.size[:2] != (width,height)
                or self.depth_sample_count != sample_count):
            if self.depth: self.depth.destroy()
            self.depth = self.device.create_texture(size=(width,height,1),format="depth32float",
                sample_count=sample_count,usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.depth_sample_count = sample_count
        if physical and (self.hdr is None or self.hdr.size[:2] != (width,height)):
            if self.hdr: self.hdr.destroy()
            if self.multisample: self.multisample.destroy()
            self.hdr = self.device.create_texture(size=(width,height,1),format="rgba16float",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_SRC)
            self.multisample = self.device.create_texture(size=(width,height,1),format="rgba16float",
                sample_count=self.sample_count,usage=wgpu.TextureUsage.RENDER_ATTACHMENT) if self.sample_count>1 else None
            self.presentation_group = self.device.create_bind_group(
                layout=self.presentation_pipeline.get_bind_group_layout(0),entries=[
                    {"binding":0,"resource":self.hdr.create_view()},
                    {"binding":1,"resource":self.presentation_sampler},
                    {"binding":2,"resource":{"buffer":self.uniform}}])

    def draw(self,target_view,width,height,camera=Camera(),look=Look(),viewport=None,clear=True):
        if not self._physical_ready and look.material == "physical":
            # On Metal the first filtered physical draw can differ from later
            # draws with bit-identical inputs. Complete one priming draw before
            # presenting it; this fence reads no wave or image data to the CPU.
            from .metal_transfer import wait_for_graphics
            self._draw(target_view,width,height,camera,look,viewport,clear)
            wait_for_graphics(self.device)
            self._physical_ready = True
        self._draw(target_view,width,height,camera,look,viewport,clear)

    def _draw(self,target_view,width,height,camera,look,viewport,clear):
        if self.frame is None:
            raise RuntimeError("Upload a wave frame before drawing")
        sky_pipeline,ocean_pipeline = self._select_material(look.material)
        physical = look.material == "physical"
        # Each side of a comparison resolves its own linear HDR image. The final
        # display pass composites only that viewport, preserving the other side.
        render_width,render_height = tuple(map(int,viewport[2:])) if physical and viewport else (width,height)
        self._render_targets(render_width,render_height,self.sample_count if physical else 1,physical)
        forward,right,up = camera.basis()
        aspect = (viewport[2]/viewport[3]) if viewport else width/height
        rotation = math.radians(look.sky_rotation)
        ocean_rotation = math.radians(self.frame.parameters.wind_direction % 360)
        sx,sy,sz = self.sun_direction if look.material=="2015" else self.lighting.dominant_direction
        sun = (sx*math.cos(rotation)+sy*math.sin(rotation),sy*math.cos(rotation)-sx*math.sin(rotation),sz)
        # Keep all intermediate radiance inside half-float range. The display
        # pass reverses this storage normalization before its highlight shoulder,
        # so even a very bright HDR sun remains recoverable at low exposure.
        hdr_scale = max(1.0,self.sky_peak*look.sky_gain/32000) if physical else 1.0
        lighting_gain = look.sky_gain/hdr_scale
        uniform = np.array([
            [camera.x,camera.y,camera.height,self.frame.time],
            [*forward,math.tan(math.radians(camera.fov)/2)],
            [*right,aspect],[*up,look.exposure+math.log2(hdr_scale)],
            [self.frame.parameters.domain,self.wave_size,look.crest_threshold,look.foam],
            [rotation,look.haze,lighting_gain*self.sky_storage_scale,ocean_rotation],[*sun,0],
            [*((self.sun_color if look.material=="2015" else self.lighting.direct_irradiance)*lighting_gain),0],
            [*self.mesh_resolution,.5,30000],
            [self.big_height,self.crest_gain,self.crest_bias,look.crest_maximum],
            [*(self.moon_color*look.sky_gain),0],
            [float(self.foam_state is not None),look.aeration,0,0],
            [.025,float(self.reflection_texture.mip_level_count-1),20000,0],
            *[[*(coefficient*lighting_gain),0] for coefficient in self.lighting.diffuse_sh],
            *[[*(coefficient*lighting_gain),0] for coefficient in self.lighting.ambient_sh],
        ],np.float32)
        self.device.queue.write_buffer(self.uniform,0,uniform)
        encoder = self.device.create_command_encoder()
        if self.mips_dirty:
            for groups in self.wave_mips:
                self._mipmaps(encoder,groups)
            self.mips_dirty = False
        if self.foam_dirty:
            self._mipmaps(encoder,self.foam_mips)
        if physical and (self.foam_material_dirty or self.foam_material_strength != look.foam):
            material_texture = self.foam_material.update(encoder,self.foam_texture,look.foam)
            if material_texture is not self.foam_material_texture:
                self.foam_material_texture = material_texture
                self._bind_textures()
            self.foam_material_strength = look.foam
            self.foam_material_dirty = False
        if self.foam_dirty:
            self.foam_dirty = False
        attachment = {"view":target_view,"clear_value":(0,0,0,1),
                      "load_op":"clear" if clear else "load","store_op":"store"}
        if physical:
            attachment.update(view=(self.multisample or self.hdr).create_view(),load_op="clear")
            if self.multisample:
                attachment.update(resolve_target=self.hdr.create_view(),store_op="discard")
        render_pass = encoder.begin_render_pass(color_attachments=[attachment],
            depth_stencil_attachment={"view":self.depth.create_view(),"depth_clear_value":1.0,
                                      "depth_load_op":"clear","depth_store_op":"store"})
        if viewport and not physical:
            render_pass.set_viewport(*viewport,0,1)
            render_pass.set_scissor_rect(*(int(v) for v in viewport))
        render_pass.set_bind_group(0,self.bind_group)
        render_pass.set_pipeline(sky_pipeline)
        render_pass.draw(3)
        render_pass.set_pipeline(ocean_pipeline)
        render_pass.set_index_buffer(self.indices,"uint32")
        render_pass.draw_indexed(self.index_count)
        render_pass.end()
        if physical:
            render_pass = encoder.begin_render_pass(color_attachments=[{
                "view":target_view,"clear_value":(0,0,0,1),
                "load_op":"clear" if clear else "load","store_op":"store"}])
            if viewport:
                render_pass.set_viewport(*viewport,0,1)
                render_pass.set_scissor_rect(*(int(v) for v in viewport))
            render_pass.set_pipeline(self.presentation_pipeline)
            render_pass.set_bind_group(0,self.presentation_group)
            render_pass.draw(3)
            render_pass.end()
        if self.gpu_transfer: self.gpu_transfer.before_graphics()
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
