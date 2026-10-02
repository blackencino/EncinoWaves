# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Deterministic, offscreen still / sequence / movie export."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from functools import lru_cache
import json
import math
import subprocess
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import wgpu
from .model import Wave_parameters, Phase_step, make_initial_state, evaluate
from .editing import make_edited_state, make_wave_basis, state_from_basis, edit_state
from .render import Ocean_renderer, make_device, Camera, Look, Shading_statistics, read_rgba
from .camera import frame_domain
from .foam import Foam_parameters, prepare_foam, advance_foam_to
from .presentation import sample_presentation, changed_control


@dataclass(frozen=True)
class Shot:
    duration: float
    title: str
    caption: str
    parameters: Wave_parameters
    comparison: bool = False
    camera: Camera | None = None
    look: Look = Look()
    start_time: float = 10.0
    shading_statistics: Shading_statistics | None = None
    comparison_shading_statistics: Shading_statistics | None = None
    post_seed: bool = False
    phase_steps: tuple[Phase_step, ...] = ()
    foam: Foam_parameters = Foam_parameters()
    foam_state: str | None = None
    comparison_foam_state: str | None = None
    foam_preroll: float = 6.0


@lru_cache(maxsize=12)
def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Avenir Next.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).exists(): return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def _physical_caption(parameters):
    def number(value,places):
        value=round(value,places)
        if value==0: return "0"
        return f"{value:.{places}f}".rstrip("0").rstrip(".") if places else f"{value:.0f}"
    return (f"Wind {number(parameters.wind_speed,1)} m/s   Fetch {number(parameters.fetch_km,0)} km   "
            f"Depth {number(parameters.depth,1 if parameters.depth<10 else 0)} m   Swell {number(parameters.swell,2)}")


def caption_image(rgba,title,caption,parameters,comparison=False,*,opacity=1.0,rounded_controls=False):
    image=Image.fromarray(rgba).convert("RGBA")
    w,h=image.size
    scale=h/1080
    overlay=Image.new("RGBA",image.size)
    draw=ImageDraw.Draw(overlay)
    opacity=max(0.,min(1.,float(opacity)))
    comparison=max(0.,min(1.,float(comparison)))
    # A small production-style lower third; never covers the central wave field.
    for y in range(int(h*.73),h):
        alpha=int(185*opacity*max(0.0,(y-h*.73)/(h*.27))**1.2)
        draw.line((0,y,w,y),fill=(5,12,16,alpha))
    margin=int(55*scale)
    draw.text((margin,h-int(142*scale)),title,font=_font(max(1,int(40*scale))),fill=(240,242,237,int(255*opacity)))
    draw.text((margin,h-int(85*scale)),caption,font=_font(max(1,int(23*scale))),fill=(210,223,222,int(255*opacity)))
    settings=(_physical_caption(parameters) if rounded_controls else
              f"Wind {parameters.wind_speed:g} m/s   Fetch {parameters.fetch_km:g} km   Depth {parameters.depth:g} m   Swell {parameters.swell:g}")
    draw.text((margin,margin),settings,font=_font(max(1,int(20*scale))),fill=(231,235,233,255),stroke_width=1,stroke_fill=(0,0,0,110))
    if comparison:
        draw.line((w//2,0,w//2,h),fill=(205,218,218,int(140*comparison)),width=max(1,int(2*scale)))
        for x,label in ((w*.08,"Earlier model"),(w*.58,"Encino Waves")):
            draw.text((int(x),int(105*scale)),label,font=_font(max(1,int(28*scale))),fill=(235,239,236,int(255*comparison)),stroke_width=1,stroke_fill=(0,0,0,int(160*comparison)))
    return np.asarray(Image.alpha_composite(image,overlay).convert("RGB"))


class Movie_writer:
    def __init__(self,path,width,height,fps,codec="h264",overwrite=False,max_mbps=None):
        path=Path(path)
        if path.exists() and not overwrite: raise FileExistsError(f"Output exists: {path}; use --overwrite explicitly")
        path.parent.mkdir(parents=True,exist_ok=True)
        self.path=path
        self.partial=path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial{path.suffix}")
        command=[imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y" if overwrite else "-n",
                 "-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","-","-an"]
        if codec=="prores":
            command += ["-c:v","prores_ks","-profile:v","3","-pix_fmt","yuv422p10le"]
        else:
            command += ["-c:v","libx264","-preset","medium","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart"]
            if max_mbps is not None:
                command += ["-maxrate",f"{max_mbps:g}M","-bufsize",f"{2*max_mbps:g}M"]
        command += [str(self.partial)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE)

    def write(self,rgb):
        try: self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        except BrokenPipeError as error: raise RuntimeError("Video encoder failed") from error

    def close(self,commit=True):
        try: self.process.stdin.close()
        except BrokenPipeError: pass
        code=self.process.wait()
        if code or not commit:
            self.partial.unlink(missing_ok=True)
            if commit: raise RuntimeError("Video encoder failed; inspect its error output")
        else:
            self.partial.replace(self.path)


def render_shots(shots,output,*,sky=None,device="auto",width=1920,height=1080,fps=24,
                 captions=True,codec="h264",overwrite=False,max_mbps=None,progress=print):
    if width%2 or height%2: raise ValueError("Movie dimensions must be even")
    graphics=make_device()
    # Retain the same geometry density in screen space for a UHD master.
    mesh_scale=max(1.0,width/1920,height/1080)
    mesh_resolution=(round(960*mesh_scale),round(576*mesh_scale))
    renderer=Ocean_renderer(graphics,sky,mesh_resolution=mesh_resolution)
    comparison_renderer=None
    target=graphics.create_texture(size=(width,height,1),format=renderer.format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
    writer=Movie_writer(output,width,height,fps,codec,overwrite,max_mbps)
    manifest={"fps":fps,"width":width,"height":height,"sky":str(sky) if sky else renderer.sky_name,
              "graphics":dict(graphics.adapter.info),"shots":[],"frames":0,"max_mbps":max_mbps,
              "mesh_resolution":mesh_resolution,"antialiasing_samples":renderer.sample_count,"codec":codec}
    start=time.perf_counter()
    complete=False
    try:
        for index,shot in enumerate(shots):
            progress(f"Shot {index+1}/{len(shots)}: {shot.title}",flush=True)
            if shot.post_seed or shot.phase_steps:
                state=make_edited_state(shot.parameters,device,shot.phase_steps)
                other=make_edited_state(shot.parameters.tessendorf(),device,shot.phase_steps) if shot.comparison else None
            else:
                state=make_initial_state(shot.parameters,device)
                other=make_initial_state(shot.parameters.tessendorf(),device) if shot.comparison else None
            if other and comparison_renderer is None:
                comparison_renderer=Ocean_renderer(graphics,sky,mesh_resolution=mesh_resolution)
            camera=shot.camera or frame_domain(shot.parameters.domain)
            look=shot.look
            foam=prepare_foam(state,shot.start_time,shot.foam,shot.foam_state,shot.foam_preroll)
            comparison_foam=prepare_foam(other,shot.start_time,shot.foam,shot.comparison_foam_state,shot.foam_preroll) if other else None
            frame_count=round(shot.duration*fps)
            manifest["shots"].append({**asdict(shot),"camera":asdict(camera),"look":asdict(look),"frames":frame_count})
            for i in range(frame_count):
                # Offline simulation time depends only on frame index.
                t=shot.start_time+i/fps
                frame=evaluate(state,t)
                renderer.upload(frame)
                foam=advance_foam_to(foam,state,frame,shot.foam)
                renderer.upload_foam(foam)
                if i==0 and shot.shading_statistics:
                    renderer.restore_shading_statistics(shot.shading_statistics)
                if other:
                    other_frame=evaluate(other,t)
                    comparison_renderer.upload(other_frame)
                    comparison_foam=advance_foam_to(comparison_foam,other,other_frame,shot.foam)
                    comparison_renderer.upload_foam(comparison_foam)
                    if i==0 and shot.comparison_shading_statistics:
                        comparison_renderer.restore_shading_statistics(shot.comparison_shading_statistics)
                    half=width//2
                    comparison_renderer.draw(target.create_view(),width,height,camera,look,viewport=(0,0,half,height))
                    renderer.draw(target.create_view(),width,height,camera,look,viewport=(half,0,width-half,height),clear=False)
                else:
                    renderer.draw(target.create_view(),width,height,camera,look)
                rgba=read_rgba(graphics,target,width,height)
                rgb=caption_image(rgba,shot.title,shot.caption,shot.parameters,shot.comparison) if captions else rgba[...,:3]
                writer.write(rgb)
                manifest["frames"]+=1
                if i and i%(fps*5)==0:
                    progress(f"  {i/fps:.0f}/{shot.duration:g} s rendered",flush=True)
            del state,other
        complete=True
    finally:
        try: writer.close(commit=complete)
        finally: target.destroy()
    manifest["render_seconds"]=time.perf_counter()-start
    manifest["duration_seconds"]=manifest["frames"]/fps
    Path(str(output)+".json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


class _Presentation_ocean:
    """Advance one authored ocean without reseeding, amplitude blends or resets."""
    def __init__(self,presentation,device):
        self.presentation=presentation
        self.basis=make_wave_basis(presentation.initial.parameters,device)
        self.state=state_from_basis(self.basis,presentation.initial.parameters)
        self.foam=prepare_foam(self.state,presentation.start_time,presentation.foam,
                               preroll=presentation.foam_preroll)
        self.comparison_state=None
        self.comparison_foam=None
        self.elapsed=0.

    def advance(self,sample,elapsed):
        if not math.isfinite(elapsed) or elapsed<self.elapsed:
            raise ValueError("Presentation frames must advance forward")
        self.elapsed=elapsed
        t=self.presentation.start_time+elapsed
        parameters=sample.state.parameters
        if parameters!=self.state.parameters:
            self.state=edit_state(self.basis,self.state,parameters,t)
        frame=evaluate(self.state,t)
        self.foam=advance_foam_to(self.foam,self.state,frame,self.presentation.foam)
        comparison_frame=None
        if sample.state.comparison>0:
            other_parameters=parameters.tessendorf()
            first=self.comparison_state is None
            if first or self.comparison_state.parameters!=other_parameters:
                self.comparison_state=state_from_basis(self.basis,other_parameters)
            # The comparison changes directional/spectral amplitudes only.
            # Its dispersion and travelling-wave phase exactly match the main sea.
            self.comparison_state=replace(self.comparison_state,phase=self.state.phase,
                                           phase_steps=self.state.phase_steps)
            if first:
                self.comparison_foam=prepare_foam(self.comparison_state,t,self.presentation.foam,
                                                  preroll=self.presentation.foam_preroll)
            comparison_frame=evaluate(self.comparison_state,t)
            self.comparison_foam=advance_foam_to(self.comparison_foam,self.comparison_state,
                                                comparison_frame,self.presentation.foam)
        return frame,self.foam,comparison_frame,self.comparison_foam


class _Comparison_overlay:
    """Fade an independently rendered full-projection comparison onto the left."""
    def __init__(self,device,main,comparison,format):
        self.device=device
        self.target=device.create_texture(size=main.size,format=format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
        self.target_view=self.target.create_view()
        self.uniform=device.create_buffer(size=16,usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)
        shader=device.create_shader_module(code="""
            @group(0) @binding(0) var current: texture_2d<f32>;
            @group(0) @binding(1) var earlier: texture_2d<f32>;
            @group(0) @binding(2) var<uniform> amount: vec4f;
            @vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
                let p=vec2f(f32((i<<1u)&2u),f32(i&2u));
                return vec4f(p*2.0-1.0,0.0,1.0);
            }
            @fragment fn fragment(@builtin(position) p:vec4f)->@location(0) vec4f {
                let pixel=vec2i(p.xy);
                let base=textureLoad(current,pixel,0);
                if p.x < f32(textureDimensions(current).x)*.5 {
                    return mix(base,textureLoad(earlier,pixel,0),amount.x);
                }
                return base;
            }
        """)
        self.pipeline=device.create_render_pipeline(layout="auto",
            vertex={"module":shader,"entry_point":"vertex"},
            fragment={"module":shader,"entry_point":"fragment","targets":[{"format":format}]})
        self.group=device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0),entries=[
            {"binding":0,"resource":main.create_view()},
            {"binding":1,"resource":comparison.create_view()},
            {"binding":2,"resource":{"buffer":self.uniform}}])

    def compose(self,opacity):
        self.device.queue.write_buffer(self.uniform,0,np.array([opacity,0,0,0],np.float32))
        encoder=self.device.create_command_encoder()
        render_pass=encoder.begin_render_pass(color_attachments=[{
            "view":self.target_view,"load_op":"clear","store_op":"store","clear_value":(0,0,0,1)}])
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0,self.group)
        render_pass.draw(3)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        return self.target

    def destroy(self):
        self.target.destroy()
        self.uniform.destroy()


def render_presentation(presentation,output,*,sky=None,device="auto",width=1920,height=1080,fps=24,
                        captions=True,codec="h264",overwrite=False,max_mbps=None,progress=print):
    """Render the continuous timeline; only displayed comparison pixels dissolve.

    Frame times are absolute and derived from the output frame index. A single
    post-seed basis, accumulated wave phase and foam history span all cues.
    ``render_shots`` remains the independent saved-scene/shot export path.
    """
    if width<=0 or height<=0 or width%2 or height%2:
        raise ValueError("Movie dimensions must be positive and even")
    if not isinstance(fps,int) or not 1<=fps<=120:
        raise ValueError("fps must be an integer between 1 and 120")
    frame_count=round(presentation.duration*fps)
    if frame_count<1: raise ValueError("Presentation is shorter than one output frame")
    graphics=make_device()
    mesh_scale=max(1.0,width/1920,height/1080)
    mesh_resolution=(round(960*mesh_scale),round(576*mesh_scale))
    renderer=Ocean_renderer(graphics,sky,mesh_resolution=mesh_resolution)
    def target():
        return graphics.create_texture(size=(width,height,1),format=renderer.format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC|wgpu.TextureUsage.TEXTURE_BINDING)
    main_target=target()
    main_view=main_target.create_view()
    comparison_renderer=comparison_target=comparison_view=overlay=None
    writer=Movie_writer(output,width,height,fps,codec,overwrite,max_mbps)
    cues=[]
    elapsed=0.
    previous=presentation.initial
    for cue in presentation.cues:
        cues.append({**asdict(cue),"start_seconds":elapsed,"end_seconds":elapsed+cue.duration,
                     "control":changed_control(previous,cue.target)})
        elapsed+=cue.duration
        previous=cue.target
    manifest={"type":"continuous_presentation","fps":fps,"width":width,"height":height,
        "sky":str(sky) if sky else renderer.sky_name,"graphics":dict(graphics.adapter.info),
        "frames":0,"max_mbps":max_mbps,"mesh_resolution":mesh_resolution,
        "antialiasing_samples":renderer.sample_count,"codec":codec,"captions":captions,
        "wave_resolution":presentation.initial.parameters.resolution,"foam_resolution":presentation.foam.resolution,
        "initial":asdict(presentation.initial),"look":asdict(presentation.look),"foam":asdict(presentation.foam),
        "start_time":presentation.start_time,"foam_preroll":presentation.foam_preroll,
        "timeline_seconds":presentation.duration,"cues":cues,
        "comparison":{"parameters":asdict(presentation.initial.parameters.tessendorf()),
                      "layout":"left-half opacity overlay","projection":"matched full frame"}}
    start=time.perf_counter()
    complete=False
    try:
        ocean=_Presentation_ocean(presentation,device)
        previous_cue=None
        for index in range(frame_count):
            elapsed=index/fps
            sample=sample_presentation(presentation,elapsed)
            cue=presentation.cues[sample.cue_index]
            if sample.cue_index!=previous_cue:
                progress(f"Cue {sample.cue_index+1}/{len(presentation.cues)} at {elapsed:.2f} s: {cue.title}",flush=True)
                previous_cue=sample.cue_index
            frame,foam,other_frame,other_foam=ocean.advance(sample,elapsed)
            renderer.upload(frame)
            renderer.upload_foam(foam)
            renderer.draw(main_view,width,height,sample.state.camera,presentation.look)
            output_target=main_target
            if other_frame is not None:
                if comparison_renderer is None:
                    comparison_renderer=Ocean_renderer(graphics,sky,mesh_resolution=mesh_resolution)
                    comparison_target=target()
                    comparison_view=comparison_target.create_view()
                    overlay=_Comparison_overlay(graphics,main_target,comparison_target,renderer.format)
                comparison_renderer.upload(other_frame)
                comparison_renderer.upload_foam(other_foam)
                # Both passes have the same full-size projection. The overlay
                # masks pixels only after rendering, so the camera never jumps.
                comparison_renderer.draw(comparison_view,width,height,sample.state.camera,presentation.look)
                output_target=overlay.compose(sample.state.comparison)
            rgba=read_rgba(graphics,output_target,width,height)
            rgb=(caption_image(rgba,cue.title,cue.caption,sample.state.parameters,sample.state.comparison,
                               opacity=sample.caption_opacity,rounded_controls=True) if captions else rgba[...,:3])
            writer.write(rgb)
            manifest["frames"]+=1
            if index and index%(fps*5)==0:
                progress(f"  {elapsed:.0f}/{presentation.duration:g} s rendered",flush=True)
        manifest["final_phase_steps"]=[asdict(step) for step in ocean.state.phase_steps]
        complete=True
    finally:
        try: writer.close(commit=complete)
        finally:
            main_target.destroy()
            if comparison_target is not None: comparison_target.destroy()
            if overlay is not None: overlay.destroy()
    manifest["render_seconds"]=time.perf_counter()-start
    manifest["duration_seconds"]=manifest["frames"]/fps
    Path(str(output)+".json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest
