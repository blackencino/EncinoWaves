# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Deterministic, offscreen still / sequence / movie export."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from functools import lru_cache
import json
import subprocess
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import wgpu
from .model import Wave_parameters, Phase_step, make_initial_state, evaluate
from .editing import make_edited_state
from .render import Ocean_renderer, make_device, Camera, Look, Shading_statistics, read_rgba
from .camera import frame_domain


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


def academy_shots(resolution=2048):
    p=Wave_parameters(resolution=resolution)
    # 300 seconds exactly; each cut is an actual fixed physical state.
    # Identical geometry, shading, seed and scale within each comparison.
    return (
        Shot(20,"Encino Waves","Christopher J. Horvath | Empirical directional wave spectra",p),
        Shot(25,"The starting point","Earlier model / Encino Waves",p,True),
        Shot(15,"Wind speed","5 m/s",replace(p,wind_speed=5,fetch_km=50)),
        Shot(15,"Wind speed","17 m/s",p),
        Shot(15,"Wind speed","30 m/s",replace(p,wind_speed=30)),
        Shot(15,"Fetch","20 km | A short distance for the wind to build waves",replace(p,fetch_km=20)),
        Shot(15,"Fetch","300 km",p),
        Shot(15,"Fetch","1,250 km | More distance for waves to develop",replace(p,fetch_km=1250)),
        Shot(20,"Directional spreading","How wave directions vary with wavelength",replace(p,spreading="hasselmann"),True),
        Shot(20,"Local wind","Swell 0 | An irregular sea",replace(p,swell=0)),
        Shot(20,"Distant weather","Swell 1 | Long, parallel wave trains",replace(p,swell=1)),
        Shot(15,"Ocean depth","150 metres",replace(p,depth=150)),
        Shot(15,"Ocean depth","5 metres | Shallow water changes the mix of waves",replace(p,depth=5)),
        Shot(20,"A range of conditions","Light wind | 3 m/s, 8 km fetch",replace(p,domain=160,wind_speed=3,fetch_km=8,depth=30)),
        Shot(20,"A range of conditions","Storm | 35 m/s, 1,250 km fetch",replace(p,domain=1800,wind_speed=35,fetch_km=1250,depth=150),camera=frame_domain(512)),
        Shot(15,"Physical controls","Wind speed. Fetch. Directional spreading. Swell. Depth.",p),
        Shot(20,"Encino Waves","Empirical directional wave spectra for computer graphics | 2015",replace(p,swell=.6)),
    )


@lru_cache(maxsize=12)
def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Avenir Next.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).exists(): return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def caption_image(rgba,title,caption,parameters,comparison=False):
    image=Image.fromarray(rgba).convert("RGBA")
    w,h=image.size
    scale=h/1080
    overlay=Image.new("RGBA",image.size)
    draw=ImageDraw.Draw(overlay)
    # A small production-style lower third; never covers the central wave field.
    for y in range(int(h*.73),h):
        alpha=int(185*max(0.0,(y-h*.73)/(h*.27))**1.2)
        draw.line((0,y,w,y),fill=(5,12,16,alpha))
    margin=int(55*scale)
    draw.text((margin,h-int(142*scale)),title,font=_font(int(40*scale)),fill=(240,242,237,255))
    draw.text((margin,h-int(85*scale)),caption,font=_font(int(23*scale)),fill=(210,223,222,255))
    settings=f"Wind {parameters.wind_speed:g} m/s   Fetch {parameters.fetch_km:g} km   Depth {parameters.depth:g} m   Swell {parameters.swell:g}"
    draw.text((margin,margin),settings,font=_font(int(20*scale)),fill=(231,235,233,255),stroke_width=1,stroke_fill=(0,0,0,110))
    if comparison:
        draw.line((w//2,0,w//2,h),fill=(205,218,218,140),width=max(1,int(2*scale)))
        for x,label in ((w*.08,"Earlier model"),(w*.58,"Encino Waves")):
            draw.text((int(x),int(105*scale)),label,font=_font(int(28*scale)),fill=(235,239,236,255),stroke_width=1,stroke_fill=(0,0,0,160))
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
    renderer=Ocean_renderer(graphics,sky,mesh_resolution=(960,576))
    comparison_renderer=None
    target=graphics.create_texture(size=(width,height,1),format=renderer.format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
    writer=Movie_writer(output,width,height,fps,codec,overwrite,max_mbps)
    manifest={"fps":fps,"width":width,"height":height,"sky":str(sky) if sky else renderer.sky_name,
              "graphics":dict(graphics.adapter.info),"shots":[],"frames":0,"max_mbps":max_mbps}
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
                comparison_renderer=Ocean_renderer(graphics,sky,mesh_resolution=(960,576))
            camera=shot.camera or frame_domain(shot.parameters.domain)
            look=shot.look
            frame_count=round(shot.duration*fps)
            manifest["shots"].append({**asdict(shot),"camera":asdict(camera),"look":asdict(look),"frames":frame_count})
            for i in range(frame_count):
                # Offline simulation time depends only on frame index.
                t=shot.start_time+i/fps
                renderer.upload(evaluate(state,t))
                if i==0 and shot.shading_statistics:
                    renderer.restore_shading_statistics(shot.shading_statistics)
                if other:
                    comparison_renderer.upload(evaluate(other,t))
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
