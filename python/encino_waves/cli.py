# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
import argparse
from dataclasses import replace
from pathlib import Path
import json
import math
import time
import numpy as np
from .model import Wave_parameters, Phase_step, make_initial_state, evaluate, texture_arrays, synchronize, select_device


def parser():
    root=argparse.ArgumentParser(prog="encino-waves",description="Encino Waves: GPU ocean synthesis and viewer")
    sub=root.add_subparsers(dest="command")
    view=sub.add_parser("view",help="Open the native viewer")
    still=sub.add_parser("still",help="Render a PNG offscreen")
    render=sub.add_parser("render",help="Render a deterministic movie")
    demo=sub.add_parser("demo",help="Render the five-minute Academy sequence")
    bench=sub.add_parser("benchmark",help="Measure synchronized GPU compute and transfer separately")
    doctor=sub.add_parser("doctor",help="Report compute and graphics device support")
    for p in (view,still,render,demo,bench):
        p.add_argument("--resolution",type=int,default=None)
        p.add_argument("--device",default="auto",help="auto, mps, cuda[:index], or cpu")
    for p in (view,still,render,demo):
        p.add_argument("--sky",type=Path,help="Local Dutch Skies .hdr or .exr panorama")
    for p in (view,still,render):
        p.add_argument("--preset",type=int,choices=range(1,7),default=3)
    for p in (still,render,demo):
        p.add_argument("output",type=Path)
        p.add_argument("--width",type=int,default=1920)
        p.add_argument("--height",type=int,default=1080)
        p.add_argument("--overwrite",action="store_true")
        p.add_argument("--foam",action=argparse.BooleanOptionalAction,default=None,help="Persistent RGB foam; --no-foam restores crest shading")
        p.add_argument("--foam-preroll",type=float,default=6.0,help="Seconds of foam buildup when no saved history is available (0-60)")
    for p in (view,still,render):
        p.add_argument("--scene",type=Path,help="JSON saved by the viewer")
    for p in (still,render):
        p.add_argument("--tessendorf",action="store_true")
        p.add_argument("--time",type=float,default=None,help="Start time in seconds; overrides saved scene time")
    render.add_argument("--seconds",type=float,default=10)
    render.add_argument("--compare",action="store_true")
    still.add_argument("--compare",action="store_true")
    for p in (render,demo):
        p.add_argument("--fps",type=int,default=24)
        p.add_argument("--codec",choices=("h264","prores"),default="h264")
        p.add_argument("--max-mbps",type=float,help="Optional H.264 bitrate ceiling for compact review copies")
        p.add_argument("--no-captions",action="store_true")
    demo.add_argument("--preview",action="store_true",help="Two seconds per shot, for review")
    bench.add_argument("--frames",type=int,default=30)
    bench.add_argument("--output",type=Path)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command is None:
        args=parser().parse_args(["view"])
    requested_resolution=getattr(args,"resolution",None)
    if args.command!="doctor": args.resolution=requested_resolution or 1024
    if args.command=="doctor":
        import torch,wgpu
        result={"torch":torch.__version__,"metal":torch.backends.mps.is_available(),"cuda":torch.cuda.is_available(),
                "graphics":[dict(a.info) for a in wgpu.gpu.enumerate_adapters_sync()]}
        if torch.cuda.is_available(): result["cuda_device"]=torch.cuda.get_device_name()
        print(json.dumps(result,indent=2)); return
    if args.command=="benchmark":
        if args.frames<1: raise ValueError("frames must be positive")
        start=time.perf_counter()
        state=make_initial_state(Wave_parameters(resolution=args.resolution),args.device)
        synchronize(state.device)
        setup=time.perf_counter()-start
        for i in range(3): evaluate(state,i/24)
        synchronize(state.device)
        times=[]
        for i in range(args.frames):
            start=time.perf_counter(); frame=evaluate(state,i/24); synchronize(state.device)
            times.append((time.perf_counter()-start)*1000)
        start=time.perf_counter(); maps=texture_arrays(frame); transfer=(time.perf_counter()-start)*1000
        result={"device":str(state.device),"resolution":args.resolution,"frames":args.frames,
                "setup_seconds":setup,"compute_median_ms":float(np.median(times)),"compute_p95_ms":float(np.percentile(times,95)),
                "texture_readback_ms":transfer,"texture_bytes":sum(m.nbytes for m in maps)}
        print(json.dumps(result,indent=2))
        if args.output: args.output.write_text(json.dumps(result,indent=2)+"\n")
        return
    if args.command=="view":
        from .viewer import Viewer
        Viewer(args.resolution,args.device,args.sky,args.preset-1,scene=args.scene).run(); return
    from .presets import scene_with_resolution
    from .render import Ocean_renderer, make_device, Camera, Look, Shading_statistics
    from .export import Shot,render_shots,academy_shots
    from .foam import Foam_parameters, prepare_foam
    if args.width<16 or args.height<16: raise ValueError("Image dimensions must be at least 16")
    if args.command=="demo":
        shots=academy_shots(args.resolution)
        shots=tuple(replace(shot,foam=replace(shot.foam,enabled=args.foam if args.foam is not None else True),foam_preroll=args.foam_preroll) for shot in shots)
        if args.preview: shots=tuple(replace(shot,duration=2) for shot in shots)
    else:
        scene=scene_with_resolution(args.preset-1,args.resolution)
        p,camera,look=scene.parameters,scene.camera,scene.look
        start_time=10.0
        shading=None
        comparison_shading=None
        post_seed=False
        phase_steps=()
        foam=Foam_parameters()
        foam_state=comparison_foam_state=None
        if args.scene:
            saved=json.loads(args.scene.read_text())
            p=Wave_parameters.from_dict(saved["parameters"])
            if requested_resolution: p=replace(p,resolution=requested_resolution)
            camera=Camera(**saved["camera"]); look=Look.from_dict(saved["look"])
            if args.sky is None and saved.get("sky"): args.sky=Path(saved["sky"])
            start_time=saved.get("time",start_time)
            post_seed=saved.get("post_seed",False)
            phase_steps=tuple(Phase_step(**step) for step in saved.get("phase_steps",()))
            foam=Foam_parameters.from_dict(saved.get("foam",{"enabled":False}))
            if saved.get("foam_state"): foam_state=str(args.scene.parent/saved["foam_state"])
            if saved.get("comparison_foam_state"): comparison_foam_state=str(args.scene.parent/saved["comparison_foam_state"])
            args.compare=args.compare or saved.get("comparison",False)
            if saved.get("shading_statistics"): shading=Shading_statistics(**saved["shading_statistics"])
            if saved.get("comparison_shading_statistics"): comparison_shading=Shading_statistics(**saved["comparison_shading_statistics"])
        if args.time is not None: start_time=args.time
        if args.foam is not None: foam=replace(foam,enabled=args.foam)
        if args.tessendorf: p=p.tessendorf()
        if args.command=="still":
            if args.output.exists() and not args.overwrite: raise FileExistsError(args.output)
            from PIL import Image
            from .editing import make_edited_state
            def make_state(parameters):
                return make_edited_state(parameters,args.device,phase_steps) if post_seed or phase_steps else make_initial_state(parameters,args.device)
            state=make_state(p)
            renderer=Ocean_renderer(make_device(),args.sky,mesh_resolution=(960,576))
            renderer.upload(evaluate(state,start_time))
            renderer.upload_foam(prepare_foam(state,start_time,foam,foam_state,args.foam_preroll))
            if shading: renderer.restore_shading_statistics(shading)
            other=None
            if args.compare:
                other=Ocean_renderer(renderer.device,args.sky,mesh_resolution=(960,576))
                other_state=make_state(p.tessendorf())
                other.upload(evaluate(other_state,start_time))
                other.upload_foam(prepare_foam(other_state,start_time,foam,comparison_foam_state,args.foam_preroll))
                if comparison_shading: other.restore_shading_statistics(comparison_shading)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(renderer.render_image(args.width,args.height,camera,look,left_renderer=other)).save(args.output)
            print(args.output); return
        if args.seconds<=0: raise ValueError("seconds must be positive")
        shots=(Shot(args.seconds,scene.name,scene.description,p,args.compare,camera,look,start_time,shading,comparison_shading,post_seed,phase_steps,
                    foam,foam_state,comparison_foam_state,args.foam_preroll),)
    if args.fps<1 or args.fps>120: raise ValueError("fps must be between 1 and 120")
    if args.max_mbps is not None and (not math.isfinite(args.max_mbps) or args.max_mbps<=0):
        raise ValueError("max-mbps must be finite and positive")
    result=render_shots(shots,args.output,sky=args.sky,device=args.device,width=args.width,height=args.height,
                       fps=args.fps,captions=not args.no_captions,codec=args.codec,overwrite=args.overwrite,max_mbps=args.max_mbps)
    print(f"Wrote {args.output}: {result['duration_seconds']:.2f} s, {result['frames']} frames")
