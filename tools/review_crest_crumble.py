"""Matched optional-crest-material review; waves and RGB history are shared."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from encino_waves.camera import look_at
from encino_waves.editing import make_edited_state
from encino_waves.export import Movie_writer
from encino_waves.foam import Foam_parameters, prepare_foam, step_foam
from encino_waves.model import Wave_parameters, evaluate
from encino_waves.presets import presentation_views
from encino_waves.render import Look, Ocean_renderer, make_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("renders/crest_crumble"))
    parser.add_argument("--cases",nargs="+",default=["ordinary","chaos","ordered","shallow","quiet","close"])
    parser.add_argument("--resolution",type=int,default=4096)
    parser.add_argument("--width",type=int,default=3840)
    parser.add_argument("--height",type=int,default=2160)
    parser.add_argument("--seconds",type=float,default=0.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    views=presentation_views(1000.)
    base=Wave_parameters(resolution=args.resolution,domain=1000.,depth=1000.)
    cases={
        "ordinary":(base,views[0].camera),
        "chaos":(replace(base,wind_speed=32.,fetch_km=20.,swell=-.6),views[0].camera),
        "ordered":(replace(base,wind_speed=32.,fetch_km=1250.,swell=1.2),views[3].camera),
        "shallow":(replace(base,wind_speed=32.,fetch_km=1250.,swell=-.7,depth=1.5),views[3].camera),
        "quiet":(replace(base,wind_speed=5.),views[0].camera),
        "close":(base,look_at((150.,-150.,28.),fov=48.)),
    }
    renderer=Ocean_renderer(make_device(),mesh_resolution=(1920,1152))
    off,on=Look(),Look(crest_crumble=True)
    results={}
    for name in args.cases:
        parameters,camera=cases[name]
        waves=make_edited_state(parameters)
        foam=prepare_foam(waves,12.,Foam_parameters(resolution=1024),preroll=6.)
        folder=args.output/name
        folder.mkdir(exist_ok=True)
        movie=None
        if args.seconds>0:
            movie=Movie_writer(folder/"comparison.mp4",2560,760,24)
        start=time.perf_counter()
        complete=False
        warmed_pairs=0
        try:
            for index in range(max(1,round(args.seconds*24))):
                frame=evaluate(waves,12.+index/24)
                foam=step_foam(foam,frame)
                renderer.upload(frame)
                renderer.upload_foam(foam)
                baseline=renderer.render_image(args.width,args.height,camera,off)
                candidate=renderer.render_image(args.width,args.height,camera,on)
                control=renderer.render_image(args.width,args.height,camera,off)
                # Check the return to baseline, not just the two different
                # settings. A first-use Metal draw can have transient sampling
                # differences after a large view/input change. Do not attribute
                # such differences to the optional material.
                if not np.array_equal(baseline,control):
                    warmed_pairs+=1
                    baseline=control
                    candidate=renderer.render_image(args.width,args.height,camera,on)
                    control=renderer.render_image(args.width,args.height,camera,off)
                    if not np.array_equal(baseline,control):
                        raise RuntimeError("Repeated baseline differs; this pair is not a valid material comparison")
                if index==0:
                    Image.fromarray(baseline).save(folder/"approved.png")
                    Image.fromarray(candidate).save(folder/"crumble.png")
                    difference=np.abs(candidate.astype(np.int16)-baseline.astype(np.int16))[...,:3]
                    results[name]={"mean_channel_difference":float(difference.mean()),
                        "max_channel_difference":int(difference.max()),
                        "changed_pixels_over_2_levels_percent":float(100*np.mean(difference.max(axis=-1)>2))}
                    print(name,results[name],flush=True)
                if movie:
                    comparison=Image.new("RGB",(2560,760),(10,17,21))
                    for column,(pixels,label) in enumerate(((baseline,"Approved"),(candidate,"Optional crest crumble"))):
                        comparison.paste(Image.fromarray(pixels).convert("RGB").resize((1280,720),Image.Resampling.LANCZOS),(column*1280,40))
                        ImageDraw.Draw(comparison).text((column*1280+20,12),name+" | "+label,fill="white")
                    movie.write(np.asarray(comparison))
            complete=True
        finally:
            if movie: movie.close(commit=complete)
        results[name]["seconds"]=time.perf_counter()-start
        results[name]["warmed_pairs"]=warmed_pairs
        (args.output/"measurements.json").write_text(json.dumps(results,indent=2)+"\n")
    print("Wrote",args.output,flush=True)


if __name__=="__main__": main()
