"""Compare the same sea with trough damping off and at its 0.5 default."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from encino_waves.model import Wave_parameters, evaluate
from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam, advance_foam_to
from encino_waves.render import Ocean_renderer, make_device, Look
from encino_waves.camera import frame_domain
from encino_waves.export import Movie_writer, _font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution",type=int,default=1024)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--seconds",type=float,default=6)
    args = parser.parse_args()
    Path("renders").mkdir(exist_ok=True)
    p = Wave_parameters(resolution=args.resolution,trough_damping=0)
    off = make_edited_state(p,args.device)
    on = replace(off,parameters=replace(p,trough_damping=.5))
    foam_parameters = Foam_parameters()
    foam = prepare_foam(off,10,foam_parameters,preroll=6)
    graphics = make_device()
    before,after = Ocean_renderer(graphics),Ocean_renderer(graphics)
    camera,look = frame_domain(p.domain),Look()
    writer = Movie_writer("renders/trough_damping_comparison.mp4",1280,800,24,
                          overwrite=True,max_mbps=8)
    complete = False
    try:
        for i in range(max(1,round(args.seconds*24))):
            time = 10+i/24
            frame = evaluate(off,time)
            foam = advance_foam_to(foam,off,frame,foam_parameters)
            before.upload(frame)
            after.upload(evaluate(on,time))
            before.upload_foam(foam)
            after.upload_foam(foam)
            after.restore_shading_statistics(before.shading_statistics)
            if i == 0:
                # Prime environment and texture mipmaps before the comparison.
                after.render_image(1280,800,camera,look,left_renderer=before)
            pixels = after.render_image(1280,800,camera,look,left_renderer=before)
            image = Image.fromarray(pixels).convert("RGB")
            draw = ImageDraw.Draw(image)
            for x,label in ((28,"Trough damping off"),(668,"Trough damping 0.5")):
                draw.text((x,28),label,font=_font(24),fill=(245,245,240),
                          stroke_width=1,stroke_fill=(0,0,0))
            writer.write(np.asarray(image))
            if i == 0:
                image.save("renders/trough_damping_comparison.png")
                for label,renderer in (("off",before),("on",after)):
                    Image.fromarray(renderer.render_image(1280,800,camera,look)).save(f"renders/trough_damping_{label}.png")
            if i%48 == 0: print(f"{i/24:.0f}/{args.seconds:g} s",flush=True)
        complete = True
    finally:
        writer.close(commit=complete)
    print(json.dumps({"device":str(on.device),"resolution":args.resolution,
                      "damping":on.parameters.trough_damping,"sky":after.sky_name}),flush=True)


if __name__ == "__main__": main()
