"""Compare fresh crest fill or wind streaks on exactly the same ocean history."""
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from encino_waves.model import Wave_parameters, evaluate
from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam, advance_foam_to, save_foam
from encino_waves.render import Ocean_renderer, make_device, Look
from encino_waves.camera import frame_domain
from encino_waves.export import Movie_writer, _font

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("detail",choices=("crests","streaks"))
parser.add_argument("--seconds",type=float,default=4)
args = parser.parse_args()
streaks = args.detail == "streaks"
parameters = Wave_parameters(resolution=1024,wind_speed=24,swell=.8) if streaks else Wave_parameters(resolution=1024)
waves = make_edited_state(parameters)
foam_parameters = Foam_parameters()
start,preroll = (60,60) if streaks else (10,8)
print(f"Building {preroll} seconds of foam history on {waves.device}",flush=True)
foam = prepare_foam(waves,start,foam_parameters,preroll=preroll)
graphics = make_device()
renderer = Ocean_renderer(graphics)
camera,look = frame_domain(parameters.domain),Look()
Path("renders").mkdir(exist_ok=True)
stem = f"renders/foam_{args.detail}_preview"
writer = Movie_writer(stem+".mp4",1280,800,24,overwrite=True,max_mbps=8)
complete = False
try:
    for i in range(round(args.seconds*24)):
        frame = evaluate(waves,start+i/24)
        foam = advance_foam_to(foam,waves,frame,foam_parameters)
        renderer.upload(frame)
        renderer.upload_foam(replace(foam,windrows=None) if streaks else foam)
        before = renderer.render_image(640,800,camera,look if streaks else replace(look,crest_foam=0))
        renderer.upload_foam(foam)
        after = renderer.render_image(640,800,camera,look)
        image = Image.fromarray(np.concatenate((before,after),axis=1)).convert("RGB")
        draw = ImageDraw.Draw(image)
        labels = ("Foam","Foam + wind streaks") if streaks else ("Accumulated foam","With light fresh crests")
        for x,label in zip((28,668),labels):
            draw.text((x,28),label,font=_font(24),fill=(245,245,240),stroke_width=1,stroke_fill=(0,0,0))
        writer.write(np.asarray(image))
        if i == 0:
            image.save(stem+".png")
            if streaks:
                density = foam.windrows.cpu().numpy()
                Image.fromarray(np.uint8(np.clip(1-np.exp(-density*8),0,1)*255)).save(stem+"_density.png")
                save_foam(stem+".npz",foam)
                scene = {"parameters":asdict(parameters),"camera":asdict(camera),"look":asdict(look),
                    "time":start,"post_seed":True,"sky":str(Path(renderer.sky_name).resolve()),
                    "foam":asdict(foam_parameters),"foam_state":Path(stem).name+".npz",
                    "shading_statistics":asdict(renderer.shading_statistics)}
                Path(stem+".json").write_text(json.dumps(scene,indent=2)+"\n")
                print(f"Streak density: mean {density.mean():.4f}, max {density.max():.4f}",flush=True)
        if i%48 == 0: print(f"{i/24:g}/{args.seconds:g} s",flush=True)
    complete = True
finally:
    writer.close(commit=complete)
print(stem+".mp4",flush=True)
