"""Render identical waves with crest shading and persistent RGB aeration."""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from encino_waves.model import Wave_parameters, evaluate, synchronize
from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam, advance_foam_to, emission_mask
from encino_waves.render import Ocean_renderer, make_device, Look
from encino_waves.camera import frame_domain
from encino_waves.export import Movie_writer, _font

Path("renders").mkdir(exist_ok=True)
waves = make_edited_state(Wave_parameters(resolution=1024))
parameters = Foam_parameters()
foam = prepare_foam(waves,10,parameters,preroll=8)
graphics = make_device()
renderer = Ocean_renderer(graphics)
old = Ocean_renderer(graphics)
camera,look = frame_domain(waves.parameters.domain),Look()
writer = Movie_writer("renders/foam_history_preview.mp4",1280,800,24,overwrite=True,max_mbps=8)
times = []
complete = False
try:
    for i in range(192):
        t = 10+i/24
        frame = evaluate(waves,t)
        synchronize(waves.device)
        start = time.perf_counter()
        foam = advance_foam_to(foam,waves,frame,parameters)
        synchronize(waves.device)
        times.append((time.perf_counter()-start)*1000)
        renderer.upload(frame)
        renderer.upload_foam(foam)
        old.upload(frame)
        pixels = renderer.render_image(1280,800,camera,look,left_renderer=old)
        image = Image.fromarray(pixels).convert("RGB")
        draw = ImageDraw.Draw(image)
        for x,label in ((28,"Crest shading"),(668,"Persistent foam + aeration")):
            draw.text((x,28),label,font=_font(25),fill=(245,245,240),stroke_width=1,stroke_fill=(0,0,0))
        writer.write(np.asarray(image))
        if i == 0:
            image.save("renders/foam_history_preview.png")
            rgb = foam.density.permute(1,2,0).cpu().numpy()
            Image.fromarray(np.uint8(np.clip(1-np.exp(-rgb*3),0,1)*255)).save("renders/foam_rgb_history.png")
        if i%48 == 0: print(f"{i/24:.0f}/8 s",flush=True)
    complete = True
finally:
    writer.close(commit=complete)
result = {"device":str(waves.device),"waves":waves.parameters.resolution,"foam":parameters.resolution,
    "foam_step_median_ms":float(np.median(times[1:])),"foam_step_p95_ms":float(np.percentile(times[1:],95)),
    "mean_rgb_density":foam.density.mean(dim=(1,2)).cpu().tolist(),
    "mean_emission_mask":float(emission_mask(frame,parameters,foam.crest_gain,foam.crest_bias).mean().cpu())}
Path("renders/foam_preview_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2),flush=True)
