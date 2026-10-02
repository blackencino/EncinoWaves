"""Compare foam coverage over the actual presentation seas, at fixed settings."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam
from encino_waves.model import evaluate
from encino_waves.presets import SCENES
from encino_waves.render import Ocean_renderer, make_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("renders/foam_response"))
    parser.add_argument("--resolution",type=int,default=1024)
    parser.add_argument("--threshold",type=float,default=.75)
    parser.add_argument("--emitter",choices=("legacy","compression"),default="compression")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    p=Foam_parameters(emission_model=args.emitter,compression_threshold=args.threshold)
    renderer=Ocean_renderer(make_device())
    sheet=Image.new("RGB",(1440,405*3))
    results=[]
    for i,scene in enumerate(SCENES):
        waves=make_edited_state(replace(scene.parameters,resolution=args.resolution))
        frame=evaluate(waves,12)
        foam=prepare_foam(waves,12,p,preroll=8)
        renderer.upload(frame)
        renderer.upload_foam(foam)
        pixels=renderer.render_image(1440,810,scene.camera,scene.look)
        image=Image.fromarray(pixels)
        image.save(args.output/f"{i+1}.png")
        image=image.resize((720,405),Image.Resampling.LANCZOS).convert("RGB")
        ImageDraw.Draw(image).text((12,12),scene.name,fill="white",stroke_width=2,stroke_fill="black")
        sheet.paste(image,((i%2)*720,(i//2)*405))
        texture=renderer.foam_material_texture
        size=texture.size[0]
        data=renderer.device.queue.read_texture({"texture":texture},
             {"bytes_per_row":size*8,"rows_per_image":size},(size,size,1))
        material=np.frombuffer(data,np.float16).reshape(size,size,4).astype(np.float32)
        record={"name":scene.name,"parameters":asdict(waves.parameters),
                "surface_coverage":float(material[...,0].mean()),
                "surface_density":float(foam.density[0].mean().cpu()),
                "height_rms_m":float(frame.displacement[...,2].std().cpu())}
        results.append(record)
        print(json.dumps(record),flush=True)
    sheet.save(args.output/"contact_sheet.jpg",quality=95)
    (args.output/"results.json").write_text(json.dumps({"foam":asdict(p),"seas":results},indent=2)+"\n")


if __name__=="__main__": main()
