"""Repeatable visualizer captures; fixed wave state, foam history and cameras."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from encino_waves.model import Wave_parameters, evaluate
from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam
from encino_waves.camera import Camera, look_at, frame_domain
from encino_waves.render import Ocean_renderer, make_device, Look


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="renders/visual_review/current")
    parser.add_argument("--sky")
    parser.add_argument("--resolution",type=int,default=1024)
    parser.add_argument("--width",type=int,default=1280)
    parser.add_argument("--height",type=int,default=720)
    parser.add_argument("--time",type=float,default=12)
    parser.add_argument("--material",choices=("physical","2015"),default="physical")
    parser.add_argument("--foam-model",choices=("legacy","compression"),default="compression")
    parser.add_argument("--emission",type=float,default=1.2)
    parser.add_argument("--compression-threshold",type=float,default=.75)
    parser.add_argument("--wind-speed",type=float,default=17)
    parser.add_argument("--fetch-km",type=float,default=300)
    parser.add_argument("--depth",type=float,default=100)
    parser.add_argument("--swell",type=float,default=.35)
    parser.add_argument("--exposure",type=float,default=0)
    parser.add_argument("--sky-rotation",type=float,default=0)
    parser.add_argument("--haze",type=float,default=.9)
    parser.add_argument("--samples",type=int,choices=(1,4),default=4)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    parameters=Wave_parameters(resolution=args.resolution,wind_speed=args.wind_speed,
                              fetch_km=args.fetch_km,depth=args.depth,swell=args.swell)
    waves=make_edited_state(parameters)
    frame=evaluate(waves,args.time)
    foam_parameters=Foam_parameters(emission_model=args.foam_model,emission=args.emission,
                                   compression_threshold=args.compression_threshold)
    foam=prepare_foam(waves,args.time,foam_parameters,preroll=8)
    renderer=Ocean_renderer(make_device(),args.sky,sample_count=args.samples)
    renderer.upload(frame)
    renderer.upload_foam(foam)
    sun=renderer.sun_direction
    sun_yaw=math.degrees(math.atan2(sun[0],sun[1]))
    cameras={"overview":frame_domain(parameters.domain)}
    for label,angle in (("sun",0),("crosslight",90),("away",180)):
        yaw=math.radians(sun_yaw+angle)
        direction=np.array((math.sin(yaw),math.cos(yaw)))
        cameras[label]=look_at((-direction[0]*450,-direction[1]*450,55),fov=50)
    cameras["sun_low"]=replace(cameras["sun"],height=12,pitch=-3)
    look=Look(material=args.material,sky_gain=2 if args.material=="2015" else 1,
              exposure=args.exposure,sky_rotation=args.sky_rotation,haze=args.haze)
    images=[]
    for name,camera in cameras.items():
        pixels=renderer.render_image(args.width,args.height,camera,look)
        Image.fromarray(pixels).save(output/f"{name}.png")
        image=Image.fromarray(pixels).convert("RGB")
        image.thumbnail((640,360))
        ImageDraw.Draw(image).text((12,12),name,fill="white",stroke_width=2,stroke_fill="black")
        images.append(image)
    sheet=Image.new("RGB",(1280,360*math.ceil(len(images)/2)))
    for i,image in enumerate(images): sheet.paste(image,((i%2)*640,(i//2)*360))
    sheet.save(output/"contact_sheet.jpg",quality=95)
    def digest(tensor): return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
    metadata={"parameters":asdict(parameters),"time":args.time,"sky":renderer.sky_name,
              "look":asdict(look),"sun_direction":sun.tolist(),
              "foam_parameters":asdict(foam_parameters),"antialiasing_samples":args.samples,
              "cameras":{name:asdict(camera) for name,camera in cameras.items()},
              "wave_displacement_sha256":digest(frame.displacement),
              "wave_normal_sha256":digest(frame.normal),"foam_sha256":digest(foam.density)}
    (output/"capture.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"output":str(output),"sky":renderer.sky_name,"sun":sun.tolist()}),flush=True)


if __name__=="__main__": main()
