"""Measure the complete visualization cost, including GPU completion.

Close other viewers before running. Startup HDR filtering is excluded. The
animation case includes waves, trough damping, RGB foam, transfer, materials,
and the final display pass. No image readback or window presentation is timed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import wgpu

from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam, step_foam
from encino_waves.model import Wave_parameters, evaluate
from encino_waves.render import Look, Ocean_renderer, make_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution",type=int,default=1024)
    parser.add_argument("--domain",type=float,default=512.)
    parser.add_argument("--foam-resolution",type=int,default=512)
    parser.add_argument("--crest-crumble",action="store_true",help="Compare the optional crest material to the approved look")
    parser.add_argument("--width",type=int,default=1920)
    parser.add_argument("--height",type=int,default=1080)
    parser.add_argument("--frames",type=int,default=30)
    parser.add_argument("--output",type=Path,default=Path("renders/visual_review/performance.json"))
    args=parser.parse_args()
    if args.frames < 1: parser.error("frames must be positive")
    device=make_device()
    fence=device.create_buffer(size=4,usage=wgpu.BufferUsage.COPY_SRC)
    def wait(): device.queue.read_buffer(fence)
    target=device.create_texture(size=(args.width,args.height,1),format="rgba8unorm",
                                 usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
    target_view=target.create_view()
    waves=make_edited_state(Wave_parameters(resolution=args.resolution,domain=args.domain))
    frame=evaluate(waves,12)
    cases={"previous_viewer":(Look(material="2015",sky_gain=2,exposure=0,sky_rotation=0,haze=1),1),
           "presentation_1_sample":(Look(),1),"presentation_4_samples":(Look(),4)}
    if args.crest_crumble:
        cases={"approved":(Look(),4),"crest_crumble":(Look(crest_crumble=True),4)}
    initial_foam={name:prepare_foam(waves,12,Foam_parameters(
        resolution=args.foam_resolution,
        emission_model="legacy" if name=="previous_viewer" else "compression"),preroll=.5) for name in cases}
    renderers={name:Ocean_renderer(device,sample_count=samples) for name,(_,samples) in cases.items()}
    for name,renderer in renderers.items():
        renderer.upload(frame)
        renderer.upload_foam(initial_foam[name])
    output={"graphics":dict(device.adapter.info),"resolution":args.resolution,
            "foam_resolution":args.foam_resolution,"domain":args.domain,
            "width":args.width,"height":args.height,"frames":args.frames,"results":{}}
    for animation in (False,True):
        histories=initial_foam.copy()
        samples={name:[] for name in cases}
        names=list(cases)
        for i in range(args.frames+5):
            # Alternating order reduces thermal and scheduling bias.
            for name in names if i%2 else names[::-1]:
                renderer=renderers[name]
                wait()
                start=time.perf_counter()
                if animation:
                    current=evaluate(waves,12+(i+1)/30)
                    histories[name]=step_foam(histories[name],current)
                    renderer.upload(current)
                    renderer.upload_foam(histories[name])
                renderer.draw(target_view,args.width,args.height,look=cases[name][0])
                wait()
                if i>=5: samples[name].append((time.perf_counter()-start)*1000)
        label="animated_frame" if animation else "paused_frame"
        output["results"][label]={name:{"median_ms":float(np.median(values)),
                                      "p95_ms":float(np.percentile(values,95))}
                                  for name,values in samples.items()}
        print(json.dumps({label:output["results"][label]}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__": main()
