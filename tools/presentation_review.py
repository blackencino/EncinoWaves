"""Export a short, deterministic review of the presentation renderer."""
import argparse
from dataclasses import replace
from pathlib import Path

from encino_waves.export import Shot, render_shots
from encino_waves.presets import SCENES, presentation_views


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("renders/presentation_review.mp4"))
    parser.add_argument("--resolution",type=int,default=2048)
    parser.add_argument("--seconds",type=float,default=6,help="Seconds per shot")
    parser.add_argument("--width",type=int,default=1920)
    parser.add_argument("--height",type=int,default=1080)
    parser.add_argument("--fps",type=int,default=24)
    parser.add_argument("--sky")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if not 0 < args.seconds <= 60: parser.error("seconds must be in (0,60]")
    shots=[]
    # Hold camera/light constant in the matched comparison. Other compositions
    # stage the existing six physical seas; no material change alters a wave.
    for index,view_index,comparison in ((2,1,False),(2,0,True),(0,0,False),
                                      (4,2,False),(3,1,False),(5,2,False)):
        scene=SCENES[index]
        parameters=replace(scene.parameters,resolution=args.resolution)
        view=presentation_views(parameters.domain)[view_index]
        title="Earlier model / Encino Waves" if comparison else scene.name
        shots.append(Shot(args.seconds,title,scene.description,parameters,comparison,
                          view.camera,view.look,start_time=12,post_seed=True,foam_preroll=8))
    render_shots(tuple(shots),args.output,sky=args.sky,width=args.width,height=args.height,
                 fps=args.fps,captions=True,overwrite=args.overwrite,max_mbps=12)


if __name__=="__main__": main()
