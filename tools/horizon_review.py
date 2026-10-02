"""Isolate horizon specks with moving waves, flat water, and a sky-only control.

Only this process's renderer instance is changed for the sky-only draw. Shared
renderer/shader sources and input assets remain untouched.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter
import torch

from encino_waves.camera import Camera
from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam, advance_foam_to
from encino_waves.model import Wave_frame, Wave_parameters, evaluate
from encino_waves.render import Look, Ocean_renderer, make_device, FOAM_SURFACE_WGSL, FOAM_DETAIL_WGSL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="renders/presentation_views/current/capture.json")
    parser.add_argument("--output", default="renders/horizon_review/current")
    parser.add_argument("--view", default="storm_horizon")
    parser.add_argument("--foam-model", choices=("legacy", "compression"), default="legacy")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--extended-controls", action="store_true")
    args = parser.parse_args()
    scene = json.loads(Path(args.scene).read_text())
    view = scene["compositions"][args.view]
    camera = Camera(**view["camera"])
    look = Look.from_dict(view["look"])
    parameters = Wave_parameters(**scene["parameters"])
    foam_parameters = Foam_parameters(emission_model=args.foam_model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    waves = make_edited_state(parameters)
    foam = prepare_foam(waves, 12, foam_parameters, preroll=8)
    initial_foam = foam
    renderer = Ocean_renderer(make_device(), scene["sky"])
    captures = {}
    for time in (12, 12.05, 12.1):
        frame = evaluate(waves, time)
        if time > 12:
            foam = advance_foam_to(foam, waves, frame, foam_parameters)
        renderer.upload(frame)
        renderer.upload_foam(foam)
        name = f"waves_{time:.2f}".replace(".", "_")
        captures[name] = renderer.render_image(args.width, args.height, camera, look)
        print(f"Captured {name}", flush=True)

    displacement = torch.zeros_like(frame.displacement)
    displacement[...,3] = -1
    normal = torch.zeros_like(frame.normal)
    normal[...,2] = 1
    flat = Wave_frame(parameters, 12, displacement, normal)
    renderer.statistics_parameters = None
    renderer.upload(flat)
    renderer.upload_foam(None)
    captures["flat_water"] = renderer.render_image(args.width, args.height, camera, look)
    original_count = renderer.index_count
    renderer.index_count = 0
    captures["sky_only"] = renderer.render_image(args.width, args.height, camera, look)
    renderer.index_count = original_count
    print("Captured flat_water and sky_only", flush=True)

    if args.extended_controls:
        moving = evaluate(waves, 12)
        for name, test_frame in (
            ("displacement_only", Wave_frame(parameters,12,moving.displacement,normal)),
            ("normals_only", Wave_frame(parameters,12,displacement,moving.normal)),
        ):
            renderer.statistics_parameters = None
            renderer.upload(test_frame)
            captures[name] = renderer.render_image(args.width,args.height,camera,look)
            print(f"Captured {name}", flush=True)

        # Counterfactual diagnostic: ordinary pixel-center interpolation can
        # extrapolate off subpixel MSAA triangles. Production uses centroids.
        shader_path = Path(__file__).resolve().parents[1]/"python/encino_waves/shaders/ocean.wgsl"
        shader_text = shader_path.read_text()
        shader_text = shader_text.replace("@interpolate(perspective, centroid) ", "")
        shader = renderer.device.create_shader_module(code=FOAM_SURFACE_WGSL+"\n"+FOAM_DETAIL_WGSL+"\n"+shader_text)
        renderer.ocean_pipeline = renderer.device.create_render_pipeline(
            layout=renderer.pipeline_layout,
            vertex={"module":shader,"entry_point":"ocean_vertex"},
            fragment={"module":shader,"entry_point":"ocean_fragment","targets":[{"format":"rgba16float"}]},
            primitive={"topology":"triangle-list","cull_mode":"none"},
            multisample={"count":renderer.sample_count},
            depth_stencil={"format":"depth32float","depth_write_enabled":True,"depth_compare":"less"})
        foam = initial_foam
        for time in (12,12.05,12.1):
            frame = evaluate(waves,time)
            if time>12:
                foam = advance_foam_to(foam,waves,frame,foam_parameters)
            renderer.statistics_parameters = None
            renderer.upload(frame)
            renderer.upload_foam(foam)
            name = f"pixel_center_{time:.2f}".replace(".","_")
            captures[name] = renderer.render_image(args.width,args.height,camera,look)
            print(f"Captured {name}", flush=True)

    # These bounds encompass the previously observed 1280x720 storm-horizon
    # specks. Keep the full frames as the authority when using another camera.
    x0, y0, x1, y1 = 50, 219, 610, 253
    sheet = Image.new("RGB", ((x1-x0)*2, 106*len(captures)), "#151a20")
    draw = ImageDraw.Draw(sheet)
    records = {}
    strips = []
    points = [(99,234), (391,234), (407,234), (533,234), (559,237)]
    for index, (name, pixels) in enumerate(captures.items()):
        image = Image.fromarray(pixels)
        image.save(output/f"{name}.png")
        strip = image.convert("RGB").crop((x0,y0,x1,y1)).resize(((x1-x0)*2,(y1-y0)*2), Image.Resampling.NEAREST)
        sheet.paste(strip, (0, index*106+28))
        draw.text((8,index*106+8), name, fill="white")
        if name.startswith("waves"):
            strips.append(strip)
        rgb = pixels[...,:3].astype(np.float64)
        luminance = rgb@np.array([.2126,.7152,.0722])
        residual = luminance-median_filter(luminance, size=3)
        hits = np.argwhere(residual[225:245]>15)
        records[name] = {
            "isolated_bright_pixels": [[int(x),int(y+225),float(residual[y+225,x])] for y,x in hits],
            "tracked_pixels": [{"xy":[x,y],"rgb":pixels[y,x,:3].tolist(),"residual":float(residual[y,x])} for x,y in points],
        }
    sheet.save(output/"horizon_controls.png")
    strips[0].save(output/"horizon_motion.gif", save_all=True, append_images=strips[1:], duration=250, loop=0)
    first = captures["waves_12_00"][219:245,:,:3].astype(np.int16)
    for name in ("waves_12_05", "waves_12_10", "flat_water", "sky_only"):
        difference = np.abs(captures[name][219:245,:,:3].astype(np.int16)-first)
        records[name]["horizon_difference_from_t12"] = {
            "max_channel": int(difference.max()), "mean_channel": float(difference.mean()),
            "changed_pixels": int(np.count_nonzero(difference.max(axis=-1))),
        }
    metadata = {"camera":asdict(camera),"look":asdict(look),"foam":asdict(foam_parameters),
                "parameters":asdict(parameters),"sky":renderer.sky_name,"captures":records}
    (output/"results.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(records,indent=2), flush=True)


if __name__ == "__main__":
    main()
