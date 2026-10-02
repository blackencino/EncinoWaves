"""Capture three ocean compositions using the existing Maya camera and waves.

Run GPU captures serially with other renderer work. ``--describe`` lists the
camera and lighting settings without creating a device or evaluating waves.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from encino_waves.editing import make_edited_state
from encino_waves.foam import Foam_parameters, prepare_foam
from encino_waves.model import Wave_parameters, evaluate
from encino_waves.presets import presentation_views
from encino_waves.render import Ocean_renderer, make_device


def compositions(domain=512.0, exposure_offset=0.0):
    names=("overview", "storm_horizon", "crosslight_crests")
    return {name:(view.camera,replace(view.look,exposure=view.look.exposure+exposure_offset))
            for name,view in zip(names,presentation_views(domain))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="renders/presentation_views/current")
    parser.add_argument("--sky")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--time", type=float, default=12)
    parser.add_argument("--exposure-offset", type=float, default=0)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    parameters = Wave_parameters(resolution=args.resolution)
    views = compositions(parameters.domain, args.exposure_offset)
    records = {name: {"camera": asdict(camera), "look": asdict(look)}
               for name, (camera, look) in views.items()}
    if args.describe:
        print(json.dumps(records, indent=2))
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    waves = make_edited_state(parameters)
    frame = evaluate(waves, args.time)
    foam = prepare_foam(waves, args.time, Foam_parameters(), preroll=8)
    renderer = Ocean_renderer(make_device(), args.sky)
    renderer.upload(frame)
    renderer.upload_foam(foam)
    sheet = Image.new("RGB", (1280, 408*len(views)), "#151a20")
    draw = ImageDraw.Draw(sheet)
    for index, (name, (camera, look)) in enumerate(views.items()):
        image = Image.fromarray(renderer.render_image(args.width, args.height, camera, look))
        image.save(output/f"{name}.png")
        image = image.convert("RGB")
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)
        y = index*408
        sheet.paste(image, (0, y+42))
        draw.text((12, y+12), name, fill="white")
        draw.text((665, y+90),
                  f"Height {camera.height:.1f} m | Heading {camera.yaw:.1f} deg\n"
                  f"Pitch {camera.pitch:.2f} deg | FOV {camera.fov:.0f} deg\n\n"
                  f"Exposure {look.exposure:+.2f} stops\n"
                  f"Sky rotation {look.sky_rotation:+.0f} deg\n"
                  f"Haze {look.haze:.2f}", fill="#c9d2dd", spacing=8)
        print(f"Captured {name}: {output/name}.png", flush=True)
    sheet.save(output/"contact_sheet.jpg", quality=95)

    def digest(tensor):
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

    metadata = {
        "parameters": asdict(parameters), "time": args.time, "sky": renderer.sky_name,
        "compositions": records,
        "wave_displacement_sha256": digest(frame.displacement),
        "wave_normal_sha256": digest(frame.normal), "foam_sha256": digest(foam.density),
    }
    (output/"capture.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
