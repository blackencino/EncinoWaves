"""Compare host uploads with native Metal packing, including synchronization.

Close other GPU viewers for uncontended results. Uses the same synthesized wave
and foam tensors for the transfer cases. The full-frame case also synthesizes
waves and advances foam. Draws include mip generation and the original ocean
shader at 1080p. None include window presentation, UI or image readback.
"""
import argparse
import gc
import json
import time
import numpy as np
import torch
import wgpu
from encino_waves.model import Wave_parameters, make_initial_state, evaluate, synchronize
from encino_waves.foam import Foam_parameters, prepare_foam, step_foam
from encino_waves.render import Ocean_renderer, make_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions",type=int,nargs="+",default=(1024,2048,4096))
    parser.add_argument("--frames",type=int,default=20)
    args = parser.parse_args()
    if args.frames < 1: parser.error("frames must be positive")
    device = make_device()
    # An ordered 4-byte fence avoids both an image readback and the upstream
    # queue.on_submitted_work_done_sync callback ABI bug in wgpu-py 0.32.
    fence = device.create_buffer(size=4,usage=wgpu.BufferUsage.COPY_SRC)
    def wait(): device.queue.read_buffer(fence)
    target = device.create_texture(size=(1920,1080,1),format="rgba8unorm",
                                  usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
    view = target.create_view()
    renderers = {mode:Ocean_renderer(device,transfer=mode) for mode in ("host","metal")}
    for n in args.resolutions:
        state = make_initial_state(Wave_parameters(resolution=n),"mps")
        frame = evaluate(state,12)
        foam = prepare_foam(state,12,Foam_parameters(resolution=512),preroll=.2)
        synchronize(state.device)
        for renderer in renderers.values():
            renderer.upload(frame)
            renderer.upload_foam(foam)
            renderer.draw(view,1920,1080)
        wait()
        result = {"resolution":n,"graphics":device.adapter.info["device"],"samples":args.frames}
        for case in ("transfer_only","transfer_and_draw","full_frame"):
            samples = {mode:[] for mode in renderers}
            histories = {mode:foam for mode in renderers}
            for i in range(args.frames+4):
                # Alternate order so drift/thermal conditions affect both modes.
                order = ("host","metal") if i%2 else ("metal","host")
                for mode in order:
                    renderer = renderers[mode]
                    renderer.foam_state = None
                    wait()
                    start = time.perf_counter()
                    current_frame = evaluate(state,12+(i+1)/24) if case=="full_frame" else frame
                    renderer.upload(current_frame)
                    if case=="full_frame":
                        histories[mode] = step_foam(histories[mode],current_frame)
                    renderer.upload_foam(histories[mode])
                    if case!="transfer_only": renderer.draw(view,1920,1080)
                    elif renderer.gpu_transfer: renderer.gpu_transfer.before_graphics()
                    # write_texture staging work is flushed even without a draw.
                    device.queue.submit([])
                    wait()
                    elapsed = (time.perf_counter()-start)*1000
                    if i >= 4: samples[mode].append(elapsed)
            result[case] = {
                mode:{"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95))}
                for mode,values in samples.items()}
        print(json.dumps(result),flush=True)
        del state,frame,foam
        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__": main()
