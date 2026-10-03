## EncinoWaves

This branch adds a functional Python GPU implementation and a native viewer.
Ocean compute uses **PyTorch: Metal on Apple silicon, CUDA on NVIDIA**. Graphics
use **wgpu: Metal, Vulkan, or D3D12**. The original C++ source remains in `src/`.
The presentation renderer adds environment lighting, filtered reflections and
persistent foam without changing the wave formulation. Optional trough damping
is a separate spatial appearance filter.

### Run

Python 3.11 or later is required. On this Mac the environment is already installed:

```sh
./run_viewer.command
```

If this worktree shares another checkout's virtual environment, prefix the
Python commands below with `PYTHONPATH=python` so they use this branch's source.
The viewer launcher already does this.

Rendering polish, the continuous School presentation, and the optional foam
fringe are consolidated on `school/gpu-showcase`. Use this checkout for the
viewer; the separate visual-polish and crest-study worktrees have been retired.
The geometric crest wrinkle is excluded.

### School rehearsal viewer

Run `./run_school_viewer.command` for the separate School presentation viewer.
It opens a large, resizable **16:9** preview fitted to the desktop, including on
Retina displays. Movie output remains fixed at **1920×1080**.
The controls stay within **y=200–880**, reserving 200 pixels of clearance at
the top and bottom of the HD frame for a conservative presentation layout.

Startup is an unrecorded preparation view: adjust the sea and camera first.
Pressing R first builds six seconds of foam history, then starts a four-second black **Encino Waves / Interactive Ocean Synthesis**
title card with **Christopher Jon Horvath** beneath the subtitle, followed by a
smooth reveal. The default sea has a 1024 m domain,
17 m/s wind, 100 m depth, 300 km fetch, trough damping 1, and half-strength foam
fringe. The camera is 145.2 m high, pitch −18.1°, heading −139.9°.
School presets use **40.735° vertical / 66.850° horizontal FOV**: a 20 mm
rectilinear lens equivalent on the [ALEXA Mini's 26.40×14.85 mm UHD image area](https://www.arri.com/en/cine-systems/cine-cameras/legacy-cine-cameras/alexa-mini).
This adapts the wide end of Russell Boyd's camera-barge lens choice in
[*Master and Commander*](https://theasc.com/article/hell-high-water-master-commander/)
to our 16:9 framing; it is a composition choice, not a universal ocean lens.

Press **R** to record a performance, then **R** to cut and automatically save.
Foam preparation and checkpoint saving finish before the recording clock starts;
the final render loads that warmed foam. For audio sync, cue from the title/REC
clock appearing after preparation, rather than from the R keypress.
The operator clock shows elapsed and remaining time, including during the title;
it is omitted from the movie. **Render rehearsal** renders the saved take offline.
**W** selects Wind Speed, **D** Ocean Depth, **F** Fetch,
**S** Swell, and **M** Foam. Foam adjusts visible foam strength from 0–200%,
is recorded for playback, and stays out of telemetry.
Left/right arrow keys accelerate the selected control; release
to let it settle with a heavier glide (0.30 s acceleration / 0.45 s release response).
The arrows' weight follows actual movement. Press the mode key
again to dismiss it, or let the label fade after 1.5 seconds without input.
Each press of **E** advances directly through the four named seas, wrapping to
the first after the last. Every example
retains the same patch and seed, with smooth changes in sea state and camera.
**I** toggles large title-safe telemetry: wind speed in knots, ocean depth in
meters, fetch in kilometers, and unitless swell. Its short fades and visibility
are recorded in the take. Patch size is omitted from the presentation overlay.
**C** toggles camera height, pitch, yaw, and field of view in the preview only;
this display and its toggle events are excluded from the recording and movie.

Live Maya camera gestures are unchanged. The recording keeps the camera before
an adjustment and its settled position after four seconds without another tweak;
offline playback makes one eased move between those endpoints. Control mode and
direction edges, and ease boundary values, use **wall-clock time**, independent
of preview frame rate. **Tab** opens setup/files, including **Reset opening ocean**.
Each recording starts its title card again from the currently configured sea.
Keep each rehearsal folder intact: its starting scene and foam checkpoints live
beside the timeline. The default location is the gitignored
`renders/school_YYYYMMDD_HHMMSS_<unique>/rehearsal.json`. Rehearsals autosave once
per second. **Save recording as** in setup copies the take and its checkpoints to
the chosen location; **Open rehearsal** reopens a saved take for rendering.

Offline playback uses exact **24 fps** timing and **4096² waves**, with the
recorded camera, lighting, at least 1024² foam, and presentation overlays. Output is an
**H.264 High MOV**, 80 Mbps target / 120 Mbps maximum, with an explicit sRGB to
Rec.709 conversion. One second of black at each end leaves **4:58** for rehearsal
content within the five-minute limit. The session's recording buttons are omitted
from the movie. Render progress is in `renders/school_export.log`.

```sh
./run_school_viewer.command
.venv/bin/python -m encino_waves school-render renders/your_take/rehearsal.json renders/school.mov
```

These renders are picture-only. Add the presenter's narration and a synchronized
SRT or WebVTT file before sharing; overlay text is not a closed-caption track.
For a separate phone voice memo, start audio first and say a short sync cue while
pressing R. Leave that cue in the original memo for alignment. The rendered
timeline begins after the movie's one-second black head; account for that offset
when adding narration and captions.
Nimbus Sans Regular is bundled for presentation overlays, with its upstream
license and attribution in `python/encino_waves/assets/fonts/`.

For the 2048-wave / 1024-foam comparison scene with the fringe enabled:

```sh
./run_viewer.command --scene scenes/foam_fringe.json
```

Compare **Off** and **Foam fringe** under **Camera & light → Crest treatment**.
Space pauses for a matched comparison; treatment changes preserve foam history.

For a fresh macOS or Linux checkout:

```sh
bash tools/setup.sh
.venv/bin/python -m encino_waves view
```

The dependencies install from binary wheels. On macOS, the viewer automatically
builds a small Metal bridge on first use, using Apple's Command Line Tools
(`xcode-select --install` if missing). The build takes a few seconds and is cached;
there is no Rust, CUDA toolkit or Conan build. This keeps wave and foam maps on
the GPU through rendering. If the bridge is unavailable, a warning identifies
the slower host-upload fallback. `ENCINO_WAVES_TRANSFER=metal` requires the native
path; `ENCINO_WAVES_TRANSFER=host` selects the old path for comparison.

On Linux, graphics need a working Vulkan driver and a
desktop session. On Windows, create a Python virtual environment, run
`python -m pip install -e '.[test]'`, then `python -m encino_waves view`.
NVIDIA compute needs a CUDA-enabled PyTorch wheel and a compatible NVIDIA driver.
`doctor` reports the actual compute and graphics devices. Auto device selection
requires a GPU; CPU reference work must explicitly request `--device cpu`.

### Artist controls

The main viewer controls are wind speed, fetch, depth, swell and wind direction.
**Hasselmann** is the default; the alternative spreading models are in
**Resolution & model**. Advanced controls include resolution through 4096²,
patch size, spectrum, dispersion, pinch and
random seed. Units are metres,
seconds, m/s, and kilometres of fetch. The original extended slider ranges are
retained; the extreme wind settings are artist controls, not a claim that the
empirical data validates every such condition. Ctrl-click a slider to enter a value.

Wind speed, fetch, depth and swell edit continuously on the GPU. They reuse
the seeded wave components; depth edits also preserve their accumulated phase.
Wind direction rotates the mesh and normals, preserving the exact wave pattern.
The simulation stays in its original +X frame; 90 degrees places it along +Y.
The controls have a short, 0.1-second response to smooth pointer motion. Domain,
resolution and seed changes build a new wave basis. Space pauses propagation
while leaving these controls live.

**Trough damping** is on by default at **0.5**, with a range of **0–1**. It smooths
the synthesized height field over a small world-space distance, then blends
toward that surface only in low troughs. The default 10 cm smoothing radius
(Gaussian sigma) quiets ripples around half a metre and smaller. Horizontal
displacement and the crest source stay unchanged; no extra FFTs are needed.
The blend is at most halfway at 0.5, and fully smoothed in deep troughs at 1.
Toggle it or adjust the amount in **Resolution & model**; edits preserve phase.
If the wave grid cannot resolve those ripples, damping does nothing: a 512 m
patch at 1024² is unchanged. This corrects the earlier metre-scale filter, which
altered the broad wave character. Scenes saved before trough damping existed
still load with it off.

The **Foam & aeration** panel controls a separate, persistent RGB map: surface foam,
shallow bubbles, and deeper bubbles. Breaking crests emit through slowly evolving
fractal breakup; existing foam diffuses, fades, and exchanges from shallow to deep.
The shader uses it for lingering surface patches and underwater scattering.
The wave spectrum and displacement are unchanged. Turn **Persistent foam** off
to use immediate crest shading.

New oceans emit from actual surface compression. Calm water produces little
foam; stronger breaking produces more, and shallow-water damping reduces the
source naturally. This is a breaking heuristic, not a calibrated Beaufort model.
The advanced settings retain **Original crests** for comparison and older scenes.

Foam builds during playback; pausing freezes its history. **Reset foam** clears
it without restarting the waves. Physical edits retain the foam from previous
conditions, however large the change. A new patch/seed, discrete model changes,
and time scrubbing clear history. Rotating the ocean rotates the existing foam.
The map defaults to 512² independently of wave resolution, with options through
2048². Surface lifetime is adjustable in the panel; emission, underwater strength,
breakup, spreading, exchange and the source algorithm are in **Advanced foam**.

The defaults are emission **1.20**, surface lifetime **1.5 s**, breakup **1.00**,
spreading **0.56 m²/s**, shallow-to-deep exchange **0.29/s**, and underwater
bubbles **1.25**.

Fresh surface foam stays connected and becomes more perforated as it ages.
Coverage is filtered before rendering so distant whitecaps keep their area.
Subtle, metre-scaled foam relief affects only the material, never the waves.
The wind-streak experiment remains removed. Saved scenes keep their RGB history
and original emission model; retired layer settings and streak maps are ignored.

Saved scenes include adjacent `.foam_state.npz` and, for comparisons,
`.comparison_foam_state.npz` checkpoints. Keep these files beside their scene
JSON. Exports resume the saved history, including at higher wave resolutions.
Without a checkpoint, offline renders build six seconds of foam before the
first output frame; use `--foam-preroll 0` for a fresh start or `--no-foam` for
immediate crest shading. Old scenes without foam settings keep foam disabled.

The camera preserves the original Z-up Maya center-of-interest model:

| Input | Action |
| --- | --- |
| Alt + left drag | Tumble around the center of interest |
| Alt + middle drag / Ctrl + left drag | Track |
| Alt + right drag / Shift + left drag | Dolly |
| F | Frame the ocean patch from above |
| Space | Play / pause |
| C | Toggle matched comparison |
| Tab | Hide / show controls |
| S | Save a 1080p still and its scene JSON in `renders/` |
| 1–6 | Load a preset |
| F11 | Fullscreen |

The comparison uses Pierson–Moskowitz with positive cosine-squared spreading
and zero swell for **Tessendorf mode**. Seed, camera, scale,
dispersion and shading are shared. Both sides use the same foam settings.

**Camera & light → Open HDR sky** loads a local latitude/longitude `.hdr` or
`.exr`. Alternatively:

```sh
.venv/bin/python -m encino_waves view --sky '/path/to/Dutch Skies/sky.exr'
```

`ENCINO_WAVES_SKY` sets the default sky. Files in `assets/local/` are also found
automatically and are ignored by Git. Commercial sky files are not bundled or
downloaded. If none is found, the viewer identifies its procedural fallback in
the model panel. This fallback is not a Dutch Skies asset.

On this Mac, **Dutch Skies 360 — Autumn Pack 01, 03a** is installed locally and
loads by default. The viewer uses the full 4000×2000 reflection HDR; automatic
discovery prefers a pack's `_Ref` panorama over its small `_Env` lighting map.
The pack remains in the ignored `assets/local/` directory for this checkout.

**Camera & light → Compose view** offers the raised overview, a storm horizon,
a view across the crests, and **Sunset** (height 92.5 m, pitch −15.4°, heading
−487.6°). These change only camera and lighting. The starting
camera still uses the original Maya framing rule, well above the surface.

The default **Presentation** material uses bounded dielectric reflection,
refraction-aware water scattering, integrated HDR sky lighting, filtered GGX
reflections, and sky-lit surface foam. Wave detail smaller than a pixel broadens
the reflection instead of sparkling. Four-sample antialiasing resolves in linear
HDR before a neutral highlight rolloff and sRGB display conversion.

The local autumn panorama is reprojected from its upper sky to remove buildings
and turbines; sky and sea share a continuous haze at the horizon. No modified
HDR file is written or distributed. **Water shading → Original 2015** keeps the
previous GLSL-derived material and original panorama available for comparison.
See [rendering notes and references](docs/IMPLEMENTATION.md#presentation-rendering)
for the approximations and validation.

### Scenes, stills and movies

Save still also writes the physical parameters, camera, light settings, time and
the phase segments needed to reproduce depth edits at any export resolution.
Reopen that JSON through **Files & export → Open scene**, or use it from the CLI.
The same panel exports a deterministic 1080p movie, with frame times independent
of playback speed. Movie exports run in a separate process; progress is in
`renders/export.log`.

```sh
.venv/bin/python -m encino_waves view --scene renders/your_scene.json
.venv/bin/python -m encino_waves still renders/ocean.png --resolution 4096
.venv/bin/python -m encino_waves render renders/shot.mp4 --scene renders/your_scene.json --seconds 15 --resolution 2048
.venv/bin/python -m encino_waves render renders/comparison.mp4 --compare --seconds 10
PYTHONPATH=python .venv/bin/python -m encino_waves demo renders/review.mp4 --preview --resolution 2048 --width 1920 --height 1080
PYTHONPATH=python .venv/bin/python -m encino_waves demo renders/school.mp4 --resolution 4096 --foam-resolution 1024 --width 3840 --height 2160
```

`demo` is exactly five minutes at 24 fps, with no cuts or reseeds. It starts in
1,000 m deep water and keeps a **1 km patch** throughout. Only one physical
control changes at a time; the Maya camera moves smoothly during separate holds.
Wave phases and foam history continue through every change. A matched earlier-model
comparison fades onto the left half without changing the camera projection.

At the same strong wind speed, short fetch and low swell give **Chaos**; long
fetch and high swell give **Lawful evil**. Reducing swell, then depth to 1.5 m,
reveals shallow chop. The film visits all four compositions, including Sunset,
while lighting remains fixed. These are smooth edits of the spectral conditions,
not a simulation of the time a real sea takes to respond to changing weather.
School exports default to **4096² waves and 1024² foam**; `--resolution 2048`
is also suitable for a lighter export. Image dimensions are independent: the
command above writes a 3840×2160 movie. The interactive viewer retains its
existing defaults.
[The narration draft](docs/VIDEO_SCRIPT.txt) follows that sequence. The generated
movie has captions but no recorded narration or music. `--preview` compresses
the same camera/control timeline to **60 seconds**, with waves still moving at
real time. Its faster edits are intended for reviewing composition and sequence.

Use `--sky`, `--width`, `--height`, `--fps`, `--time` (single shots),
`--no-captions`, or `--codec prores` as needed. ProRes should use a `.mov` output.
`--foam-resolution` also works for stills and saved-scene movies; changing a
checkpoint's foam resolution rebuilds its history with the requested preroll.
`--max-mbps 8` creates a compact H.264 review copy; omit it for the quality default
(CRF 16). Full-quality ocean movies need several GB of free space. Existing
outputs require `--overwrite`. Failed exports discard their temporary movie and
leave an existing completed output intact.

For a short presentation review, reproducible lighting comparisons, or a complete
GPU frame benchmark:

```sh
PYTHONPATH=python .venv/bin/python tools/presentation_review.py
PYTHONPATH=python .venv/bin/python tools/visual_review.py
PYTHONPATH=python .venv/bin/python tools/benchmark_visualizer.py
```

Review captures and movies stay under the ignored `renders/` directory. The
visual-review manifest records hashes of the displacement and normal fields.

### Numerical verification and device acceptance

The wave core has no viewer dependency and evaluates at absolute time:

```python
from encino_waves import Wave_parameters, make_initial_state, evaluate

parameters = Wave_parameters(resolution=2048, wind_speed=17, fetch_km=300, depth=30)
state = make_initial_state(parameters, "auto")
frame = evaluate(state, 12.0)
# GPU tensors: frame.displacement[..., :3] is dx, dy, height;
# frame.displacement[..., 3] is crest measure; frame.normal[..., :3] is the normal.
```

Parameters and state records are frozen values. Evaluation does not mutate its
inputs; tensor members are read-only by contract.

For live controls, keep the seeded basis and evaluate the per-wavenumber spectrum
downstream of it:

```python
from dataclasses import replace
from encino_waves import make_wave_basis, state_from_basis, preserve_phase

basis = make_wave_basis(parameters, "auto")  # once for this grid and seed
state = state_from_basis(basis, parameters)  # spectrum + dispersion on the GPU
edited = replace(parameters, wind_speed=24, depth=8)
target = state_from_basis(basis, edited)
state = preserve_phase(state, target, time=12.0)
frame = evaluate(state, 12.0)
```

`preserve_phase` returns a new value; subsequent evaluations remain independent
of call order. The saved phase segments reproduce the current edited sea. They
are not a keyframed recording of the parameter controls.

```sh
.venv/bin/python -m encino_waves doctor
.venv/bin/python -m pytest -q
.venv/bin/python tools/viewer_smoke.py
.venv/bin/python -m encino_waves benchmark --resolution 4096 --frames 30
.venv/bin/python tools/benchmark_editing.py --resolution 1024 --frames 30
.venv/bin/python tools/benchmark_foam.py --resolution 4096 --frames 30
.venv/bin/python tools/benchmark_transfer.py --frames 20  # Mac: host vs Metal
.venv/bin/python tools/foam_preview.py
```

On the NVIDIA workstation, use the corresponding virtual environment Python and
run the tests, then `benchmark --device cuda --resolution 4096`,
`still renders/cuda.png --device cuda --resolution 2048`, and
`view --device cuda`. CUDA hardware was not available during the Mac port;
its conditional tests must pass on that machine before claiming hardware parity.
The direct texture bridge currently supports Metal only; CUDA still uses the
host display transfer while both wave synthesis and foam run on its GPU.

[Implementation notes](docs/IMPLEMENTATION.md) document paper/source differences,
random realizations, FFT normalization, renderer transfer cost and verification
limits. The original headers supply a checked-in numerical fixture; the tests
compare their equations and replay their spectral coefficients through FFTW and
the available GPU. The new portable random stream is not bitwise identical to
every historical C++ standard library. Spectral snapshots can preserve an exact
realization with `save_initial_state` / `load_initial_state` in `state_io.py`.

### Background

The basic architecture of these Waves is based on the TweakWaves application
written by Chris Horvath for Tweak Films in 2001.  This, in turn, was based
on the SIGGRAPH papers and courses by [Jerry Tessendorf][tessendorf], and by
the paper ["A Simple Fluid Solver based on the FTT" by Jos Stam][simplesolver].

[tessendorf]:   http://jerrytessendorf.blogspot.com/
[simplesolver]: http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/jgt01.pdf

The TMA, JONSWAP, and Pierson Moskowitz Wave Spectra, as well as the
directional spreading functions are formulated based on the descriptions
given in "Ocean Waves: The Stochastic Approach",
by Michel K. Ochi, published by Cambridge Ocean Technology Series, 1998,2005.

This library is written as a working implementation of the paper:

> Christopher J. Horvath. 2015.   
> [Empirical directional wave spectra for computer graphics.](http://dl.acm.org/authorize?N90195)   
> In Proceedings of the 2015 Symposium on Digital Production (DigiPro '15),   
> Los Angeles, Aug. 8, 2015, pp. 29-39.    


### License 

Copyright &copy; 2015 Christopher Jon Horvath

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
