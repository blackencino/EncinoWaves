## EncinoWaves

This branch adds a functional Python GPU implementation and a native viewer.
Ocean compute uses **PyTorch: Metal on Apple silicon, CUDA on NVIDIA**. Graphics
use **wgpu: Metal, Vulkan, or D3D12**. The original C++ source remains in `src/`.

### Run

Python 3.11 or later is required. On this Mac the environment is already installed:

```sh
./run_viewer.command
```

For a fresh macOS or Linux checkout:

```sh
bash tools/setup.sh
.venv/bin/python -m encino_waves view
```

The dependencies install from binary wheels; no C++, CUDA, Conan, or shader
compiler is needed. On Linux, graphics need a working Vulkan driver and a
desktop session. On Windows, create a Python virtual environment, run
`python -m pip install -e '.[test]'`, then `python -m encino_waves view`.
NVIDIA compute needs a CUDA-enabled PyTorch wheel and a compatible NVIDIA driver.
`doctor` reports the actual compute and graphics devices. Auto device selection
requires a GPU; CPU reference work must explicitly request `--device cpu`.

### Artist controls

The viewer exposes wind speed, fetch, depth, swell, wind direction and the four
directional spreading models. Advanced controls include resolution through
4096², patch size, spectrum, dispersion, pinch and random seed. Units are metres,
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

The **Foam & aeration** panel adds a separate, persistent RGB map: surface foam,
shallow bubbles, and deeper bubbles. Crests emit through slowly evolving fractal
breakup; existing foam diffuses, fades, and exchanges from shallow to deep.
The shader uses it for lingering surface patches and underwater scattering.
The wave spectrum and displacement are unchanged. Turn **Persistent foam** off
to return to the original crest shading.

Foam builds during playback; pausing freezes its history. **Reset foam** clears
it without restarting the waves. Large physical edits, a new patch/seed, and
time scrubbing also clear history. Rotating the ocean rotates the existing foam.
The map defaults to 512² independently of wave resolution, with options through
2048². Emission, lifetime, breakup, spreading, exchange, and underwater strength
are adjustable in the panel.

The defaults are emission **1.20**, surface lifetime **1.5 s**, breakup **1.00**,
spreading **0.56 m²/s**, shallow-to-deep exchange **0.29/s**, and underwater
bubbles **1.25**. **Fresh crests** adds a light immediate crest layer (0.15),
with separate **Crest breakup** (0.25), alongside accumulated foam.

**Wind streaks** retains a small fraction of emitted foam for longer and gathers
it into bands along the wind. New streak emission ramps up over **13.9–20.7 m/s**
(Beaufort 7–8) and **swell 0.5–0.8**; both conditions are required. The default
amount is **0.45**, with no streaks in the default, lower-swell sea. Let strong
seas play for 30–60 seconds to build the history. Set the amount to zero to
disable and clear it. Existing streaks fade when conditions ease; the usual
reset on major sea changes still applies.

**Streak controls** exposes spacing, lifetime, gathering, **Side-to-side bend**
(2 m), and **Bend time** (120 s). A broad, slowly changing crosswind deformation
moves existing streaks as they age. This is a periodic appearance approximation,
separate from the wave spectrum; it does not simulate Langmuir circulation.
The streak preview tool also saves a scene with a minute of accumulated history;
open it with `./run_viewer.command --scene renders/foam_streaks_preview.json`.

Saved scenes include adjacent `.foam_state.npz` and, for comparisons,
`.comparison_foam_state.npz` checkpoints. Keep these files beside their scene
JSON. Exports resume the saved history, including at higher wave resolutions.
Without a checkpoint, offline renders build six seconds of foam before the
first output frame; use `--foam-preroll 0` for a fresh start or `--no-foam` for
the original shading. Old scenes without foam settings keep their original look.

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
dispersion and shading are shared. Each side uses the original viewer's crest
normalization for its own wave field.

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

The water shader is ported from this project's `OceanTestShaders.cpp`, including
the scattering, Fresnel, crest shading, gamma and fog. The starting camera uses
the original framing rule and is well above the surface.

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
.venv/bin/python -m encino_waves demo renders/review.mp4 --preview
.venv/bin/python -m encino_waves demo renders/academy.mp4 --resolution 2048
```

`demo` is exactly five minutes at 24 fps: wind, fetch, directional spreading,
swell, depth and contrasting conditions, with two matched comparisons.
[The narration draft](docs/VIDEO_SCRIPT.txt) follows that sequence. The generated
movie has captions but no recorded narration or music. `--preview` makes a
34-second review with two seconds per shot.

Use `--sky`, `--width`, `--height`, `--fps`, `--time` (single shots),
`--no-captions`, or `--codec prores` as needed. ProRes should use a `.mov` output.
`--max-mbps 8` creates a compact H.264 review copy; omit it for the quality default
(CRF 16). Full-quality ocean movies need several GB of free space. Existing
outputs require `--overwrite`. Failed exports discard their temporary movie and
leave an existing completed output intact.

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
.venv/bin/python tools/foam_preview.py
.venv/bin/python tools/foam_detail_preview.py crests
.venv/bin/python tools/foam_detail_preview.py streaks
.venv/bin/python tools/benchmark_foam.py --streaks
```

On the NVIDIA workstation, use the corresponding virtual environment Python and
run the tests, then `benchmark --device cuda --resolution 4096`,
`still renders/cuda.png --device cuda --resolution 2048`, and
`view --device cuda`. CUDA hardware was not available during the Mac port;
its conditional tests must pass on that machine before claiming hardware parity.

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
