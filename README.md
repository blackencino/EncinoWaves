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

The water shader is ported from this project's `OceanTestShaders.cpp`, including
the scattering, Fresnel, crest shading, gamma and fog. The starting camera uses
the original framing rule and is well above the surface.

### Scenes, stills and movies

Save still also writes the physical parameters, camera, light settings and time.
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

The core has no viewer dependency and evaluates at absolute time:

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

```sh
.venv/bin/python -m encino_waves doctor
.venv/bin/python -m pytest -q
.venv/bin/python tools/viewer_smoke.py
.venv/bin/python -m encino_waves benchmark --resolution 4096 --frames 30
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
