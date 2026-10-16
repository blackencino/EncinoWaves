# School web preview

The browser app lives in `web/` and is published at
<https://blackencino.github.io/EncinoWaves/>. Its source branch is
`school/web-preview`, based on `school/gpu-showcase` at `ba782de`; the `gh-pages` branch contains only
the static website. It has no build step, server component, account, or
recording/export functions. All scripts, fonts, and sky assets are served from
the repository; there are no CDN dependencies.

## Run locally

```sh
python3 -m http.server 8000 --directory web
```

Open <http://localhost:8000/>. ES modules and assets need an HTTP server;
opening `index.html` as a `file:` URL is not supported.

## Controls

| Input | Action |
| --- | --- |
| W / D / F / S / M | Select wind / depth / fetch / swell / foam |
| T | Toggle Tessendorf / Encino Waves |
| G / P | Select historical manual gain / pinch |
| Left / right arrow | Accelerate the selected control, then glide to rest |
| E / Next sea | Cycle the four authored School examples |
| Space / Play or Pause | Pause or resume waves and foam |
| I | Toggle sea telemetry |
| C | Toggle camera telemetry |
| Tab / Controls | Open settings, sea selection, and quality |
| H | Hide or show the interface |
| Drag / Alt + left drag | Orbit the center of interest |
| Ctrl + left / Alt + middle drag | Track |
| Wheel / Shift + left / Alt + right drag | Dolly |

The opening ocean and four School compositions retain the native 1024 m patch,
parameters, cameras, and 40.735° vertical lens. Example changes take three
seconds. Reset opening ocean restores that starting composition. Touch devices
can drag to orbit and use the visible controls; a dedicated multitouch camera
interaction has not been added.

## Historical Tessendorf comparison

Press **T**, or choose **Tessendorf · early Tweak** in Controls. A direct link
can use `?model=tessendorf`; quality overrides still work. **G** selects manual
amplitude gain and **P** selects pinch, both adjusted with the arrow keys.
The selected model and manual gain survive sea changes. Fetch, swell and foam
controls are hidden in historical mode because they do not affect that model.

This mode follows the early Tweak/SimOcean implementation: the direct Phillips
wavenumber spectrum, two-sided cosine-squared wind alignment, an explicit
`AMP / domain²` coefficient scale, and finite-depth dispersion. Depth changes
wave speed without a depth-dependent spectrum envelope. There is no later
frequency-based directional tuning or short-wave damping. A sequential 48-bit
Rand48 stream and float32 polar Box–Muller reproduce its four real Gaussian
draws per nonzero half-plane bin, forming two independent complex coefficients.
This differs from the modern signed-amplitude / independent-phase sampling.
Historical random realizations depend on grid resolution; small floating-point
library differences from old builds remain possible.

The starting manual gain **100** is a presentation setting for the School's
large domain. **Use archived gain · 3.4375** recalls the toy viewer's `55/16`
setting; that viewer used a much smaller domain, so its numerical setting does
not imply matching visible wave height here. Shading is deliberately simple:
Fresnel sky reflection and a fixed water tint, with existing horizon haze. Foam,
aeration, crest detail and volume scattering are disabled. This is a numerical
comparison in the current viewer, not a reconstruction of the old renderer.

## Rendering and fidelity

`simulation.js` runs the TMA spectrum, Hasselmann spreading, native seeded PCG
wave components, and inverse FFT in WebGL 2. Six real fields are packed into
three complex transforms. Wave phase is retained during physical parameter
edits. Persistent RGB foam uses compression emission, decay, and bubble exchange;
its diffusion uses a stable stencil approximation and its breakup noise is a
separate deterministic browser implementation.

`renderer.js` ports the native projected ocean grid, water Fresnel response,
GGX sky reflections, SH9 environment lighting, foam material and crest fringe,
horizon haze, and PBR Neutral tone mapping. The six-second foam warmup happens
before the view opens. The sky is the same licensed Dutch Skies 360 photograph,
with the native 10° horizon trim. The 2048 px sky and 512 px filtered reflection
atlases encode square-root HDR radiance in RGB8; the shader restores their HDR
range. `tools/build_web_sky.py` reproduces these assets from the separately
supplied source HDR.

| Quality | Wave/foam grid | Projected mesh | Display pixel cap |
| --- | --- | --- | --- |
| Light | 256² | 240 × 144 | 1.0 million |
| Balanced | 512² | 400 × 240 | 1.8 million |
| High | 1024² | 640 × 384 | 3.2 million |

Laptops start at Balanced; iPhones start at High (1024²); other small touch
devices start at Light. This includes iPhone 17 Pro / Pro Max, iPhone 18 Pro /
Pro Max, and future capable iPhones without maintaining a model whitelist.
[Safari's device string](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/)
does not identify the hardware model, so automatic iPhone quality uses actual
playback performance. After ten seconds of visible playback, six accumulated
seconds below 24 fps can lower the automatic wave grid by one level. Fast frames
clear that history; pauses, background gaps, and quality changes reset it.
Sustained slow frames also reduce display resolution. URL and manual quality
choices disable automatic wave-grid changes.
`?quality=low`, `?quality=medium`, and `?quality=high` override the initial choice.
Changing quality rebuilds the wave grid and foam history. The native School
preview uses 2048² waves, so browser waves and foam have less fine detail,
especially at Light quality. Browser foam material integration is 256², and
tone mapping occurs before the default framebuffer's antialias resolve.

The app requires WebGL 2 and floating-point color render targets. Unsupported
devices and lost graphics contexts show a visible recovery message. Recent
iOS Safari is a target, but a physical iPhone/iPad has not been tested.

The photograph and derived lighting images have a separate license; see
[`web/assets/SKY-LICENSE.txt`](../web/assets/SKY-LICENSE.txt). Source code is
Apache-2.0. Nimbus Sans attribution and licenses are beside the font in
`web/assets/`.

## Validation

```sh
node --test web/tests/controls.test.mjs
node --test web/tests/quality.test.mjs
node --test web/tests/legacy-random.test.mjs
node web/tests/smoke.mjs --simulation
node web/tests/smoke.mjs
node web/tests/smoke.mjs --tessendorf
```

The browser tests need Playwright or `playwright-core` and an installed Chrome.
For an existing installation outside the repository, pass
`--playwright /absolute/path/to/playwright-core/index.mjs` or set
`PLAYWRIGHT_MODULE`. Set `CHROME_PATH` for a nonstandard browser location.
These are test-only dependencies. The app ships no npm dependencies.

The smoke test serves under `/EncinoWaves/` to catch GitHub Pages path errors.
It checks actual nonblank GPU output, console/network/GL errors, animation,
pause, keyboard wind changes, Maya orbit, all four scenes, each quality,
sliders, reset, and a 390 × 844 touch viewport. It also verifies the visible
error when WebGL is unavailable. Reports and screenshots go in the ignored
`renders/web-smoke/` directory (numerical reports use `renders/web-simulation/`).
`BASE_URL=https://blackencino.github.io/EncinoWaves/`
runs the UI smoke test against the deployed site; the numerical fixture is
intentionally excluded from the published site.

Initial validation used Chrome 155 on an Apple M2 Max with the ANGLE Metal
renderer, not software emulation. All browser interaction and GL checks passed.
The native-reference FFT fixture compares displacement and normals at two
times and across a phase-continuous parameter edit: displacement RMS error was
at most 0.000779 m, maximum absolute error 0.003835 m, and normal RMS error below
0.000146, including half-float texture quantization. All six control tests pass.
Actual macOS Safari also rendered the opening, example transitions, and High
quality; High showed approximately 60 fps on this machine. Chrome's narrow
touch viewport validates layout and basic controls, not the iOS graphics driver.

The historical GPU fixture covers propagation, a phase-continuous depth change,
inactive fetch/swell edits and doubled manual gain. It agrees with the native
float32 reference within 0.00217 m displacement RMS and 0.000174 normal RMS,
including half-float texture quantization. Regenerate it with
`tools/web_legacy_reference.py --model-root /path/to/native/checkout` using the
native branch containing the Phillips implementation. The dedicated historical
browser check covers the mode toggle, gain/pinch controls, archived gain,
scene changes, 1024 resolution and a narrow touch viewport.

## Publish to GitHub Pages

Commit the finished source and run from the repository root:

```sh
git push origin school/web-preview
bash tools/deploy_web.sh
```

The script archives the committed `web/` tree into a temporary checkout of
`gh-pages`, excludes `tests/`, adds `.nojekyll` and `source-commit.txt`, commits,
and pushes without force. It preserves the deployment branch's existing
history and refuses uncommitted or untracked web files. Authentication and
repository write access are required. It does not change GitHub repository
settings. In **Settings → Pages**, select **Deploy from a branch**, branch
**gh-pages**, folder **/ (root)**. GitHub may take a short time to publish.

`source-commit.txt` at the site root identifies the exact source revision.
To restore an earlier deployment, revert the desired `gh-pages` deployment
commit and push that branch normally, or restore the source and run the script
again. The deploy script can be tested safely against a local bare Git remote.
