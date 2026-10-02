# GPU implementation

The Python implementation is on `academy/gpu-showcase`. The C++ library and
OpenGL viewer remain available in `src/` as the historical reference.

## API choice

Use PyTorch 2.14 for ocean compute: `mps` on Apple silicon, `cuda` on NVIDIA.
Use wgpu 0.32 for native graphics: Metal on macOS, Vulkan on Linux, D3D12 on
Windows. These are native applications, not browser applications. There is no
Conan, CUDA compiler, C++ extension, or shader compiler installation step.

PyTorch provides the same complex tensor and FFT API on both compute devices.
This implementation explicitly requests an unnormalized inverse (`norm="forward"`),
matching FFTW and cuFFT. Auto-selection requires a GPU; it does not quietly
fall back to CPU. `--device cpu` is available for reference work.

The renderer uses two RGBA16F textures. PyTorch fields remain float32; half
precision is confined to display textures. The current boundary between Torch
and wgpu is an explicit host readback and texture upload. It is **not zero-copy**.
At large resolutions this is material, so the benchmark reports it separately.
A native Metal/CUDA interop extension would reduce that cost but add build and
maintenance work. For the deadline, the wheel-only implementation is preferable.

Measured on this Mac (M2 Max, 38 GPU cores, 32 GB, macOS 15.5), 20 synchronized
frames after warmup, with no simultaneous render:

| Wave resolution | Setup | Compute median | Compute p95 | Texture readback |
| --- | --- | --- | --- | --- |
| 1024² | 0.57 s | 4.86 ms | 5.32 ms | 1.38 ms |
| 2048² | 2.07 s | 18.54 ms | 20.50 ms | 2.71 ms |
| 4096² | 8.45 s | 76.78 ms | 78.35 ms | 10.88 ms |

These measurements predate trough damping and describe the unfiltered path:
propagation, six inverse transforms, crests and normals.
These are **not viewer frame rates**: texture upload, rendering, UI, presentation
and encoding are additional. Readback is one measured transfer, not a percentile.
1024² is the interactive default; 2048² and 4096² are useful export resolutions.

References: [PyTorch MPS](https://docs.pytorch.org/docs/2.14/notes/mps.html),
[irfft2 normalization](https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft2.html),
[wgpu binary wheels and graphics backends](https://wgpu-py.readthedocs.io/en/stable/start.html).

## Functional core

`Wave_parameters` is a frozen value. `make_initial_state(parameters, device)`
returns a frozen record of spectral tensors. `evaluate(state, time)` returns
new displacement and normal tensors without changing the state. Time is absolute;
seeking backwards or exporting frames out of order gives the same result.
Tensor members are read-only by API contract; PyTorch has no const tensor type.

`make_initial_state` retains the float64 CPU reference setup. The viewer uses
`make_wave_basis` followed by `state_from_basis`: CPU setup fixes the lattice and
random variates once; the paper's spectrum, directional spreading, normalization,
and dispersion are then evaluated on the Torch GPU. This post-seed stage feeds
propagation, six inverse FFTs, crest extraction, optional trough damping and
displaced-surface normals. Trough damping adds a second batch of six inverse FFTs.
The two signs of travelling waves have separate random amplitudes and phases,
as required by section 7.1.2 of the paper.

The spectral synthesis uses the paper equations without extra wave layers or
amplitude fitting. The separate spatial trough filter is described below.
Domain size is in metres, fetch in kilometres,
wind in m/s, depth in metres. The domain is a periodic patch. Resolution changes
preserve the random variates at shared wavenumbers.

## Continuous parameter editing

Wind speed, fetch, depth and swell do not require new random variates.
They change the per-wavenumber spectrum after the seed stage. The basis caches
random complex factors, geometry, derivative multipliers and a mapping from each
mode to its distinct radial wavenumber. Isotropic quantities and directional
normalization are evaluated once per distinct radius, then gathered onto the
full spectral grid. This is an exact grouping, not a sampled/interpolated lookup.
The 64-point Gauss integral uses even symmetry to evaluate 32 paired samples;
scratch memory is bounded by chunking the radial grid. The live path uses float32
and is checked against the float64 reference across all spreading models,
equation conventions and the viewer's parameter extremes.

Wind speed edits change amplitudes but leave dispersion unchanged. Depth edits change
both the spectrum (including its frequency-to-wavenumber Jacobian) and dispersion.
Substituting a new frequency in `omega * absolute_time` would restart each mode's
phase. Instead, `preserve_phase(previous, target, edit_time)` retains the phase at
that instant and propagates subsequently with the new frequency. Its inputs are
not mutated. Editing a paused sea changes its spectrum without advancing phase.

Each outgoing dispersion segment is recorded as a small immutable `Phase_step`.
Scene JSON stores these segments and replays only radial dispersion on load,
including when exporting at a higher resolution. NPZ snapshots also store the
phase tensor losslessly. The current edited state can be evaluated at any time
in any order; seeking does not undo parameter edits or replay a parameter movie.
Both comparison panels share the same phase history.

The viewer follows numeric controls with a 0.1-second exponential response and
evaluates the actual physical parameters at every step, without crossfading wave
fields or interpolating spectral amplitudes. Once settled it stops recomputing
the spectrum. Domain, resolution and seed edits rebuild the basis in a worker;
other edits have no CPU-build debounce. Presets deliberately start a fresh state.
These are artist transitions between equilibrium spectra. The paper does not
specify transient wind growth, changing bathymetry, or a time-varying-depth fluid
solver; this extension makes no such physical claim.

Wind direction is a mesh transform, following the original `Parameters.h`
contract: synthesize wind along +X and place the field externally. Positive
angles rotate +X towards +Y about world Z. The first Python viewer instead
subtracted this angle inside directional spreading. That reweighted different
random samples on the fixed Fourier lattice and changed the particular sea;
it was not a rigid rotation and was not an FFTW-versus-Torch difference.

`Wave_frame` maps stay in ocean space. For a rotation R, the renderer samples at
`q = inverse(R) * world_xy`, rotates horizontal displacement and normals by R,
and leaves height and crest scalars unchanged. The view-adaptive grid remains
camera-aligned, covering the view at every ocean orientation. Camera and sky
stay fixed when the artist rotates the ocean. Direction-only edits reuse all
spectral tensors, phase history and shading statistics; while paused they also
reuse the FFT results and GPU textures. Both comparison panels and exports use
the same transform. A core consumer using `texture_arrays` must apply the frame's
`parameters.wind_direction` when placing its mesh or texture reference frame.

New NPZ snapshots use version 3 to record this placement convention. Versions
1 and 2 already baked direction into their stored coefficients: loading them
preserves those fields exactly and applies no additional mesh rotation. Scene
JSON reconstructs the spectrum and uses the corrected rotation behavior.

The half-angle spreading factor uses the equivalent sine expression so it is
exactly zero at +/- pi. This avoids amplifying the rounding error in float32
`cos(pi/2)` when the spreading exponent is small, and matches the float64 path.

On this M2 Max, synchronized warm edits of wind and depth together, with
rendering omitted (30 samples at 1024/2048, 20 at 4096):

| Resolution | One-time basis | Spectrum + phase edit | Propagation, FFTs, normals |
| --- | --- | --- | --- |
| 1024² | 0.14 s | 2.30 ms median / 2.45 ms p95 | 4.77 ms median |
| 2048² | 0.38 s | 6.66 ms median / 7.12 ms p95 | 18.75 ms median |
| 4096² | 1.51 s | 27.01 ms median / 27.72 ms p95 | 77.90 ms median |

These unfiltered measurements exclude trough damping, texture transfers,
graphics and UI. Run
`python tools/benchmark_editing.py --resolution 1024 --device mps` (or `cuda`)
to measure the editing path separately from the static reference setup above.
CUDA uses the same tensor operations; NVIDIA hardware remains untested here.

## Paper and historical implementations

The paper and original 2015 source are not numerically identical in every detail.
Hasselmann is the default directional spreading model, as in the original C++
parameters. Tessendorf mode still selects positive cosine-squared spreading.
The default `convention="paper"` uses:

- Fetch converted to metres for both spectrum and spreading.
- Full-circle normalization of directional spreading (equations 31 and 43).
- The swell coefficient **16** printed in equation 45.
- The smooth TMA depth approximation used by the original implementation and
  explicitly permitted in section 5.1.5.
- The original implementation's Pierson-Moskowitz constants and JONSWAP peak
  parameterization. JONSWAP gamma is explicitly 3.3 (equation 28).

The maintained C++ headers now also use the corrected fetch units, full-circle
directional normalization and paper swell coefficient 16. Donelan-Banner and
cosine-squared negative swell mix toward positive isotropic density. The product
normalization uses 64-point Gauss-Legendre quadrature, with the analytic Donelan
integral when swell is nonpositive. The old seeded JONSWAP gamma and RNG remain
in C++; supplying that gamma to the Python reference isolates equation checks
from the random-stream differences.

The 2025 Torch comparison (`234835a`, on `feature/simplified_debug`) rewrote the
smooth TMA depth factor as `sigmoid(3.6 * (wh - 1.125))`, where
`wh = omega * sqrt(depth / gravity)`. The CPU reference and GPU implementation
both use that form. It is algebraically identical to the original
`0.5 + 0.5*tanh(1.8 * (wh - 1.125))`, avoiding cancellation toward the low end of
the curve. The factor's physical input bounds limit that cancellation, so this
is a modest numerical improvement, not a change in the spectrum model. This
evaluation is also backported to C++, along with log-domain spectrum evaluation
and an explicit zero-energy DC mode.

The same historical commit replaced the direct Hasselmann gamma-function ratio
with log-gamma evaluation to prevent overflowing intermediate factors. The
current GPU setup already combines this with the gamma duplication identity:
`log(Q) = lgamma(s+1) - lgamma(s+0.5) - log(2*sqrt(pi))`. This avoids separately
forming the large power of two and gamma factors; it is particularly useful at
large swell. The dispersion derivative also avoids overflowing `cosh(h*k)` by
using `sech²(h*k) = 1 - tanh²(h*k)`. Both corrections are backported to C++.

`legacy_2015` preserves the master's fetch-unit omission, its half-circle
36-segment normalization for product spreading, and its 16.1 swell coefficient.
`houndstooth` uses corrected fetch units, the same half-circle integration,
and 16.1. These are equation compatibility options, **not a claim of bitwise
identity to every old executable**. In particular, the old `std::normal_distribution`
stream depends on the C++ standard library, and the old gamma was a seeded random
draw. The new portable wavenumber hash uses the PCG permutation from the later
functional branch with explicit integer truncation and Box-Muller variates.
The explicit gamma parameter can reproduce a known old gamma value.

Negative swell mixes towards isotropic spreading. The original Donelan implementation
has a negative sign on its isotropic target; the new implementation keeps that
mixture nonnegative instead of propagating the sign error. This difference also
applies to the compatibility options.

The default crest eigenvalue uses the actual horizontal-displacement pinch,
matching houndstooth. The 2015 master hard-coded 1.25 when computing crests.
The normal calculation is a direct functional port of `Normals.h`: periodic
central differences of displaced points.

Numerical tests check normalization, dispersion derivatives, limiting cases,
resolution stability, history independence and displacement/crest/normal outputs
against FFTW. `tests/reference/original_oracle.cpp` uses the unmodified scalar
headers extracted from pinned commit `b7db469` to generate the historical fixture.
`test_cpp_numerics.py` separately compiles the maintained C++ headers and checks
float/double spectra, dispersion, nonnegative spreading and angular energy.
Tests compare the dispersion,
spectrum and spreading equations, then replay those original spectral coefficients
through GPU propagation against a double-precision FFTW reference. This checks an
actual original realization without conflating it with the new RNG.

`state_io.save_initial_state` / `load_initial_state` losslessly store the initial
complex fields, dispersion and multipliers in an NPZ with parameters. They allow
the same realization to move between CPU, Metal and CUDA; the transform tolerance
still reflects float32 arithmetic. Recreating every historical seed from parameters
alone is not established, because of the stochastic stream distinction above.

## Spatial trough damping

`trough.py` ports the active trough-damping path in the original
`src/EncinoWaves/Propagation.h` and `Filter.h`. It is enabled at 0.5 by default;
both the GPU implementation and maintained C++ path allow amounts from 0 to 1.
This is a separate production appearance filter, not an alteration of the
paper's spectrum or a claim that the spatially modified result has the same PSD.

The original inverted smooth wavelength band removes detail from 1 to 4 metres,
with transitions from 0 to 1 and 4 to 6 metres. A second batch of six inverse
FFTs produces filtered height, horizontal displacement and displacement
derivatives. Its negative minimum eigenvalue supplies the spatial guide, using
the original fixed positive pinch of 1.25 even when displayed pinch is zero.
With `z = (guide - mean(guide)) / (2.2 * std(guide))`, the retention is
`1 - amount + amount * smoothstep(0, 1, z)`. Height and horizontal displacement
are `filtered + retention * (original - filtered)`. The selected detail retains
between 50% and 100% of its amplitude at the 0.5 default, and between 0% and 100%
at maximum strength (1).

As in the original active C++ path, the emission crest map is preserved. Normals
are computed from the final displaced points, so they include the spatial blend's
gradient. Flat guide fields bypass damping rather than divide by zero. The GPU
statistics count each periodic texel once; the old C++ statistics also counted
the duplicated border. Turning the filter off skips the extra FFTs entirely.
Live amount/band edits reuse the spectral tensors and phase history. Parameters
are explicit values saved with scenes and spectral snapshots; older saved oceans
without trough settings load with damping off for surface compatibility.

Tests compare the spatial result to an independent double-precision FFTW
reference and verify attenuation bounds, mask orientation, final-surface normals,
flat water, unchanged crest emission, live edits and snapshot replay on CPU/Metal.
`python tools/trough_preview.py` makes an ignored local on/off movie and stills,
sharing one foam history and the installed HDR between both views.

On this M2 Max, 10 synchronized propagation samples after three warmups,
excluding texture transfers, foam, rendering and UI:

| Resolution | Damping off, median | Damping 0.5, median |
| --- | --- | --- |
| 1024² | 4.80 ms | 6.45 ms |
| 2048² | 18.58 ms | 25.67 ms |
| 4096² | 76.84 ms | 110.46 ms |

Run `python tools/benchmark_trough.py --device mps` (or `cuda`) to compare the
two paths. These are compute times, not complete viewer frame rates.

## Two-domain repetition reduction: next experiment

This remains a design proposal, deliberately separate from the trough port.
Two independent realizations with different physical periods can reduce visible
repetition. FFT array dimensions can stay powers of two: it is the domain lengths
in metres that need a poorly commensurate ratio. An irrational ratio has no exact
shared period in ideal arithmetic; a co-prime integer pair has a long least common
multiple. Neither fact alone guarantees that recognizable swell stops repeating.

Use smooth nonnegative **power** windows `W0(k) + W1(k) = 1` over the covered
wavenumbers, allocating all power to the supported field outside their overlap.
Evaluate the same physical spectrum on each lattice, including that lattice's
own `(2*pi/L)^2` integration area, and multiply spectral amplitudes by `sqrt(Wi)`.
Use independent random streams so cross-covariance vanishes in expectation.
The sum then targets the original spectral power, subject to the ordinary
finite-lattice sampling and band-limit error; it does not preserve the old
individual realization. Check band-integrated energy across seeds, significant
wave height and directional moments before judging rendered comparisons.

Sample both fields in world coordinates and combine displacements/derivatives
before computing nonlinear crest and normal quantities. Baking the sum back into
one existing periodic map would reintroduce its short period. A single periodic
foam map cannot represent the full combined crest history exactly, so foam
storage and the placement of trough damping need explicit treatment too.

Spectrum partitioning into displacement cascades is also described in
[Arc Blanc, section 3.3](https://jcgt.org/published/0014/01/05/paper-lowres.pdf).
Its hard bands and domain sizes differ from the proposed overlapping,
poorly commensurate two-domain experiment above.

## Original viewer baseline

The water shader in `shaders/ocean.wgsl` is ported from
`src/EncinoWaves/Tests/OceanTestShaders.cpp`, specifically
`g_fragmentShaderTextureSkyBase`. It preserves the original 1.3 refractive index,
Fresnel transmission scaling, Pacific scattering and extinction constants,
Henyey-Greenstein phase, sun and opposite sky sample, specular exponent,
normalized crest shading, layered fog and 2.2 display gamma. The sky gain of
2 matches `OceanTestTextureSky.cpp`. A denominator guard handles horizontal fog
rays; an algebraically equivalent fog evaluation avoids overflow from high
cameras. Exposure is explicit and defaults to zero. There is no filmic tonemapper.
HDR pixels that exceed half-float range are stored with a uniform scale that is
restored in the shader. Light extraction selects the upper hemisphere; the old
loader searched all pixels. A real sky panorama's sun remains in that hemisphere.
The supplied Dutch Skies 360 Autumn Pack 01, 03a is installed under the ignored
`assets/local/` directory. Discovery selects its 4000×2000 `_Ref.hdr`, preserving
linear HDR values and full panorama resolution, instead of the 360×180 blurred
`_Env.hdr`. The original gain, display gamma, camera and water shader are retained.
The native viewer and offline renderer share this selection; saved scenes store
the resolved HDR path. The original files and sIBL descriptor remain together.
Scene JSON includes the height and crest statistics from the original shader's
initial normalization, for each comparison side. Reopening a saved scene is
tested to reproduce identical pixels at the same render settings.

A perspective grid replaces the old fixed mesh / geometry-shader tiling. Texture
mipmaps limit distant geometric aliasing. The underlying displacement is still
one repeated physical ocean patch. Normals are interpolated from a full-resolution
texture rather than a fixed triangle vertex normal. These rendering differences
are separate from the numerical core.

`camera.py` ports the Z-up center-of-interest model in `GLCamera.cpp`. Tumble and
dolly preserve the pivot; track moves eye and pivot together. The original
400-degree tumble and exponential dolly sensitivities are retained. Initial
framing and the F key use the original `ViewScene::getBounds` / `GLCamera::frame`
rule, so the initial camera is well above the surface. No bare left-drag fly camera.

The large-domain storm shot in the movie uses a separately staged camera at the
default 512 m patch framing distance. Applying the 1800 m domain's full framing
distance made the original fog obscure the shot.

## Persistent foam and aeration

`foam.py` is a separate functional appearance simulation. `Foam_state` contains
explicit previous density and time; the synthesized wave state remains unchanged
and history-independent. RGB stores nonnegative surface, shallow, and deep
aeration densities in the ocean's undisplaced periodic coordinates. The mesh
displacement carries this map along with the surface, and wind direction rotates
the complete field. This version has no additional current/advection solver.

Emission uses the original normalized negative minimum-eigenvalue crest map,
with the same default 0.5–1.1 smooth threshold as the original shader. Calibration
is held with the foam history. Source precision matches the half-float display
map, avoiding amplified sub-texel residuals in nearly flat water. The model does
not infer a measured whitecap
fraction from wind speed; these remain artist controls. Four octaves of seeded,
spatially periodic quintic noise evolve slowly in time and modulate emission.
Thresholding precedes downsampling, preserving the contribution of narrow crests
when a 4096² wave field drives a smaller foam map.

Before each emission, three batched real FFTs and inverse FFTs solve periodic
discrete diffusion using `exp(-D * dt * L)`. This preserves constant fields and
wraps both edges, without a stencil stability restriction for small patches.
Surface/shallow/deep diffusivities use a 1:3:7 ratio. Each channel fades with its
own half-life; shallow-to-deep exchange and decay use the exact local linear
solution. Diffusion and exchange are split operators. New aeration is deposited
80% at the surface and 20% shallow; the deeper map fills through exchange.
Emission is integrated over the frame interval, so results have timestep error
when the source varies, rather than claiming frame-rate-independent exactness.

The default 512² density state occupies 3 MiB in float32; its noise basis and
FFT scratch are additional. The display boundary uploads RGBA16F, with RGB as
density and alpha as stable, mip-filtered bubble detail. Subsurface density
changes the existing shader's scattering, extinction and phase coefficients;
surface density supplies a separate foam coverage layer. Dilute remnants become
transparent rather than whitening the whole sea. Disabling persistent foam
uses the original crest shading, with the original water constants.

The default surface half-life
is 1.5 s, diffusion 0.56 m²/s, exchange 0.29/s, and underwater strength 1.25;
emission remains 1.2/s with full fractal breakup.

The wind-streak experiment and its extra immediate crest overlay have been
removed. The emission, aging, breakup and foam shader are restored to `d390aea`,
with the artist-selected defaults above. At identical wave fields and controls,
the restored RGB history matches that implementation exactly on Metal. The
four emission octaves and separate fine shader grain retain their original
scales and weights. Version 1 and 2 checkpoints still load: only the RGB density
is restored, and known retired appearance settings and windrow arrays are
ignored. The experiment remains available in Git history (`cafc4cf`).

On this M2 Max, synchronized foam update work measured approximately 2.2 ms with
1024² waves and a 512² foam map, and 6.9 ms with 4096² waves and the same foam map
(7.1 ms p95, 20 samples for the latter). These exclude texture readback/upload
and rendering. The higher wave resolution costs more in crest-source filtering;
the history grid itself stays 512². `tools/benchmark_foam.py` measures this stage
separately; `tools/foam_preview.py` renders an eight-second comparison.

The viewer resets on time scrubbing, backward/large time jumps, domain/seed/model
changes, or cumulative large physical changes relative to the last reset:
wind speed ratio >1.35, fetch/depth ratio >2, swell change >0.35, or pinch change
>0.3. It also provides an explicit Reset foam button. Direction, camera, sky and
wave resolution preserve history; changing foam resolution or noise scale resets
it. Paused frames do not decay or deposit again. Each comparison sea has its own
history under matched controls.

Scene JSON references lossless float32 NPZ checkpoints beside the JSON, including
time, calibration, foam controls and the reference sea. Noise is regenerated from
the saved seed. Checkpoints resume in still/movie export, even at a higher wave
resolution, because the foam grid is independent. Shots without history use a
six-second preroll at 30 Hz (`--foam-preroll`); low-FPS exports subdivide long
output intervals. A changed output time starts new preroll rather than presenting
the old checkpoint as the correct history at that time. Existing old scene JSON
defaults to the original shading until foam is explicitly enabled.

The local Tweak tree contains a precursor in
`~/dvlp/src/bin/water/moveTxt2/main.cpp`: wrapped history advection, diffusion,
then emission composited with decay. `fftCrest/main.cpp` supplies related crest
source filtering. The exact three-channel depth-exchange implementation was not
located. This implementation follows the requested process using current GPU
operations; it does not incorporate the old library or add build dependencies.
For published context, [Tessendorf, Reinhardt and Gao, Whitecap Phenomenology for
Ocean Surface Simulation](https://jtessen.people.clemson.edu/gilligan/html/whitecap_fraction.pdf)
describes minimum-eigenvalue emission and persistent, decaying whitecap textures.

## Verification limits

The Mac's Metal compute, offscreen graphics, and rendered ImGui interface are
exercised locally. On October 1, 2026 the numerical suite passed 88 checks
with twenty-two CUDA-only skips. The UI check exercises camera events, continuous
wind/depth editing, phase-preserving scene round-trip, comparison and all expanded
panels, including with the supplied Dutch Skies HDR. Direction edits preserve
the exact spectral coefficients across all four spreading models. The UI test
also forbids spectrum and FFT evaluation during a paused direction edit. Rotating
mesh, camera and environment together reproduces the original rendered image
within 0.003 mean 8-bit channel levels for both comparison seas. Saved rotated
scenes round-trip pixel-exactly and export through the movie renderer.
Populated RGB foam checkpoints also round-trip pixel-exactly; reset leaves the
wave frame unchanged. Foam tests check periodic mass conservation and translation,
seam diffusion, separate decay, shallow/deep exchange, retention after emission
stops, pause behavior, reset conditions and replay at higher wave resolutions.
The desktop window was reported as occluded
by macOS during the final check; live mouse interaction still needs an unlocked,
visible desktop. The window cancels before computing when occluded.

The NVIDIA
workstation is not accessible from this session. CUDA uses the same implementation
and has a device-conditional FFTW conformance test, but its hardware performance
and renderer have not yet been measured. Run the acceptance commands in README
on the workstation rather than treating portability as a completed hardware test.
