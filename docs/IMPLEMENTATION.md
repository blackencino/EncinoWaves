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

Compute includes propagation, six inverse transforms, crests and normals.
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

Setup currently runs on CPU in float64 and transfers the initial state once.
Per-frame propagation, six inverse FFTs, crest extraction, and displaced-surface
normals run on the selected GPU. The two signs of travelling waves have separate
random amplitudes and phases, as required by section 7.1.2 of the paper.

No extra wave layers, procedural surface noise, amplitude fitting, or band-pass
filters are added to the ocean. Domain size is in metres, fetch in kilometres,
wind in m/s, depth in metres. The domain is a periodic patch. Resolution changes
preserve the random variates at shared wavenumbers.

## Paper and historical implementations

The paper and checked-in source are not numerically identical in every detail.
The default `convention="paper"` uses:

- Fetch converted to metres for both spectrum and spreading.
- Full-circle normalization of directional spreading (equations 31 and 43).
- The swell coefficient **16** printed in equation 45.
- The smooth TMA depth approximation used by the original implementation and
  explicitly permitted in section 5.1.5.
- The original implementation's Pierson-Moskowitz constants and JONSWAP peak
  parameterization. JONSWAP gamma is explicitly 3.3 (equation 28).

`legacy_2015` preserves the master's fetch-unit omission, its half-circle
36-segment normalization for product spreading, and its 16.1 swell coefficient.
`houndstooth` uses corrected fetch units, the same half-circle integration,
and 16.1. These are equation compatibility options, **not a claim of bitwise
identity to every old executable**. In particular, the old `std::normal_distribution`
stream depends on the C++ standard library, and the old gamma was a seeded random
draw. The new portable wavenumber hash uses the PCG permutation from the later
functional branch with explicit integer truncation and Box-Muller variates.
The explicit gamma parameter can reproduce a known old gamma value.

Negative swell mixes towards isotropic spreading. The older Donelan implementation
has a negative sign on its isotropic target; the new implementation keeps that
mixture nonnegative instead of propagating the sign error. This difference also
applies to the compatibility options.

The default crest eigenvalue uses the actual horizontal-displacement pinch,
matching houndstooth. The 2015 master hard-coded 1.25 when computing crests.
The normal calculation is a direct functional port of `Normals.h`: periodic
central differences of displaced points.

Numerical tests check normalization, dispersion derivatives, limiting cases,
resolution stability, history independence and displacement/crest/normal outputs
against FFTW. `tests/reference/original_oracle.cpp` includes the unmodified 2015
scalar headers to generate a checked-in fixture. Tests compare the dispersion,
spectrum and spreading equations, then replay those original spectral coefficients
through GPU propagation against a double-precision FFTW reference. This checks an
actual original realization without conflating it with the new RNG.

`state_io.save_initial_state` / `load_initial_state` losslessly store the initial
complex fields, dispersion and multipliers in an NPZ with parameters. They allow
the same realization to move between CPU, Metal and CUDA; the transform tolerance
still reflects float32 arithmetic. Recreating every historical seed from parameters
alone is not established, because of the stochastic stream distinction above.

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

## Verification limits

The Mac's Metal compute, offscreen graphics, and rendered ImGui interface are
exercised locally. On September 30, 2026 the numerical suite passed 41 checks
with two CUDA-only skips. The UI check exercises camera events, scene round-trip,
comparison and all expanded panels. The desktop window was reported as occluded
by macOS during the final check; live mouse interaction still needs an unlocked,
visible desktop. The window cancels before computing when occluded.

The NVIDIA
workstation is not accessible from this session. CUDA uses the same implementation
and has a device-conditional FFTW conformance test, but its hardware performance
and renderer have not yet been measured. Run the acceptance commands in README
on the workstation rather than treating portability as a completed hardware test.
