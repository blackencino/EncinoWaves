# Tessendorf / early Tweak baseline

Press **T** in either native viewer to switch between Encino Waves and the
early baseline. In School, **G** selects Phillips amplitude gain and **P**
selects pinching; left/right arrows adjust the selected control. **W** and **D**
retain wind and depth control. Each control key is a toggle: its upper-left
indicator stays visible until that key is pressed again or another control is
selected. **I** toggles a compact panel of all sea parameters in the same corner,
highlighting the selected control. In the historical mode this includes gain
and pinch; fetch, swell and foam controls are inactive.

The baseline reconstructs the original course equations with the surviving
early Tweak implementation conventions. It is not a claim that every surviving
2004 production setting was identical to the course baseline. The archived
production sources already contain directional adjustments; those adjustments
are deliberately absent here.

- Phillips density in wave-number space: `exp(-1/(k L)^2) cos(theta)^2 / k^4`,
  with `L = U^2/g`. Both directional lobes are present. No directional
  normalization, fetch dependence, swell narrowing or short-wave cutoff is added.
- Depth affects only `omega = sqrt(g k tanh(k depth))`. It does not attenuate
  the spectrum. The modern spatial trough filter is also off.
- The archived Rand48 generator and polar Box–Muller Gaussian draws create
  complex initial coefficients. This preserves the early complex Gaussian
  construction rather than the modern signed normal amplitude and independent
  uniform phase construction. The surviving sources draw four independent
  Gaussian values for the two complex coefficients; they do not establish a
  shared-positive/negative-random-value bug.
- Manual height gain follows SimOcean's `AMP / domain^2` scaling of the unnormalized
  inverse transform; horizontal displacement has its own independent `PINCH`
  multiplier. School opens at **150** height gain and **75** pinch: 50% taller than
  the prior 100-gain opening, deliberately a little overdriven for the live tuning
  demonstration. Subsequent artist gain choices survive model switches and examples.
  This is an explicit artist setting, not a fit to the modern ocean. The setup button restores the archived
  SimOcean default **55/16 = 3.4375**, which is very small on the present large
  metre-scale patch. Gain remains visible and editable. No automatic wave-height
  matching is performed.
- Shading is deliberately simple: Fresnel sky reflection plus a constant water
  tint. Foam, crest fringe and underwater scattering are bypassed. The current
  sky dome, exposure and camera are retained; this is not a pixel-identical
  reconstruction of an old OpenGL renderer.

Source basis: Tessendorf's 2001 SIGGRAPH course notes, equations 23, 25 and 26;
archived `SimOcean.cpp` (Phillips setup, amplitude scaling and Gaussian draws);
`DeepOcean2D.cpp`, CVS revision 1.6 dated March 1, 2004; and the archived
`TwkMathRandom.h` / `TwkMathRand48.cpp` generator. Original files are unchanged.
The independent generator is verified against values from the archived C++.

The existing modern parameters and appearance settings survive the mode switch.
The same modern random stream remains in use outside the baseline. Historical
seeding is sequential and therefore intentionally depends on grid resolution.
