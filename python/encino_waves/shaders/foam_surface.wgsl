// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Shared by the coverage prefilter and close-fragment ocean shading.
// RGB remains surface / shallow / deep aeration density. This bounded ratio is
// a material maturity proxy, not a reconstruction of physical bubble age.
fn foam_surface(density: vec3f, grain: f32, strength: f32) -> vec2f {
    let d = max(density, vec3f(0.0));
    let older_air = max(d.y + d.z - 0.25 * d.x, 0.0);
    let fresh = d.x / max(d.x + 4.0 * older_air, 1e-8);
    let detail = clamp(grain, 0.0, 1.0);
    // Both patterns have unit mean for mean grain=.5. New sheets are connected;
    // mature foam admits deeper holes. Coverage thresholds match the legacy
    // material so this changes structure without adding another source gain.
    let modulation = mix(0.35 + 1.3 * detail, 0.85 + 0.3 * detail, fresh);
    let coverage = smoothstep(0.07, 0.5, d.x * max(strength, 0.0) * modulation);
    return vec2f(coverage, coverage * fresh);
}
