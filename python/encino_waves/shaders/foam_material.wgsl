// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Prepend foam_surface.wgsl when compiling this module.
// strength, compression threshold, compression width, crest enabled
struct Foam_material_uniforms { values: vec4f };
@group(0) @binding(0) var foam_input: texture_2d<f32>;
@group(0) @binding(1) var foam_filter: sampler;
@group(0) @binding(2) var<uniform> foam_settings: Foam_material_uniforms;
@group(0) @binding(3) var crest_input: texture_2d<f32>;

fn exclusive_crest(uv: vec2f, raw: vec4f, grain: f32, coverage: f32) -> f32 {
    let stretch = -textureSampleLevel(crest_input, foam_filter, uv, 0.0).w;
    return (1.0 - coverage) * crest_surface(stretch, raw.a, grain,
        foam_settings.values.y, foam_settings.values.z, foam_settings.values.x);
}

@vertex fn foam_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn foam_integrate(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let size = vec2f(textureDimensions(foam_input, 0));
    let pixel = floor(position.xy);
    var integrated = vec3f(0.0);
    var crest = 0.0;
    let crest_samples = max(4u, u32(ceil(max(
        f32(textureDimensions(crest_input, 0).x) / size.x,
        f32(textureDimensions(crest_input, 0).y) / size.y))));
    // The alpha detail repeats four times across the history tile. Integrate it
    // at four samples per axis, before any nonlinear coverage is mip-filtered.
    for (var y = 0u; y < 4u; y += 1u) {
        for (var x = 0u; x < 4u; x += 1u) {
            let uv = (pixel + (vec2f(f32(x), f32(y)) + 0.5) * 0.25) / size;
            let raw = textureSampleLevel(foam_input, foam_filter, uv, 0.0);
            let grain = clamp(textureSampleLevel(foam_input, foam_filter, uv * 4.0, 0.0).a, 0.0, 1.0);
            let material = foam_surface(raw.rgb, grain, foam_settings.values.x);
            integrated += vec3f(material, material.x * grain);
            if foam_settings.values.w > 0.5 && crest_samples == 4u {
                crest += exclusive_crest(uv, raw, grain, material.x);
            }
        }
    }
    // A coarse history grid must not skip narrow crests in the finer wave grid.
    // Keep the approved RGB quadrature unchanged; refine only exclusive area.
    if foam_settings.values.w > 0.5 && crest_samples > 4u {
        for (var y = 0u; y < crest_samples; y += 1u) {
            for (var x = 0u; x < crest_samples; x += 1u) {
                let uv = (pixel + (vec2f(f32(x), f32(y)) + 0.5) / f32(crest_samples)) / size;
                let raw = textureSampleLevel(foam_input, foam_filter, uv, 0.0);
                let grain = clamp(textureSampleLevel(foam_input, foam_filter, uv * 4.0, 0.0).a, 0.0, 1.0);
                let material = foam_surface(raw.rgb, grain, foam_settings.values.x);
                crest += exclusive_crest(uv, raw, grain, material.x);
            }
        }
    }
    // R: area coverage, G: coverage * freshness, B: coverage * grain. Filtering
    // these moments is linear. Divide G/B by R only after sampling the mip chain.
    // A: additional crest area outside the existing foam, integrated together
    // BEFORE filtering. Multiplying two separately filtered masks is incorrect.
    let material = integrated * (1.0 / 16.0);
    // Different quadrature at unusually coarse history resolutions can disagree
    // slightly about the old foam edge. Never exceed the remaining area.
    return vec4f(material, min(crest / f32(crest_samples * crest_samples), 1.0 - material.x));
}

@fragment fn foam_mip(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let destination_size = max(textureDimensions(foam_input, 0) / vec2u(2u), vec2u(1u));
    let uv = position.xy / vec2f(destination_size);
    return textureSampleLevel(foam_input, foam_filter, uv, 0.0);
}
