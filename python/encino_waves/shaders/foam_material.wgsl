// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Prepend foam_surface.wgsl when compiling this module.
struct Foam_material_uniforms { strength: vec4f };
@group(0) @binding(0) var foam_input: texture_2d<f32>;
@group(0) @binding(1) var foam_filter: sampler;
@group(0) @binding(2) var<uniform> foam_settings: Foam_material_uniforms;

@vertex fn foam_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn foam_integrate(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let size = vec2f(textureDimensions(foam_input, 0));
    let pixel = floor(position.xy);
    var integrated = vec3f(0.0);
    // The alpha detail repeats four times across the history tile. Integrate it
    // at four samples per axis, before any nonlinear coverage is mip-filtered.
    for (var y = 0u; y < 4u; y += 1u) {
        for (var x = 0u; x < 4u; x += 1u) {
            let uv = (pixel + (vec2f(f32(x), f32(y)) + 0.5) * 0.25) / size;
            let density = textureSampleLevel(foam_input, foam_filter, uv, 0.0).rgb;
            let grain = clamp(textureSampleLevel(foam_input, foam_filter, uv * 4.0, 0.0).a, 0.0, 1.0);
            let material = foam_surface(density, grain, foam_settings.strength.x);
            integrated += vec3f(material, material.x * grain);
        }
    }
    // R: area coverage, G: coverage * freshness, B: coverage * grain. Filtering
    // these moments is linear. Divide G/B by R only after sampling the mip chain.
    return vec4f(integrated * (1.0 / 16.0), 1.0);
}

@fragment fn foam_mip(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let destination_size = max(textureDimensions(foam_input, 0) / vec2u(2u), vec2u(1u));
    let uv = position.xy / vec2f(destination_size);
    return textureSampleLevel(foam_input, foam_filter, uv, 0.0);
}
