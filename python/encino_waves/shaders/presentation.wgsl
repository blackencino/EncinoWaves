// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Resolve and filter linear HDR before display encoding.
struct Uniforms {
    eye_time:vec4f, forward_fov:vec4f, right_aspect:vec4f, up_exposure:vec4f,
};
@group(0) @binding(0) var image:texture_2d<f32>;
@group(0) @binding(1) var image_sampler:sampler;
@group(0) @binding(2) var<uniform> u:Uniforms;
struct Vertex { @builtin(position) position:vec4f, @location(0) uv:vec2f };
@vertex fn vertex(@builtin(vertex_index) index:u32) -> Vertex {
    let uv=vec2f(f32((index<<1u)&2u),f32(index&2u));
    var out:Vertex;
    out.position=vec4f(uv*2.0-1.0,0.0,1.0);
    out.uv=vec2f(uv.x,1.0-uv.y);
    return out;
}

// Khronos PBR Neutral (Apache-2.0), adapted from GLSL to WGSL.
// https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
// Keep midtone color while rolling HDR cloud breaks and glints into the display.
fn display_color(linear: vec3f) -> vec3f {
    var color=max(linear*exp2(u.up_exposure.w),vec3f(0.0));
    let floor=min(color.r,min(color.g,color.b));
    let offset=select(.04,floor-6.25*floor*floor,floor<.08);
    color=max(color-vec3f(offset),vec3f(0.0));
    let peak=max(color.r,max(color.g,color.b));
    if peak>.76 {
        let shoulder=1.0-.0576/(peak-.52);
        color*=shoulder/peak;
        color=mix(color,vec3f(shoulder),1.0-1.0/(1.0+.15*(peak-shoulder)));
    }
    return select(12.92*color,1.055*pow(color,vec3f(1.0/2.4))-.055,color>vec3f(.0031308));
}

@fragment fn fragment(in:Vertex) -> @location(0) vec4f {
    return vec4f(display_color(textureSample(image,image_sampler,in.uv).rgb),1.0);
}
