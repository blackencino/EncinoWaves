@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vertex(@builtin(vertex_index) i: u32) -> Vertex {
    let uv = vec2f(f32((i<<1u)&2u),f32(i&2u));
    var v: Vertex;
    v.position = vec4f(uv*2.0-1.0,0.0,1.0);
    v.uv = vec2f(uv.x,1.0-uv.y);
    return v;
}
@fragment fn fragment(in: Vertex) -> @location(0) vec4f {
    return textureSample(image,image_sampler,in.uv);
}
