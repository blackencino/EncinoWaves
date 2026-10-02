// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Normalized GGX environment convolution for split-sum IBL (N = V = R).
// Perceptual roughness r maps to the microfacet parameter alpha = r*r.
struct Parameters {
    source: vec4f,  // width, height, final ordinary mip, sharp resample LOD
    convolution: vec4f,  // perceptual roughness, sample count, unused, unused
};
@group(0) @binding(0) var<uniform> p: Parameters;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var source_texture: texture_2d<f32>;
const pi: f32 = 3.141592653589793;

struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Vertex {
    let uv = vec2f(f32((index<<1u)&2u), f32(index&2u));
    var out: Vertex;
    out.position = vec4f(uv*2.0-1.0, 0.0, 1.0);
    out.uv = vec2f(uv.x, 1.0-uv.y);
    return out;
}

fn direction_uv(direction: vec3f) -> vec2f {
    return vec2f(atan2(direction.y, direction.x)/(2.0*pi)+0.5,
                 acos(clamp(direction.z, -1.0, 1.0))/pi);
}

fn source_area(direction: vec3f) -> f32 {
    let theta = acos(clamp(direction.z, -1.0, 1.0));
    let row = min(floor(theta*p.source.y/pi), p.source.y-1.0);
    let half_row_angle = 0.5*pi/p.source.y;
    let center_angle = (row+0.5)*pi/p.source.y;
    // Exact latitude-band area. This sine product avoids subtracting two
    // nearly equal cosines at a pole; even the polar row has nonzero area.
    return max((4.0*pi/p.source.x)*sin(center_angle)*sin(half_row_angle), 1e-12);
}

@fragment fn fragment(in: Vertex) -> @location(0) vec4f {
    let phi = (in.uv.x-0.5)*2.0*pi;
    let theta = in.uv.y*pi;
    let normal = vec3f(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
    if p.convolution.x < 1e-6 {
        return vec4f(textureSampleLevel(source_texture, source_sampler, in.uv, p.source.w).rgb, 1.0);
    }
    // A longitude-aligned frame is continuous across the panorama seam.
    let tangent = vec3f(-sin(phi), cos(phi), 0.0);
    let bitangent = cross(normal, tangent);
    let alpha = p.convolution.x*p.convolution.x;
    let alpha2 = alpha*alpha;
    let count = u32(p.convolution.y);
    var radiance = vec3f(0.0);
    var total_weight = 0.0;
    for (var i=0u; i<count; i+=1u) {
        let xi = vec2f((f32(i)+0.5)/f32(count), f32(reverseBits(i))*2.3283064365386963e-10);
        let azimuth = 2.0*pi*xi.x;
        let cos_theta = sqrt((1.0-xi.y)/max(1.0+(alpha2-1.0)*xi.y, 1e-7));
        let sin_theta = sqrt(max(0.0, 1.0-cos_theta*cos_theta));
        let half_vector = tangent*(cos(azimuth)*sin_theta)
                        + bitangent*(sin(azimuth)*sin_theta) + normal*cos_theta;
        let light = normalize(2.0*dot(normal, half_vector)*half_vector-normal);
        let no_light = max(dot(normal, light), 0.0);
        if no_light > 0.0 {
            let no_half = max(dot(normal, half_vector), 0.0);
            let denominator = no_half*no_half*(alpha2-1.0)+1.0;
            let distribution = alpha2/(pi*denominator*denominator);
            // pdf(L)=D(H)*(N.H)/(4*V.H); N=V makes this exactly D/4.
            let pdf = max(distribution*0.25, 1e-9);
            let sample_area = 1.0/(f32(count)*pdf);
            let lod = clamp(0.5*log2(sample_area/source_area(light)), 0.0, p.source.z);
            radiance += textureSampleLevel(source_texture, source_sampler, direction_uv(light), lod).rgb*no_light;
            total_weight += no_light;
        }
    }
    // Normalization preserves a constant environment at every roughness.
    return vec4f(max(radiance/max(total_weight, 1e-7), vec3f(0.0)), 1.0);
}
