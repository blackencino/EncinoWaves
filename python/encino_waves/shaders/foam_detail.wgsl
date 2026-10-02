// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Sample the static map ONCE in undisplaced ocean metres divided by its 4 m
// period. RG are resolved slopes, B has mean one, A is mean squared slope.
// None of these functions changes foam coverage or the underlying water normal.
fn foam_detail_normal(base_normal: vec3f, detail: vec4f, ocean_rotation: f32) -> vec3f {
    let normal = normalize(base_normal);
    let ocean_x = vec3f(cos(ocean_rotation), sin(ocean_rotation), 0.0);
    var tangent = ocean_x - dot(ocean_x, normal) * normal;
    // The normal normally points well above the horizon. Keep the tangent
    // frame defined for arbitrary normals without introducing a world seam.
    if dot(tangent, tangent) < 1e-6 {
        let ocean_y = vec3f(-sin(ocean_rotation), cos(ocean_rotation), 0.0);
        tangent = ocean_y - dot(ocean_y, normal) * normal;
    }
    tangent = normalize(tangent);
    let bitangent = cross(normal, tangent);
    return normalize(normal - detail.x * tangent - detail.y * bitangent);
}

fn foam_detail_variance(detail: vec4f) -> f32 {
    // Trace of the unresolved slope covariance. At level zero it is zero;
    // the fine slope energy is retained as the resolved normal smooths away.
    return max(detail.w - dot(detail.xy, detail.xy), 0.0);
}
