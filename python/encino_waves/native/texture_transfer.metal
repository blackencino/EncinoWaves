// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void wave_texture(device float4 const* field [[buffer(0)]],
                         texture2d<half, access::write> target [[texture(0)]],
                         uint2 const xy [[thread_position_in_grid]]) {
    if (xy.x >= target.get_width() || xy.y >= target.get_height()) return;
    target.write(half4(field[xy.y * target.get_width() + xy.x]), xy);
}

kernel void foam_texture(device float const* density [[buffer(0)]],
                         device float const* grain [[buffer(1)]],
                         texture2d<half, access::write> target [[texture(0)]],
                         uint2 const xy [[thread_position_in_grid]]) {
    uint const width = target.get_width();
    uint const height = target.get_height();
    if (xy.x >= width || xy.y >= height) return;
    uint const i = xy.y * width + xy.x;
    uint const plane = width * height;
    float3 const rgb = min(float3(density[i], density[i + plane],
                                 density[i + 2 * plane]), 60000.0f);
    // Identical to the original NumPy packing, including separate rounding of
    // multiply and add. The alpha channel is stable fine bubble breakup.
    float const alpha = clamp(0.5f + grain[i] * 7.5f, 0.0f, 1.0f);
    target.write(half4(float4(rgb, alpha)), xy);
}
