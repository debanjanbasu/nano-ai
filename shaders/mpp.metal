// Prefill GEMMs on Metal Performance Primitives (`mpp::tensor_ops::matmul2d`): INT4_G64 weights are
// dequantized to a half scratch, then multiplied with f32 activations (f32 x f16 -> f32).
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace mpp::tensor_ops;

// 64x64 output tiles, 4 SIMD groups: the steadiest tile across the LM shapes at M >= 256 on M4 Pro.
constant constexpr int MPP_TILE = 64;
using mpp_extents = dextents<int32_t, 2>;

// INT4_G64 `[rows, columns]` -> half `[rows, columns]`, `columns % 64 == 0`. One thread per packed
// `uint` (8 columns): byte `e` holds column `2e` (low nibble) and `2e + 1` (high nibble);
// `w = code * scale + bias` with `(scale, bias) = groups[row, column / 64]`.
// Grid: ceil(total / 256) threadgroups of 256 threads, `total = rows * columns / 8`.
kernel void mpp_int4_dequant(
    device const uint *codes [[buffer(0)]], device const half2 *groups [[buffer(1)]],
    device half4 *out [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &total [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    const uint per_row = columns / 8;
    const uint row = gid / per_row, c8 = gid % per_row;
    const uint bits = codes[gid];
    const float2 sb = float2(groups[ulong(row) * (columns / 64) + c8 / 8]);
    const float4 lo = fma(float4(as_type<uchar4>(bits & 0x0F0F0F0Fu)), sb.x, sb.y);
    const float4 hi = fma(float4(as_type<uchar4>((bits >> 4) & 0x0F0F0F0Fu)), sb.x, sb.y);
    out[ulong(gid) * 2] = half4(lo.x, hi.x, lo.y, hi.y);
    out[ulong(gid) * 2 + 1] = half4(lo.z, hi.z, lo.w, hi.w);
}

constexpr constant auto mpp_desc = matmul2d_descriptor(
    MPP_TILE, MPP_TILE, static_cast<int>(dynamic_extent), false, true, false);
using mpp_op = matmul2d<mpp_desc, execution_simdgroups<4>>;

// `output[t, r] = sum_k input[t, k] * weights[r, k]` for `dims = (tokens, rows, columns, _)`.
// Grid: (ceil(rows / 64), ceil(tokens / 64)) threadgroups of 128 threads.
kernel void mpp_matmul(
    device float *input [[buffer(0)]], device half *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const int m = dims.x, n = dims.y, k = dims.z;
    tensor<device float, mpp_extents, tensor_inline> a(input, mpp_extents(k, m));
    tensor<device half, mpp_extents, tensor_inline> b(weights, mpp_extents(k, n));
    tensor<device float, mpp_extents, tensor_inline> c(output, mpp_extents(n, m));
    const int m0 = tg.y * MPP_TILE, n0 = tg.x * MPP_TILE;
    mpp_op op;
    if (m0 + MPP_TILE <= m && n0 + MPP_TILE <= n) {
        auto ta = a.slice<dynamic_extent, MPP_TILE>(0, m0);
        auto tb = b.slice<dynamic_extent, MPP_TILE>(0, n0);
        auto tc = c.slice<MPP_TILE, MPP_TILE>(n0, m0);
        op.run(ta, tb, tc);
    } else {
        auto ta = a.slice(0, m0);
        auto tb = b.slice(0, n0);
        auto tc = c.slice(n0, m0);
        op.run(ta, tb, tc);
    }
}

template <typename A, typename B, typename C>
static inline void mpp_swiglu_tile(thread A &a, thread B &gate, thread B &up, thread C &out) {
    mpp_op op;
    auto g = op.get_destination_cooperative_tensor<A, B, float>();
    auto u = op.get_destination_cooperative_tensor<A, B, float>();
    op.run(a, gate, g);
    op.run(a, up, u);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < g.get_capacity(); ++i) {
        if (g.is_valid_element(i)) g[i] = nano_silu(g[i]) * u[i];
    }
    g.store(out);
}

// `output[t, r] = silu(gate . input[t]) * (up . input[t])` with `weights` = gate rows then up rows
// (`2 * rows` half rows), `dims = (tokens, rows, columns, _)`. Grid as `mpp_matmul`.
kernel void mpp_matmul_swiglu(
    device float *input [[buffer(0)]], device half *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const int m = dims.x, n = dims.y, k = dims.z;
    tensor<device float, mpp_extents, tensor_inline> a(input, mpp_extents(k, m));
    tensor<device half, mpp_extents, tensor_inline> bg(weights, mpp_extents(k, n));
    tensor<device half, mpp_extents, tensor_inline> bu(weights + ulong(n) * k, mpp_extents(k, n));
    tensor<device float, mpp_extents, tensor_inline> c(output, mpp_extents(n, m));
    const int m0 = tg.y * MPP_TILE, n0 = tg.x * MPP_TILE;
    if (m0 + MPP_TILE <= m && n0 + MPP_TILE <= n) {
        auto ta = a.slice<dynamic_extent, MPP_TILE>(0, m0);
        auto tg_ = bg.slice<dynamic_extent, MPP_TILE>(0, n0);
        auto tu = bu.slice<dynamic_extent, MPP_TILE>(0, n0);
        auto tc = c.slice<MPP_TILE, MPP_TILE>(n0, m0);
        mpp_swiglu_tile(ta, tg_, tu, tc);
    } else {
        auto ta = a.slice(0, m0);
        auto tg_ = bg.slice(0, n0);
        auto tu = bu.slice(0, n0);
        auto tc = c.slice(n0, m0);
        mpp_swiglu_tile(ta, tg_, tu, tc);
    }
}
