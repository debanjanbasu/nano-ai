//! GPU side of Neural Engine (`AppleNeuralEngine.framework`) offload for MiniCPM-V 4.7 1B:
//! - Pack F32 activation rows `[tokens, width]` into channel-major `IOSurface`s (`[width, grid_rows]`):
//!   fp16 for the language-model FFN, int8 with per-token fp16 scales (`tx[t] = max |x[t]| / 127 * 128`)
//!   for the vision MLP.
//! - Stage `INT4_G64` (`uchar` nibbles + `half2` `(scale, bias)`) matrices into fp16 `IOSurface`s
//!   (`[rows, cols]`) holding their exact values, or `INT4_G64`/F32 matrices into per-row int8 surfaces
//!   with per-row fp16 scales (`s[r] = max |w[r]| / 127 * 128`) for the vision MLP.
//! - Join the ANE's channel-major fp16 output (`y[c][t]`) back into row-major F32 activations
//!   (`out[t][c]`), checking for non-finite values.
#include "common.h"

constant constexpr float kAnePeak = 127.0f;
constant constexpr float kAneUnit = 128.0f;
constant constexpr float kAneFloor = 0x1p-14f;

static inline bool mc_ane_bad(half value) {
    return (as_type<ushort>(value) & 0x7c00) == 0x7c00;
}

struct McAneScaleParams {
    uint tokens;
    uint width;
};

// tx[t] = half(max |x[t][:]| / 127 * 128), padding rows t in [tokens, grid) get 0.
// grid: function_rows x 256 threads.
kernel void mc_ane_token_scales(
    device const float *x [[buffer(0)]],
    device half *tx [[buffer(1)]],
    constant McAneScaleParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float peaks[8];
    if (row >= p.tokens) {
        if (tid == 0) {
            tx[row] = half(0.0h);
        }
        return;
    }
    device const float4 *values = (device const float4 *)(x + ulong(row) * p.width);
    float peak = 0.0f;
    for (uint i = tid; i < p.width / 4; i += 256) {
        const float4 v = fabs(values[i]);
        peak = max(peak, max(max(v.x, v.y), max(v.z, v.w)));
    }
    peak = simd_max(peak);
    if (lane == 0) {
        peaks[group] = peak;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = 0.0f;
        for (uint g = 0; g < 8; ++g) {
            m = max(m, peaks[g]);
        }
        tx[row] = half(max(m, kAneFloor) / kAnePeak * kAneUnit);
    }
}

struct McAnePackParams {
    uint tokens;
    uint width;
    uint stride;
};

// x[t][c] -> int8 codes rint(x * 128 / tx[t]) channel-major: out[c][t]
// grid: (function_rows / 32, width / 32) x (32, 8) threads.
kernel void mc_ane_pack(
    device const float *x [[buffer(0)]],
    device const half *tx [[buffer(1)]],
    device char *out [[buffer(2)]],
    constant McAnePackParams &p [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]]
) {
    threadgroup float staged[32][33];
    const uint t0 = tile.x * 32;
    const uint c0 = tile.y * 32;
    for (uint j = pos.y; j < 32; j += 8) {
        const uint t = t0 + j;
        float value = 0.0f;
        if (t < p.tokens) {
            value = x[ulong(t) * p.width + c0 + pos.x] * (kAneUnit / float(tx[t]));
        }
        staged[j][pos.x] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = pos.y; j < 32; j += 8) {
        out[ulong(c0 + j) * p.stride + t0 + pos.x] =
            char(clamp(rint(staged[pos.x][j]), -kAnePeak, kAnePeak));
    }
}

struct McAneStageInt4Params {
    uint rows;
    uint columns;
    uint stride;
    uint scale_stride;
};

// Stage an INT4_G64 matrix [rows, columns] into per-row int8 surface `out` [rows, columns]
// and per-row fp16 scales `scales` [rows, 1].
// grid: ceil(rows / 8) x 256 threads (one SIMD group per row).
kernel void mc_ane_stage_int4(
    device const uchar *codes [[buffer(0)]],
    device const half2 *groups [[buffer(1)]],
    device char *out [[buffer(2)]],
    device half *scales [[buffer(3)]],
    constant McAneStageInt4Params &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group * 8 + simdgroup;
    if (row >= p.rows) return;
    const uint num_groups = (p.columns + 63) / 64;
    device const uchar *row_codes = codes + ulong(row) * (num_groups * 32);
    device const half2 *row_groups = groups + ulong(row) * num_groups;

    // 1. Find row peak across all group endpoints and values
    float peak = 0.0f;
    for (uint g = lane; g < num_groups; g += 32) {
        const float2 sb = float2(row_groups[g]);
        peak = max(peak, max(fabs(sb.y), fabs(fma(15.0f, sb.x, sb.y))));
    }
    peak = simd_max(peak);
    const float row_scale = max(peak, kAneFloor) / kAnePeak;
    const float inv_scale = 1.0f / row_scale;
    if (lane == 0) {
        scales[ulong(row) * p.scale_stride] = half(row_scale * kAneUnit);
    }

    // 2. Dequantize each pair of nibbles and quantize to int8 in [-127, 127]
    device char *row_out = out + ulong(row) * p.stride;
    if (p.columns % 128 == 0) {
        for (uint c = lane * 4; c < p.columns; c += 128) {
            const ushort packed = *reinterpret_cast<device const ushort *>(row_codes + c / 2);
            const float2 sb = float2(row_groups[c / 64]) * inv_scale;
            const char q0 = char(clamp(rint(fma(float(packed & 0x0Fu), sb.x, sb.y)), -kAnePeak, kAnePeak));
            const char q1 = char(clamp(rint(fma(float((packed >> 4) & 0x0Fu), sb.x, sb.y)), -kAnePeak, kAnePeak));
            const char q2 = char(clamp(rint(fma(float((packed >> 8) & 0x0Fu), sb.x, sb.y)), -kAnePeak, kAnePeak));
            const char q3 = char(clamp(rint(fma(float(packed >> 12), sb.x, sb.y)), -kAnePeak, kAnePeak));
            *reinterpret_cast<device char4 *>(row_out + c) = char4(q0, q1, q2, q3);
        }
    } else {
        for (uint c = lane * 2; c < p.columns; c += 64) {
            const uchar packed = row_codes[c / 2];
            const float2 sb = float2(row_groups[c / 64]) * inv_scale;
            const char q0 = char(clamp(rint(fma(float(packed & 0x0Fu), sb.x, sb.y)), -kAnePeak, kAnePeak));
            const char q1 = char(clamp(rint(fma(float(packed >> 4), sb.x, sb.y)), -kAnePeak, kAnePeak));
            *reinterpret_cast<device char2 *>(row_out + c) = char2(q0, q1);
        }
    }
}

// x[t][c] -> fp16 channel-major out[c][t] (`stride` in halves); padding tokens get 0.
// grid: (function_rows / 32, width / 32) x (32, 8) threads.
kernel void mc_ane_pack_half(
    device const float *x [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant McAnePackParams &p [[buffer(2)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]]
) {
    threadgroup float staged[32][33];
    const uint t0 = tile.x * 32;
    const uint c0 = tile.y * 32;
    for (uint j = pos.y; j < 32; j += 8) {
        const uint t = t0 + j;
        staged[j][pos.x] = (t < p.tokens) ? x[ulong(t) * p.width + c0 + pos.x] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = pos.y; j < 32; j += 8) {
        out[ulong(c0 + j) * p.stride + t0 + pos.x] = half(staged[pos.x][j]);
    }
}

// Stage an INT4_G64 matrix [rows, columns] (columns % 128 == 0) into fp16 surface `out`
// [rows, columns] (`stride` in halves) holding its dequantized values `q * scale + bias`.
// grid: ceil(rows / 8) x 256 threads (one SIMD group per row).
kernel void mc_ane_stage_int4_half(
    device const uchar *codes [[buffer(0)]],
    device const half2 *groups [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant McAneStageInt4Params &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group * 8 + simdgroup;
    if (row >= p.rows) return;
    const uint num_groups = p.columns / 64;
    device const uchar *row_codes = codes + ulong(row) * (num_groups * 32);
    device const half2 *row_groups = groups + ulong(row) * num_groups;
    device half *row_out = out + ulong(row) * p.stride;
    for (uint c = lane * 4; c < p.columns; c += 128) {
        const ushort packed = *reinterpret_cast<device const ushort *>(row_codes + c / 2);
        const float2 sb = float2(row_groups[c / 64]);
        const float4 q = float4(packed & 0x0Fu, (packed >> 4) & 0x0Fu, (packed >> 8) & 0x0Fu, packed >> 12);
        *reinterpret_cast<device half4 *>(row_out + c) = half4(fma(q, sb.x, sb.y));
    }
}

// Stage an F32 matrix [rows, columns] into per-row int8 surface `out` [rows, columns]
// and per-row fp16 scales `scales` [rows, 1].
// grid: ceil(rows / 8) x 256 threads (one SIMD group per row).
kernel void mc_ane_stage_f32(
    device const float *weights [[buffer(0)]],
    device char *out [[buffer(1)]],
    device half *scales [[buffer(2)]],
    constant McAneStageInt4Params &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group * 8 + simdgroup;
    if (row >= p.rows) return;
    device const float *row_w = weights + ulong(row) * p.columns;
    float peak = 0.0f;
    for (uint c = lane; c < p.columns; c += 32) {
        peak = max(peak, fabs(row_w[c]));
    }
    peak = simd_max(peak);
    const float row_scale = max(peak, kAneFloor) / kAnePeak;
    const float inv_scale = 1.0f / row_scale;
    if (lane == 0) {
        scales[ulong(row) * p.scale_stride] = half(row_scale * kAneUnit);
    }
    device char *row_out = out + ulong(row) * p.stride;
    for (uint c = lane; c < p.columns; c += 32) {
        row_out[c] = char(clamp(rint(row_w[c] * inv_scale), -kAnePeak, kAnePeak));
    }
}

// Stage an F32 bias vector [rows] into an FP16 surface `out` [rows, 1] with row stride `scale_stride`.
kernel void mc_ane_stage_bias(
    device const float *bias [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant uint2 &p [[buffer(2)]], // (rows, scale_stride)
    uint tid [[thread_position_in_grid]]
) {
    if (tid < p.x) {
        out[ulong(tid) * p.y] = half(bias[tid]);
    }
}

struct McAneJoinLinearParams {
    uint tokens;
    uint rows;
    uint ystride;
    uint mode; // 0 = store, 1 = gelu_tanh, 3 = accumulate
};

// Join ANE linear output `y[r][t] * tx[t] + bias[r]` into row-major F32 `out[t][r]`
// with epilogue mode (0 = store, 1 = gelu_tanh, 3 = accumulate).
// grid: (ceil(tokens / 32), ceil(rows / 32)) x (32, 8).
kernel void mc_ane_join_linear(
    device const half *y [[buffer(0)]],
    device const half *tx [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *out [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    constant McAneJoinLinearParams &p [[buffer(5)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]]
) {
    threadgroup float staged[32][33];
    threadgroup float token_scales[32];
    const uint t0 = tile.x * 32;
    const uint r0 = tile.y * 32;
    const uint t = t0 + pos.x;
    const bool inside_t = t < p.tokens;
    bool bad = false;
    for (uint j = pos.y; j < 32; j += 8) {
        const uint r = r0 + j;
        const half value = (r < p.rows) ? y[ulong(r) * p.ystride + t] : half(0.0h);
        bad |= (inside_t && r < p.rows) && mc_ane_bad(value);
        staged[j][pos.x] = float(value);
    }
    if (pos.y == 0) {
        const half input = (p.mode == 4) ? y[ulong(p.rows) * p.ystride + t] : tx[t];
        bad |= inside_t && mc_ane_bad(input);
        token_scales[pos.x] = float(input);
    }
    if (bad) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = pos.y; j < 32; j += 8) {
        const uint tok = t0 + j;
        const uint r = r0 + pos.x;
        if (tok < p.tokens && r < p.rows) {
            const float val = fma(staged[pos.x][j], token_scales[j], bias[r]);
            const ulong idx = ulong(tok) * p.rows + r;
            if (p.mode == 1) {
                const float c = clamp(val, -10.0f, 10.0f);
                const float k = 0.7978845608028654f;
                out[idx] = 0.5f * val * (1.0f + precise::tanh(k * (c + 0.044715f * c * c * c)));
            } else if (p.mode == 3 || p.mode == 4) {
                out[idx] += val;
            } else {
                out[idx] = val;
            }
        }
    }
}

struct McAneJoinParams {
    uint tokens;
    uint width;
    uint ystride;
    float scale;
};

// Join ANE FFN output: out[t][c] = y[c][t] * scale in F32 for t < tokens.
// grid: (ceil(tokens / 32), width / 32) x (32, 8).
kernel void mc_ane_join(
    device const half *y [[buffer(0)]],
    device float *out [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant McAneJoinParams &p [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]]
) {
    threadgroup float staged[32][33];
    const uint t0 = tile.x * 32;
    const uint c0 = tile.y * 32;
    const uint t = t0 + pos.x;
    const bool inside = t < p.tokens;
    bool bad = false;
    for (uint j = pos.y; j < 32; j += 8) {
        const half value = y[ulong(c0 + j) * p.ystride + t];
        bad |= inside && mc_ane_bad(value);
        staged[j][pos.x] = float(value);
    }
    if (bad) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = pos.y; j < 32; j += 8) {
        const uint row = t0 + j;
        if (row < p.tokens) {
            out[ulong(row) * p.width + c0 + pos.x] = staged[pos.x][j] * p.scale;
        }
    }
}
