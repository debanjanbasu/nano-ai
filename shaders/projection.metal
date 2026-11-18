// Projections for MiniCPM-V 4.7 1B:
// 1. Per-row int8 (`int8` + `f32` row scales): vector (1..2 rows), wide 8x8 simdgroup matrix (3..8 rows),
//    concatenated single-row matvec, and fused SwiGLU matvec.
// 2. Group-64 int4 (`uchar` packed nibbles + `half2` (scale, bias)): vector (1 row), wide (2..8 rows, 1 or 2 tiles),
//    and fused SwiGLU matvec.
// 3. Ternary PTQ1_0 (128 trits + FP16 scale in 28 bytes): half-FMA prefix decoder matvec, fused SwiGLU,
//    small-batch (2..4 rows) lane-pair shuffle kernel, and 5..8 row wide simdgroup-matrix kernel,
//    plus 512-block signed FWHT rotation kernels.
// 4. BF16 matvec for the 16-row GDN alpha/beta projections and embedding gather.
#include "common.h"

// ============================================================================
// 1. Per-row INT8 Kernels
// ============================================================================

template<uint R>
struct Int8Rows {
    device const char *rows[R];
    float scales[R];
};

template<uint R>
static inline Int8Rows<R> int8_rows(
    device const char *weights, device const float *scales,
    uint first_row, uint rows, uint columns
) {
    Int8Rows<R> result;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < R; ++r) {
        const uint row = min(first_row + r, rows - 1);
        result.rows[r] = weights + ulong(row) * columns;
        result.scales[r] = scales[row];
    }
    return result;
}

template<uint R, uint T>
static inline void int8_dot(
    thread const Int8Rows<R> &w, device const float *input, uint columns, uint lane,
    thread float *sums
) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < R * T; ++i) sums[i] = 0.0f;
    for (uint c = lane * 16; c < columns; c += 512) {
        float4 x[T][4];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            device const float4 *p =
                reinterpret_cast<device const float4 *>(input + ulong(t) * columns + c);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) x[t][q] = p[q];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            const uint4 bits = *reinterpret_cast<device const uint4 *>(w.rows[r] + c);
            const float4 v[4] = {
                float4(as_type<char4>(bits.x)), float4(as_type<char4>(bits.y)),
                float4(as_type<char4>(bits.z)), float4(as_type<char4>(bits.w))};
            #pragma clang loop unroll(full)
            for (uint t = 0; t < T; ++t) {
                float s = sums[t * R + r];
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4; ++q) s += dot(v[q], x[t][q]);
                sums[t * R + r] = s;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            sums[t * R + r] = simd_sum(sums[t * R + r]) * w.scales[r];
        }
    }
}

template<uint R, uint T>
static inline void int8_matmul_rows(
    device const char *weights, device const float *scales, device const float *input,
    device float *output, uint rows, uint columns, uint group, uint lane
) {
    const uint first_row = group * R;
    const Int8Rows<R> w = int8_rows<R>(weights, scales, first_row, rows, columns);
    float sums[R * T];
    int8_dot<R, T>(w, input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            if (lane == r && first_row + r < rows) {
                output[ulong(t) * rows + first_row + r] = sums[t * R + r];
            }
        }
    }
}

kernel void int8_matvec(
    device const char *weights [[buffer(0)]], device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    int8_matmul_rows<4, 1>(weights, scales, input, output, rows, columns, group, lane);
}

kernel void int8_matmul_2(
    device const char *weights [[buffer(0)]], device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    int8_matmul_rows<4, 2>(weights, scales, input, output, rows, columns, group, lane);
}

kernel void int8_matmul_wide(
    device const char *weights [[buffer(0)]], device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint split = 8;
    threadgroup float shared[split * 64];
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = group * 8;
    device const float *x_row = input + ulong(fm % tokens) * columns + 16 * fn;
    device const char *w_rows[2];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        w_rows[r] = weights + ulong(min(first_row + fn + r, rows - 1)) * columns + 16 * fm;
    }
    simdgroup_float8x8 sums = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint start = simd * 128; start < columns; start += split * 128) {
        float w[2][16];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            const uint4 bits = *reinterpret_cast<device const uint4 *>(w_rows[r] + start);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) {
                const float4 v = float4(as_type<char4>(bits[q]));
                #pragma clang loop unroll(full)
                for (uint e = 0; e < 4; ++e) w[r][4 * q + e] = v[e];
            }
        }
        float xa[16], xb[16];
        #pragma clang loop unroll(full)
        for (uint q = 0; q < 4; ++q) {
            const float4 a = *reinterpret_cast<device const float4 *>(x_row + start + 4 * q);
            const float4 b = *reinterpret_cast<device const float4 *>(x_row + start + 16 + 4 * q);
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 4; ++e) {
                xa[4 * q + e] = a[e];
                xb[4 * q + e] = b[e];
            }
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 16; ++j) {
            simdgroup_float8x8 x;
            x.thread_elements()[0] = xa[j];
            x.thread_elements()[1] = xb[j];
            simdgroup_float8x8 b;
            b.thread_elements()[0] = w[0][j];
            b.thread_elements()[1] = w[1][j];
            simdgroup_multiply_accumulate(sums, x, b, sums);
        }
    }
    shared[simd * 64 + lane * 2] = sums.thread_elements()[0];
    shared[simd * 64 + lane * 2 + 1] = sums.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0 || fm >= tokens) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint s = 0; s < split; ++s) total += shared[s * 64 + lane * 2 + r];
        const uint row = first_row + fn + r;
        if (row < rows) output[ulong(fm) * rows + row] = total * scales[row];
    }
}

kernel void int8_matvec_swiglu(
    device const char *gate [[buffer(0)]], device const float *gate_scales [[buffer(1)]],
    device const char *up [[buffer(2)]], device const float *up_scales [[buffer(3)]],
    device const float *input [[buffer(4)]], device float *output [[buffer(5)]],
    constant uint &rows [[buffer(6)]], constant uint &columns [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = group * 2;
    const Int8Rows<2> g = int8_rows<2>(gate, gate_scales, first_row, rows, columns);
    const Int8Rows<2> u = int8_rows<2>(up, up_scales, first_row, rows, columns);
    const Int8Rows<4> both = {
        {g.rows[0], g.rows[1], u.rows[0], u.rows[1]},
        {g.scales[0], g.scales[1], u.scales[0], u.scales[1]}};
    float sums[4];
    int8_dot<4, 1>(both, input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        if (lane == r && first_row + r < rows) {
            output[first_row + r] = nano_silu(sums[r]) * sums[2 + r];
        }
    }
}

// ============================================================================
// 2. Group-64 INT4 Kernels
// ============================================================================

// Small-batch (1..2 tokens) INT4_G64 matmul: one SIMD group owns `R` rows. Each lane covers 8
// consecutive columns (one `uint` of codes per row) per 256-column step, so every weight byte is
// read once for all tokens and the activations stay in 9T registers. Token rows past `tokens`
// re-read the last row and are never stored. 3+ tokens use the `simdgroup_matrix` kernels below,
// whose 8-token tiles are compute-bound but cheaper than 3+ scalar FMA streams on Apple GPUs.
constant constexpr uint MV_ROWS = 4;

template<uint R>
struct Int4Rows {
    device const uint *codes[R];
    device const half2 *groups[R];
};

template<uint R>
static inline void int4_rows_into(
    thread Int4Rows<R> &m, uint slot, device const uchar *weights, device const half2 *groups,
    uint rows, uint columns, uint first_row, uint count
) {
    const uint row_groups = (columns + 63) / 64;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < count; ++r) {
        const uint row = min(first_row + r, rows - 1);
        m.codes[slot + r] = reinterpret_cast<device const uint *>(weights + ulong(row) * row_groups * 32);
        m.groups[slot + r] = groups + ulong(row) * row_groups;
    }
}

template<uint T>
struct MvInput {
    device const float *rows[T];
};

template<uint T>
static inline MvInput<T> mv_input(device const float *input, uint columns, uint tokens) {
    MvInput<T> x;
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) x.rows[t] = input + ulong(min(t, tokens - 1)) * columns;
    return x;
}

template<uint R, uint T>
static inline void int4_mv_core(
    thread const Int4Rows<R> &m, thread const MvInput<T> &x, uint columns, uint lane,
    thread float *acc
) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < R * T; ++i) acc[i] = 0.0f;
    for (uint c = lane * 8; c < columns; c += 256) {
        float4 xe[T], xo[T];
        float xs[T];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            const float4 a = *reinterpret_cast<device const float4 *>(x.rows[t] + c);
            const float4 b = *reinterpret_cast<device const float4 *>(x.rows[t] + c + 4);
            xe[t] = float4(a.xz, b.xz);
            xo[t] = float4(a.yw, b.yw);
            const float4 s = a + b;
            xs[t] = (s.x + s.y) + (s.z + s.w);
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            const uint bits = m.codes[r][c / 8];
            const float2 sb = float2(m.groups[r][c / 64]);
            const float4 lo = float4(as_type<uchar4>(bits & 0x0F0F0F0Fu));
            const float4 hi = float4(as_type<uchar4>((bits >> 4) & 0x0F0F0F0Fu));
            #pragma clang loop unroll(full)
            for (uint t = 0; t < T; ++t) {
                const float d = dot(lo, xe[t]) + dot(hi, xo[t]);
                acc[r * T + t] = fma(sb.x, d, fma(sb.y, xs[t], acc[r * T + t]));
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < R * T; ++i) acc[i] = simd_sum(acc[i]);
}

// BF16 rows (`[rows, columns]` raw `ushort`) with the same lane layout: one `uint4` (8 values) per row.
template<uint R, uint T>
static inline void bf16_mv_core(
    device const ushort *weights, uint rows, uint first_row, thread const MvInput<T> &x, uint columns,
    uint lane, thread float *acc
) {
    device const uint4 *w[R];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < R; ++r) {
        w[r] = reinterpret_cast<device const uint4 *>(weights + ulong(min(first_row + r, rows - 1)) * columns);
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < R * T; ++i) acc[i] = 0.0f;
    for (uint c = lane * 8; c < columns; c += 256) {
        float4 xa[T], xb[T];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            xa[t] = *reinterpret_cast<device const float4 *>(x.rows[t] + c);
            xb[t] = *reinterpret_cast<device const float4 *>(x.rows[t] + c + 4);
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            const uint4 bits = w[r][c / 8];
            const float4 wa = float4(as_type<float>(bits.x << 16), as_type<float>(bits.x & 0xFFFF0000u),
                                     as_type<float>(bits.y << 16), as_type<float>(bits.y & 0xFFFF0000u));
            const float4 wb = float4(as_type<float>(bits.z << 16), as_type<float>(bits.z & 0xFFFF0000u),
                                     as_type<float>(bits.w << 16), as_type<float>(bits.w & 0xFFFF0000u));
            #pragma clang loop unroll(full)
            for (uint t = 0; t < T; ++t) acc[r * T + t] += dot(wa, xa[t]) + dot(wb, xb[t]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < R * T; ++i) acc[i] = simd_sum(acc[i]);
}

template<uint R, uint T>
static inline void mv_store(
    thread const float *acc, device float *output, uint rows, uint first_row, uint tokens, uint lane
) {
    #pragma clang loop unroll(full)
    for (uint r = 0; r < R; ++r) {
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            if (lane == ((r * T + t) & 31) && t < tokens && first_row + r < rows) {
                output[ulong(t) * rows + first_row + r] = acc[r * T + t];
            }
        }
    }
}

template<uint T>
static inline void int4_mv_block(
    device const uchar *weights, device const half2 *groups, thread const MvInput<T> &x,
    device float *output, uint rows, uint columns, uint tokens, uint first_row, uint lane
) {
    Int4Rows<MV_ROWS> m;
    int4_rows_into<MV_ROWS>(m, 0, weights, groups, rows, columns, first_row, MV_ROWS);
    float acc[MV_ROWS * T];
    int4_mv_core<MV_ROWS, T>(m, x, columns, lane, acc);
    mv_store<MV_ROWS, T>(acc, output, rows, first_row, tokens, lane);
}

// Grid: ceil(rows / 8) threadgroups of 64 threads (2 SIMD groups x 4 rows).
template<uint T>
[[kernel]] void int4_mv(
    device const uchar *weights [[buffer(0)]], device const half2 *groups [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = (group * simds + simd) * MV_ROWS;
    if (first_row >= rows) return;
    const MvInput<T> x = mv_input<T>(input, columns, tokens);
    int4_mv_block<T>(weights, groups, x, output, rows, columns, tokens, first_row, lane);
}
template [[host_name("int4_mv_1")]] [[kernel]] void int4_mv<1>(
    device const uchar *, device const half2 *, device const float *, device float *,
    constant uint &, constant uint &, constant uint &, uint, uint, uint, uint);
template [[host_name("int4_mv_2")]] [[kernel]] void int4_mv<2>(
    device const uchar *, device const half2 *, device const float *, device float *,
    constant uint &, constant uint &, constant uint &, uint, uint, uint, uint);

// Up to three INT4 matrices sharing one input (Q/K/V or K/V): `dims` = (rows0, rows1, rows2, columns),
// every `rows_i` a multiple of 4; a zero-row matrix is skipped.
// Grid: ceil((rows0 + rows1 + rows2) / 8) threadgroups of 64 threads.
template<uint T>
[[kernel]] void int4_mv3(
    device const uchar *w0 [[buffer(0)]], device const half2 *g0 [[buffer(1)]],
    device const uchar *w1 [[buffer(2)]], device const half2 *g1 [[buffer(3)]],
    device const uchar *w2 [[buffer(4)]], device const half2 *g2 [[buffer(5)]],
    device const float *input [[buffer(6)]],
    device float *out0 [[buffer(7)]], device float *out1 [[buffer(8)]], device float *out2 [[buffer(9)]],
    constant uint4 &dims [[buffer(10)]], constant uint &tokens [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint first_row = (group * simds + simd) * MV_ROWS;
    const MvInput<T> x = mv_input<T>(input, dims.w, tokens);
    if (first_row < dims.x) {
        int4_mv_block<T>(w0, g0, x, out0, dims.x, dims.w, tokens, first_row, lane);
        return;
    }
    first_row -= dims.x;
    if (first_row < dims.y) {
        int4_mv_block<T>(w1, g1, x, out1, dims.y, dims.w, tokens, first_row, lane);
        return;
    }
    first_row -= dims.y;
    if (first_row < dims.z) int4_mv_block<T>(w2, g2, x, out2, dims.z, dims.w, tokens, first_row, lane);
}
template [[host_name("int4_mv3_1")]] [[kernel]] void int4_mv3<1>(
    device const uchar *, device const half2 *, device const uchar *, device const half2 *,
    device const uchar *, device const half2 *, device const float *, device float *, device float *,
    device float *, constant uint4 &, constant uint &, uint, uint, uint, uint);
template [[host_name("int4_mv3_2")]] [[kernel]] void int4_mv3<2>(
    device const uchar *, device const half2 *, device const uchar *, device const half2 *,
    device const uchar *, device const half2 *, device const float *, device float *, device float *,
    device float *, constant uint4 &, constant uint &, uint, uint, uint, uint);

// Fused Gated DeltaNet input projection: BF16 `in_proj_a` / `in_proj_b` ([ab_rows, columns] each)
// in the first SIMD groups, then INT4 `in_proj_qkv` and `in_proj_z`.
// `dims` = (rows_qkv, rows_z, ab_rows, columns). Grid: ceil((2 ab_rows + rows_qkv + rows_z) / 8) x 64.
template<uint T>
[[kernel]] void int4_mv_gdn(
    device const uchar *w_qkv [[buffer(0)]], device const half2 *g_qkv [[buffer(1)]],
    device const uchar *w_z [[buffer(2)]], device const half2 *g_z [[buffer(3)]],
    device const ushort *w_a [[buffer(4)]], device const ushort *w_b [[buffer(5)]],
    device const float *input [[buffer(6)]],
    device float *out_qkv [[buffer(7)]], device float *out_z [[buffer(8)]],
    device float *out_a [[buffer(9)]], device float *out_b [[buffer(10)]],
    constant uint4 &dims [[buffer(11)]], constant uint &tokens [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint first_row = (group * simds + simd) * MV_ROWS;
    const MvInput<T> x = mv_input<T>(input, dims.w, tokens);
    if (first_row < 2 * dims.z) {
        const bool is_b = first_row >= dims.z;
        const uint row = first_row - (is_b ? dims.z : 0);
        float acc[MV_ROWS * T];
        bf16_mv_core<MV_ROWS, T>(is_b ? w_b : w_a, dims.z, row, x, dims.w, lane, acc);
        mv_store<MV_ROWS, T>(acc, is_b ? out_b : out_a, dims.z, row, tokens, lane);
        return;
    }
    first_row -= 2 * dims.z;
    if (first_row < dims.x) {
        int4_mv_block<T>(w_qkv, g_qkv, x, out_qkv, dims.x, dims.w, tokens, first_row, lane);
        return;
    }
    first_row -= dims.x;
    if (first_row < dims.y) int4_mv_block<T>(w_z, g_z, x, out_z, dims.y, dims.w, tokens, first_row, lane);
}
template [[host_name("int4_mv_gdn_1")]] [[kernel]] void int4_mv_gdn<1>(
    device const uchar *, device const half2 *, device const uchar *, device const half2 *,
    device const ushort *, device const ushort *, device const float *, device float *, device float *,
    device float *, device float *, constant uint4 &, constant uint &, uint, uint, uint, uint);
template [[host_name("int4_mv_gdn_2")]] [[kernel]] void int4_mv_gdn<2>(
    device const uchar *, device const half2 *, device const uchar *, device const half2 *,
    device const ushort *, device const ushort *, device const float *, device float *, device float *,
    device float *, device float *, constant uint4 &, constant uint &, uint, uint, uint, uint);

// Fused SwiGLU: each SIMD group computes 2 gate + the same 2 up rows, then `silu(gate) * up`.
// Grid: ceil(rows / 4) threadgroups of 64 threads.
template<uint T>
[[kernel]] void int4_mv_swiglu(
    device const uchar *gate [[buffer(0)]], device const half2 *gate_groups [[buffer(1)]],
    device const uchar *up [[buffer(2)]], device const half2 *up_groups [[buffer(3)]],
    device const float *input [[buffer(4)]], device float *output [[buffer(5)]],
    constant uint &rows [[buffer(6)]], constant uint &columns [[buffer(7)]],
    constant uint &tokens [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = (group * simds + simd) * 2;
    if (first_row >= rows) return;
    Int4Rows<4> m;
    int4_rows_into<4>(m, 0, gate, gate_groups, rows, columns, first_row, 2);
    int4_rows_into<4>(m, 2, up, up_groups, rows, columns, first_row, 2);
    const MvInput<T> x = mv_input<T>(input, columns, tokens);
    float acc[4 * T];
    int4_mv_core<4, T>(m, x, columns, lane, acc);
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            if (lane == r * T + t && t < tokens && first_row + r < rows) {
                output[ulong(t) * rows + first_row + r] = nano_silu(acc[r * T + t]) * acc[(2 + r) * T + t];
            }
        }
    }
}
template [[host_name("int4_mv_swiglu_1")]] [[kernel]] void int4_mv_swiglu<1>(
    device const uchar *, device const half2 *, device const uchar *, device const half2 *,
    device const float *, device float *, constant uint &, constant uint &, constant uint &,
    uint, uint, uint, uint);
template [[host_name("int4_mv_swiglu_2")]] [[kernel]] void int4_mv_swiglu<2>(
    device const uchar *, device const half2 *, device const uchar *, device const half2 *,
    device const float *, device float *, constant uint &, constant uint &, constant uint &,
    uint, uint, uint, uint);

// `SPLIT` SIMD groups share each tile's columns (8 on wide GPUs; fewer keeps small GPUs from
// spending most of each threadgroup on the cross-SIMD reduction).
template<uint TILES, uint SPLIT>
static inline void int4_wide(
    device const uchar *weights, device const half2 *groups, device const float *input,
    device float *output, uint rows, uint columns, uint tokens, uint2 group, uint simd,
    uint lane, threadgroup float *shared
) {
    const uint row_group = group.x;
    const uint token_offset = group.y * 8;
    const uint chunk_tokens = min(8u, tokens - token_offset);
    input += ulong(token_offset) * columns;
    output += ulong(token_offset) * rows;
    constexpr uint split = SPLIT;
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = row_group * 8 * TILES;
    const uint row_groups = (columns + 63) / 64;
    const uint row_code_bytes = row_groups * 32;
    device const float *x_row = input + ulong(fm % chunk_tokens) * columns;
    device const uchar *w_rows[2 * TILES];
    device const half2 *g_rows[2 * TILES];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2 * TILES; ++r) {
        const uint row = min(first_row + 8 * (r / 2) + fn + r % 2, rows - 1);
        w_rows[r] = weights + ulong(row) * row_code_bytes;
        g_rows[r] = groups + ulong(row) * row_groups;
    }
    simdgroup_float8x8 sums[TILES];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TILES; ++i) sums[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (columns % 128 == 0) {
        device const float *x_fast = x_row + 16 * fn;
        for (uint start = simd * 128; start < columns; start += split * 128) {
            float w[2 * TILES][16];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 2 * TILES; ++r) {
                const uint2 bits = *reinterpret_cast<device const uint2 *>(w_rows[r] + start / 2 + 8 * fm);
                const float2 sb = float2(g_rows[r][(start + 16 * fm) / 64]);
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 2; ++q) {
                    const float4 lo = float4(as_type<uchar4>(bits[q] & 0x0F0F0F0Fu));
                    const float4 hi = float4(as_type<uchar4>((bits[q] >> 4) & 0x0F0F0F0Fu));
                    #pragma clang loop unroll(full)
                    for (uint e = 0; e < 4; ++e) {
                        w[r][8 * q + 2 * e] = fma(lo[e], sb.x, sb.y);
                        w[r][8 * q + 2 * e + 1] = fma(hi[e], sb.x, sb.y);
                    }
                }
            }
            float xa[16], xb[16];
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) {
                const float4 a = *reinterpret_cast<device const float4 *>(x_fast + start + 4 * q);
                const float4 b = *reinterpret_cast<device const float4 *>(x_fast + start + 16 + 4 * q);
                #pragma clang loop unroll(full)
                for (uint e = 0; e < 4; ++e) {
                    xa[4 * q + e] = a[e];
                    xb[4 * q + e] = b[e];
                }
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 16; ++j) {
                simdgroup_float8x8 x;
                x.thread_elements()[0] = xa[j];
                x.thread_elements()[1] = xb[j];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < TILES; ++i) {
                    simdgroup_float8x8 b;
                    b.thread_elements()[0] = w[2 * i][j];
                    b.thread_elements()[1] = w[2 * i + 1][j];
                    simdgroup_multiply_accumulate(sums[i], x, b, sums[i]);
                }
            }
        }
    } else {
        const uint num_chunks = (columns + 127) / 128;
        for (uint chunk = simd; chunk < num_chunks; chunk += split) {
            const uint start = chunk * 128;
            float w[2 * TILES][16];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 2 * TILES; ++r) {
                const uint col_w = start + 16 * fm;
                if (col_w < columns) {
                    const uint2 bits = *reinterpret_cast<device const uint2 *>(w_rows[r] + col_w / 2);
                    const float2 sb = float2(g_rows[r][col_w / 64]);
                    #pragma clang loop unroll(full)
                    for (uint q = 0; q < 2; ++q) {
                        const float4 lo = float4(as_type<uchar4>(bits[q] & 0x0F0F0F0Fu));
                        const float4 hi = float4(as_type<uchar4>((bits[q] >> 4) & 0x0F0F0F0Fu));
                        #pragma clang loop unroll(full)
                        for (uint e = 0; e < 4; ++e) {
                            const uint idx = 8 * q + 2 * e;
                            w[r][idx] = (col_w + idx < columns) ? fma(lo[e], sb.x, sb.y) : 0.0f;
                            w[r][idx + 1] = (col_w + idx + 1 < columns) ? fma(hi[e], sb.x, sb.y) : 0.0f;
                        }
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 16; ++j) w[r][j] = 0.0f;
                }
            }
            float xa[16], xb[16];
            const uint col_x = start + 16 * fn;
            if (col_x + 32 <= columns) {
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4; ++q) {
                    const float4 a = *reinterpret_cast<device const float4 *>(x_row + col_x + 4 * q);
                    const float4 b = *reinterpret_cast<device const float4 *>(x_row + col_x + 16 + 4 * q);
                    #pragma clang loop unroll(full)
                    for (uint e = 0; e < 4; ++e) {
                        xa[4 * q + e] = a[e];
                        xb[4 * q + e] = b[e];
                    }
                }
            } else if (col_x < columns) {
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4; ++q) {
                    const float4 a = (col_x + 4 * q < columns)
                        ? *reinterpret_cast<device const float4 *>(x_row + col_x + 4 * q)
                        : float4(0.0f);
                    const float4 b = (col_x + 16 + 4 * q < columns)
                        ? *reinterpret_cast<device const float4 *>(x_row + col_x + 16 + 4 * q)
                        : float4(0.0f);
                    #pragma clang loop unroll(full)
                    for (uint e = 0; e < 4; ++e) {
                        xa[4 * q + e] = a[e];
                        xb[4 * q + e] = b[e];
                    }
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 16; ++j) {
                    xa[j] = 0.0f;
                    xb[j] = 0.0f;
                }
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 16; ++j) {
                simdgroup_float8x8 x;
                x.thread_elements()[0] = xa[j];
                x.thread_elements()[1] = xb[j];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < TILES; ++i) {
                    simdgroup_float8x8 b;
                    b.thread_elements()[0] = w[2 * i][j];
                    b.thread_elements()[1] = w[2 * i + 1][j];
                    simdgroup_multiply_accumulate(sums[i], x, b, sums[i]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TILES; ++i) {
        shared[(i * split + simd) * 64 + lane * 2] = sums[i].thread_elements()[0];
        shared[(i * split + simd) * 64 + lane * 2 + 1] = sums[i].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (fm >= chunk_tokens) return;
    for (uint tile = simd; tile < TILES; tile += split) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            float total = 0.0f;
            #pragma clang loop unroll(full)
            for (uint s = 0; s < split; ++s) total += shared[(tile * split + s) * 64 + lane * 2 + r];
            const uint row = first_row + 8 * tile + fn + r;
            if (row < rows) output[ulong(fm) * rows + row] = total;
        }
    }
}

template<uint S>
[[kernel]] void int4_matmul_wide(
    device const uchar *weights [[buffer(0)]], device const half2 *groups [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[S * 64];
    int4_wide<1, S>(weights, groups, input, output, rows, columns, tokens, group, simd, lane, shared);
}

template<uint S>
[[kernel]] void int4_matmul_wide_2(
    device const uchar *weights [[buffer(0)]], device const half2 *groups [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[2 * S * 64];
    int4_wide<2, S>(weights, groups, input, output, rows, columns, tokens, group, simd, lane, shared);
}

// `lm_head`-sized matrices: 32 rows per threadgroup reuse each input fragment four times.
template<uint S>
[[kernel]] void int4_matmul_wide_4(
    device const uchar *weights [[buffer(0)]], device const half2 *groups [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[4 * S * 64];
    int4_wide<4, S>(weights, groups, input, output, rows, columns, tokens, group, simd, lane, shared);
}

template<uint S>
[[kernel]] void int4_matmul_wide_concat3(
    device const uchar *w0 [[buffer(0)]], device const half2 *g0 [[buffer(1)]],
    device const uchar *w1 [[buffer(2)]], device const half2 *g1 [[buffer(3)]],
    device const uchar *w2 [[buffer(4)]], device const half2 *g2 [[buffer(5)]],
    device const float *input [[buffer(6)]],
    device float *out0 [[buffer(7)]], device float *out1 [[buffer(8)]], device float *out2 [[buffer(9)]],
    constant uint &rows0 [[buffer(10)]], constant uint &rows1 [[buffer(11)]], constant uint &rows2 [[buffer(12)]],
    constant uint &columns [[buffer(13)]], constant uint &tokens [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[S * 64];
    const uint g0_cnt = rows0 / 8;
    const uint g1_cnt = rows1 / 8;
    if (group.x < g0_cnt) {
        int4_wide<1, S>(w0, g0, input, out0, rows0, columns, tokens, group, simd, lane, shared);
    } else if (group.x < g0_cnt + g1_cnt) {
        int4_wide<1, S>(w1, g1, input, out1, rows1, columns, tokens, uint2(group.x - g0_cnt, group.y), simd, lane, shared);
    } else {
        int4_wide<1, S>(w2, g2, input, out2, rows2, columns, tokens, uint2(group.x - g0_cnt - g1_cnt, group.y), simd, lane, shared);
    }
}

template<uint S>
[[kernel]] void int4_matmul_wide_2_concat2(
    device const uchar *w0 [[buffer(0)]], device const half2 *g0 [[buffer(1)]],
    device const uchar *w1 [[buffer(2)]], device const half2 *g1 [[buffer(3)]],
    device const float *input [[buffer(4)]],
    device float *out0 [[buffer(5)]], device float *out1 [[buffer(6)]],
    constant uint &rows0 [[buffer(7)]], constant uint &rows1 [[buffer(8)]],
    constant uint &columns [[buffer(9)]], constant uint &tokens [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[2 * S * 64];
    const uint g0_cnt = rows0 / 16;
    if (group.x < g0_cnt) {
        int4_wide<2, S>(w0, g0, input, out0, rows0, columns, tokens, group, simd, lane, shared);
    } else {
        int4_wide<2, S>(w1, g1, input, out1, rows1, columns, tokens, uint2(group.x - g0_cnt, group.y), simd, lane, shared);
    }
}

template<uint S>
[[kernel]] void int4_matmul_wide_swiglu(
    device const uchar *gate_weights [[buffer(0)]], device const half2 *gate_groups [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]], device const half2 *up_groups [[buffer(3)]],
    device const float *input [[buffer(4)]], device float *output [[buffer(5)]],
    constant uint &rows [[buffer(6)]], constant uint &columns [[buffer(7)]],
    constant uint &tokens [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[2 * S * 64];
    const uint row_group = group.x;
    const uint token_offset = group.y * 8;
    const uint chunk_tokens = min(8u, tokens - token_offset);
    input += ulong(token_offset) * columns;
    output += ulong(token_offset) * rows;
    constexpr uint split = S;
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = row_group * 8;
    const uint row_groups = columns / 64;
    const uint row_code_bytes = row_groups * 32;
    device const float *x_fast = input + ulong(fm % chunk_tokens) * columns + 16 * fn;
    device const uchar *wg_rows[2], *wu_rows[2];
    device const half2 *gg_rows[2], *gu_rows[2];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        const uint row = min(first_row + fn + r, rows - 1);
        wg_rows[r] = gate_weights + ulong(row) * row_code_bytes;
        gg_rows[r] = gate_groups + ulong(row) * row_groups;
        wu_rows[r] = up_weights + ulong(row) * row_code_bytes;
        gu_rows[r] = up_groups + ulong(row) * row_groups;
    }
    simdgroup_float8x8 sum_g = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 sum_u = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint start = simd * 128; start < columns; start += split * 128) {
        float wg[2][16], wu[2][16];
        const uint grp_idx = (start + 16 * fm) / 64;
        const uint code_off = start / 2 + 8 * fm;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            const uint2 bits_g = *reinterpret_cast<device const uint2 *>(wg_rows[r] + code_off);
            const float2 sb_g = float2(gg_rows[r][grp_idx]);
            const uint2 bits_u = *reinterpret_cast<device const uint2 *>(wu_rows[r] + code_off);
            const float2 sb_u = float2(gu_rows[r][grp_idx]);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 2; ++q) {
                const float4 lo_g = float4(as_type<uchar4>(bits_g[q] & 0x0F0F0F0Fu));
                const float4 hi_g = float4(as_type<uchar4>((bits_g[q] >> 4) & 0x0F0F0F0Fu));
                const float4 lo_u = float4(as_type<uchar4>(bits_u[q] & 0x0F0F0F0Fu));
                const float4 hi_u = float4(as_type<uchar4>((bits_u[q] >> 4) & 0x0F0F0F0Fu));
                #pragma clang loop unroll(full)
                for (uint e = 0; e < 4; ++e) {
                    wg[r][8 * q + 2 * e] = fma(lo_g[e], sb_g.x, sb_g.y);
                    wg[r][8 * q + 2 * e + 1] = fma(hi_g[e], sb_g.x, sb_g.y);
                    wu[r][8 * q + 2 * e] = fma(lo_u[e], sb_u.x, sb_u.y);
                    wu[r][8 * q + 2 * e + 1] = fma(hi_u[e], sb_u.x, sb_u.y);
                }
            }
        }
        float xa[16], xb[16];
        #pragma clang loop unroll(full)
        for (uint q = 0; q < 4; ++q) {
            const float4 a = *reinterpret_cast<device const float4 *>(x_fast + start + 4 * q);
            const float4 b = *reinterpret_cast<device const float4 *>(x_fast + start + 16 + 4 * q);
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 4; ++e) {
                xa[4 * q + e] = a[e];
                xb[4 * q + e] = b[e];
            }
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 16; ++j) {
            simdgroup_float8x8 x;
            x.thread_elements()[0] = xa[j];
            x.thread_elements()[1] = xb[j];
            simdgroup_float8x8 bg, bu;
            bg.thread_elements()[0] = wg[0][j];
            bg.thread_elements()[1] = wg[1][j];
            bu.thread_elements()[0] = wu[0][j];
            bu.thread_elements()[1] = wu[1][j];
            simdgroup_multiply_accumulate(sum_g, x, bg, sum_g);
            simdgroup_multiply_accumulate(sum_u, x, bu, sum_u);
        }
    }
    shared[simd * 64 + lane * 2] = sum_g.thread_elements()[0];
    shared[simd * 64 + lane * 2 + 1] = sum_g.thread_elements()[1];
    shared[(split + simd) * 64 + lane * 2] = sum_u.thread_elements()[0];
    shared[(split + simd) * 64 + lane * 2 + 1] = sum_u.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0 || fm >= chunk_tokens) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        float gate_val = 0.0f;
        float up_val = 0.0f;
        #pragma clang loop unroll(full)
        for (uint s = 0; s < split; ++s) {
            gate_val += shared[s * 64 + lane * 2 + r];
            up_val += shared[(split + s) * 64 + lane * 2 + r];
        }
        const uint row = first_row + fn + r;
        if (row < rows) output[ulong(fm) * rows + row] = nano_silu(gate_val) * up_val;
    }
}

// Each wide kernel for 8 SIMD groups per threadgroup (`NAME`) and for 2 (`NAME_s2`, small GPUs).
#define WIDE_MV_ARGS device const uchar *, device const half2 *, device const float *, device float *, \
    constant uint &, constant uint &, constant uint &, uint2, uint, uint
#define WIDE_KERNELS(S, SUFFIX) \
template [[host_name("int4_matmul_wide" SUFFIX)]] [[kernel]] void int4_matmul_wide<S>(WIDE_MV_ARGS); \
template [[host_name("int4_matmul_wide_2" SUFFIX)]] [[kernel]] void int4_matmul_wide_2<S>(WIDE_MV_ARGS); \
template [[host_name("int4_matmul_wide_4" SUFFIX)]] [[kernel]] void int4_matmul_wide_4<S>(WIDE_MV_ARGS); \
template [[host_name("int4_matmul_wide_concat3" SUFFIX)]] [[kernel]] void int4_matmul_wide_concat3<S>( \
    device const uchar *, device const half2 *, device const uchar *, device const half2 *, \
    device const uchar *, device const half2 *, device const float *, device float *, device float *, \
    device float *, constant uint &, constant uint &, constant uint &, constant uint &, constant uint &, \
    uint2, uint, uint); \
template [[host_name("int4_matmul_wide_2_concat2" SUFFIX)]] [[kernel]] void int4_matmul_wide_2_concat2<S>( \
    device const uchar *, device const half2 *, device const uchar *, device const half2 *, \
    device const float *, device float *, device float *, constant uint &, constant uint &, \
    constant uint &, constant uint &, uint2, uint, uint); \
template [[host_name("int4_matmul_wide_swiglu" SUFFIX)]] [[kernel]] void int4_matmul_wide_swiglu<S>( \
    device const uchar *, device const half2 *, device const uchar *, device const half2 *, \
    device const float *, device float *, constant uint &, constant uint &, constant uint &, \
    uint2, uint, uint);
WIDE_KERNELS(8, "")
WIDE_KERNELS(2, "_s2")

// ============================================================================
// 3. Ternary PTQ1_0 & Signed Hadamard (512-block) Kernels
// ============================================================================

kernel void fwht512_forward(
    device const float *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &blocks_per_row [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[512];
    const ulong base = ulong(block) * 512;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) values[i] = input[base + i * 64 + tid];
    fwht_512_values<false>(values, signs, output, blocks_per_row, block, tid, shared);
}

kernel void fwht512_inverse(
    device const float *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &blocks_per_row [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[512];
    const ulong base = ulong(block) * 512;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) values[i] = input[base + i * 64 + tid];
    fwht_512_values<true>(values, signs, output, blocks_per_row, block, tid, shared);
}

template<uint rows_per_group>
struct Ptq1Rows {
    device const Ptq1Block *blocks[rows_per_group];
};

template<uint rows_per_group>
static inline Ptq1Rows<rows_per_group> ptq1_rows(
    device const Ptq1Block *weights, uint first_row, uint rows, uint blocks
) {
    Ptq1Rows<rows_per_group> result;
    #pragma clang loop unroll(full)
    for (uint row = 0; row < rows_per_group; ++row) {
        result.blocks[row] = weights + ulong(min(first_row + row, rows - 1)) * blocks;
    }
    return result;
}

template<uint rows_per_group>
static inline void ptq1_matvec_reuse_impl(
    Ptq1Rows<rows_per_group> row_blocks,
    device const float *input,
    uint columns,
    uint lane,
    thread float *sums
) {
    const uint blocks = columns / 128;
    const uint part = lane & 3;
    const float power = float4(1.0f, 3.0f, 9.0f, 27.0f)[part];
    #pragma clang loop unroll(full)
    for (uint row = 0; row < rows_per_group; ++row) sums[row] = 0.0f;
    for (uint block = lane / 4; block < blocks; block += 8) {
        device const float *x = input + block * 128;
        float4 wide[5];
        float2 narrow[5];
        float4 wide_sum = 0.0f;
        float2 narrow_sum = 0.0f;
        {
            float4 v[5];
            float2 u[5];
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 5; ++n) {
                v[n] = *reinterpret_cast<device const float4 *>(x + 16 * n + 4 * part);
                u[n] = *reinterpret_cast<device const float2 *>(x + 80 + 8 * n + 2 * part);
                wide_sum += v[n];
                narrow_sum += u[n];
            }
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) {
                wide[n] = v[n] - 3.0f * v[n + 1];
                narrow[n] = u[n] - 3.0f * u[n + 1];
            }
            wide[4] = v[4];
            narrow[4] = u[4];
        }
        const float2 high_inputs = *reinterpret_cast<device const float2 *>(x + 120 + 2 * part);
        const float input_sum = (wide_sum.x + wide_sum.y) + (wide_sum.z + wide_sum.w) +
            (narrow_sum.x + narrow_sum.y) + (high_inputs.x + high_inputs.y);
        #pragma clang loop unroll(full)
        for (uint row = 0; row < rows_per_group; ++row) {
            device const Ptq1Block &w = row_blocks.blocks[row][block];
            const uchar4 quad = *reinterpret_cast<device const uchar4 *>(&w.qs[4 * part]);
            const uchar2 pair = *reinterpret_cast<device const uchar2 *>(&w.qs[16 + 2 * part]);
            const uchar2 high = *reinterpret_cast<device const uchar2 *>(&w.qh[0]);
            const uchar bytes[6] = {quad.x, quad.y, quad.z, quad.w, pair.x, pair.y};
            float total = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 6; ++i) {
                const half code = ptq1_code(bytes[i]);
                #pragma clang loop unroll(full)
                for (uint n = 0; n < 5; ++n) {
                    total += float(ptq1_prefix(code, n + 1)) *
                        (i < 4 ? wide[n][i] : narrow[n][i - 4]);
                }
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                const float code = float(high[i]) * (1.0f / 256.0f) * power;
                total += (floor(3.0f * code) - 3.0f * floor(code)) * high_inputs[i];
            }
            sums[row] += (total - input_sum) * float(w.d);
        }
    }
    #pragma clang loop unroll(full)
    for (uint row = 0; row < rows_per_group; ++row) sums[row] = simd_sum(sums[row]);
}

kernel void ptq1_matvec(
    device const Ptq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = group * 8;
    float sums[8];
    ptq1_matvec_reuse_impl<8>(
        ptq1_rows<8>(weights, first_row, rows, columns / 128), input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < 8; ++row) {
        if (lane == 0 && first_row + row < rows) output[first_row + row] = sums[row];
    }
}

kernel void ptq1_matvec_swiglu(
    device const Ptq1Block *gate [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    device const Ptq1Block *up [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = group * 4, blocks = columns / 128;
    const Ptq1Rows<4> gate_rows = ptq1_rows<4>(gate, first_row, rows, blocks);
    const Ptq1Rows<4> up_rows = ptq1_rows<4>(up, first_row, rows, blocks);
    Ptq1Rows<8> both;
    #pragma clang loop unroll(full)
    for (uint row = 0; row < 4; ++row) {
        both.blocks[row] = gate_rows.blocks[row];
        both.blocks[4 + row] = up_rows.blocks[row];
    }
    float sums[8];
    ptq1_matvec_reuse_impl<8>(both, input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < 4; ++row) {
        if (lane == 0 && first_row + row < rows) {
            output[first_row + row] = nano_silu(sums[row]) * sums[4 + row];
        }
    }
}

// Small-batch (2..4 rows) PTQ1_0 kernel using lane-pair `simd_shuffle_xor(own[i], 1)` activation sharing
// and balanced half-FMA trits (`bonsai_small_batch.h`). Up to 27% faster than `ptq1_wide` on 2..4 rows!
static inline void ptq1_values_16(
    device const float *input,
    ulong base,
    uint block,
    uint part,
    thread float (&values)[16]
) {
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 3; ++k) {
        const uint offset = k < 2 ? 2 * part + k : 80 + part;
        const uint stride = k < 2 ? 16 : 8;
        #pragma clang loop unroll(full)
        for (uint n = 0; n < 5; ++n) {
            values[5 * k + n] = input[base + block * 128 + offset + n * stride];
        }
    }
    values[15] = input[base + block * 128 + 120 + part];
}

template<uint TOKENS, uint ROWS_PER_LANE>
static inline void ptq1_small_batch_impl(
    device const Ptq1Block *weights,
    device const float *input,
    device float *output,
    uint rows,
    uint columns,
    uint group,
    uint lane
) {
    constexpr uint steps = (TOKENS + 1) / 2;
    const uint slice = lane & 1;
    const uint pair = lane >> 1;
    const uint part = pair & 7;
    const uint first_row = (2 * group + slice) * ROWS_PER_LANE;
    const uint blocks = columns / 128;
    float sums[ROWS_PER_LANE][2 * steps];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_LANE; ++r) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2 * steps; ++i) sums[r][i] = 0.0f;
    }
    for (uint block = pair / 8; block < blocks; block += 2) {
        half factors[ROWS_PER_LANE][16];
        float scales[ROWS_PER_LANE];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PER_LANE; ++r) {
            const uint source = min(first_row + r, rows - 1);
            device const Ptq1Block &b = weights[ulong(source) * blocks + block];
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 3; ++k) {
                const uint byte_idx = k < 2 ? 2 * part + k : 16 + part;
                ptq1_trits(ptq1_code(b.qs[byte_idx]), &factors[r][5 * k]);
            }
            const float power = float4(1.0f, 3.0f, 9.0f, 27.0f)[part >> 1];
            const float high = float(b.qh[part & 1]) * (1.0f / 256.0f) * power;
            factors[r][15] = half(floor(3.0f * high) - 3.0f * floor(high) - 1.0f);
            scales[r] = float(b.d);
        }
        #pragma clang loop unroll(full)
        for (uint step = 0; step < steps; ++step) {
            const bool paired = 2 * step + 1 < TOKENS;
            const uint token = 2 * step + (paired ? slice : 0);
            float own[16];
            ptq1_values_16(input, ulong(token) * columns, block, part, own);
            #pragma clang loop unroll(full)
            for (uint m = 0; m < (paired ? 2u : 1u); ++m) {
                float values[16];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 16; ++i) {
                    values[i] = m == 0 ? own[i] : simd_shuffle_xor(own[i], 1);
                }
                #pragma clang loop unroll(full)
                for (uint r = 0; r < ROWS_PER_LANE; ++r) {
                    float total = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 16; ++i) {
                        total = fma(float(factors[r][i]), values[i], total);
                    }
                    sums[r][2 * step + m] = fma(total, scales[r], sums[r][2 * step + m]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < ROWS_PER_LANE; ++r) {
        #pragma clang loop unroll(full)
        for (uint step = 0; step < steps; ++step) {
            const bool paired = 2 * step + 1 < TOKENS;
            #pragma clang loop unroll(full)
            for (uint m = 0; m < (paired ? 2u : 1u); ++m) {
                float total = sums[r][2 * step + m];
                #pragma clang loop unroll(full)
                for (uint distance = 2; distance < 32; distance *= 2) {
                    total += simd_shuffle_xor(total, distance);
                }
                const uint token = 2 * step + (paired ? (slice ^ m) : 0);
                if (pair == 0 && first_row + r < rows) {
                    output[ulong(token) * rows + first_row + r] = total;
                }
            }
        }
    }
}

kernel void ptq1_matmul_2(
    device const Ptq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    ptq1_small_batch_impl<2, 4>(weights, input, output, rows, columns, group, lane);
}

kernel void ptq1_matmul_3(
    device const Ptq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    ptq1_small_batch_impl<3, 4>(weights, input, output, rows, columns, group, lane);
}

kernel void ptq1_matmul_4(
    device const Ptq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    ptq1_small_batch_impl<4, 4>(weights, input, output, rows, columns, group, lane);
}

// Multi-row PTQ1 matmul for 2..8 tokens using simdgroup_float8x8 and balanced half-FMA trits.
// Shares weight decoding across up to 8 token rows simultaneously!
template<uint TILES>
static inline void ptq1_wide_impl(
    device const Ptq1Block *weights,
    device const float *input,
    device float *output,
    uint rows,
    uint columns,
    uint tokens,
    uint2 group,
    uint simd,
    uint lane,
    threadgroup float *shared
) {
    const uint row_group = group.x;
    const uint token_offset = group.y * 8;
    const uint chunk_tokens = min(8u, tokens - token_offset);
    input += ulong(token_offset) * columns;
    output += ulong(token_offset) * rows;
    constexpr uint split = 8;
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = row_group * 8 * TILES;
    const uint blocks = columns / 128;
    ulong row_base[TILES][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TILES; ++t) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            row_base[t][r] = ulong(min(first_row + 8 * t + fn + r, rows - 1)) * blocks;
        }
    }
    const ulong input_row = ulong(min(fm, chunk_tokens - 1)) * columns + fn;
    const ushort power = ushort4(1, 3, 9, 27)[fm >> 1];
    float sums[TILES][2] = {};
    for (uint block = simd; block < blocks; block += split) {
        half trits[TILES][2][16];
        float scales[TILES][2];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TILES; ++t) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 2; ++r) {
                device const Ptq1Block &blk = weights[row_base[t][r] + block];
                scales[t][r] = float(blk.d);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 3; ++k) {
                    const uint byte = k == 0 ? fm : (k == 1 ? fm + 8 : 16 + fm);
                    half f[5];
                    ptq1_trits(ptq1_code(blk.qs[byte]), f);
                    #pragma clang loop unroll(full)
                    for (uint n = 0; n < 5; ++n) {
                        trits[t][r][k < 2 ? 2 * n + k : 10 + n] = f[n];
                    }
                }
                const ushort high = (ushort(blk.qh[fm & 1]) * power) & 255;
                trits[t][r][15] = half((high * 3) >> 8) - 1.0h;
            }
        }
        const ulong base = input_row + ulong(block) * 128;
        simdgroup_float8x8 partial[TILES];
        #pragma clang loop unroll(full)
        for (uint tile = 0; tile < 16; ++tile) {
            simdgroup_float8x8 x;
            const float2 pair = *reinterpret_cast<device const float2 *>(input + base + tile * 8);
            x.thread_elements()[0] = pair.x;
            x.thread_elements()[1] = pair.y;
            #pragma clang loop unroll(full)
            for (uint t = 0; t < TILES; ++t) {
                simdgroup_float8x8 w;
                w.thread_elements()[0] = float(trits[t][0][tile]);
                w.thread_elements()[1] = float(trits[t][1][tile]);
                if (tile == 0) {
                    simdgroup_multiply(partial[t], x, w);
                } else {
                    simdgroup_multiply_accumulate(partial[t], x, w, partial[t]);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TILES; ++t) {
            sums[t][0] = fma(partial[t].thread_elements()[0], scales[t][0], sums[t][0]);
            sums[t][1] = fma(partial[t].thread_elements()[1], scales[t][1], sums[t][1]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TILES; ++t) {
        shared[(t * split + simd) * 64 + lane * 2] = sums[t][0];
        shared[(t * split + simd) * 64 + lane * 2 + 1] = sums[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0 || fm >= chunk_tokens) return;
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TILES; ++t) {
        #pragma clang loop unroll(full)
        for (uint s = 1; s < split; ++s) {
            sums[t][0] += shared[(t * split + s) * 64 + lane * 2];
            sums[t][1] += shared[(t * split + s) * 64 + lane * 2 + 1];
        }
        const uint row = first_row + 8 * t + fn;
        const ulong out = ulong(fm) * rows + row;
        if (row < rows) output[out] = sums[t][0];
        if (row + 1 < rows) output[out + 1] = sums[t][1];
    }
}

kernel void ptq1_matmul_wide(
    device const Ptq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[8 * 64];
    ptq1_wide_impl<1>(weights, input, output, rows, columns, tokens, group, simd, lane, shared);
}

kernel void ptq1_matmul_wide_2(
    device const Ptq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[2 * 8 * 64];
    ptq1_wide_impl<2>(weights, input, output, rows, columns, tokens, group, simd, lane, shared);
}

// ============================================================================
// 4. BF16 Matvec & Embedding Gather Kernels
// ============================================================================

kernel void bf16_mv(
    device const ushort *weights [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group % rows, token = group / rows;
    const ulong weight_row = ulong(row) * columns;
    const ulong input_row = ulong(token) * columns;
    float sum = 0.0f;
    uint column = lane;
    for (; column + 7 * 32 < columns; column += 8 * 32) {
        float weight[8], value[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            weight[i] = as_type<float>(uint(weights[weight_row + column + i * 32]) << 16);
            value[i] = input[input_row + column + i * 32];
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) sum = fma(weight[i], value[i], sum);
    }
    for (; column < columns; column += 32) {
        const float weight = as_type<float>(uint(weights[weight_row + column]) << 16);
        sum = fma(weight, input[input_row + column], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) output[group] = sum;
}

// Multi-row BF16 pair matmul (`w_a` [rows, columns] + `w_b` [rows, columns]) for `tokens` rows
// (`rows` a multiple of 4). Grid: `(rows / 2, tokens)` threadgroups of 32 threads; each SIMD group
// computes 4 rows of `w_a` or `w_b` for one token with vectorized `uint4` loads.
kernel void bf16_mv_pair(
    device const ushort *w_a [[buffer(0)]],
    device const ushort *w_b [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *out_a [[buffer(3)]],
    device float *out_b [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint block = group.x * 4;
    const bool is_b = block >= rows;
    const uint row = block - (is_b ? rows : 0);
    const MvInput<1> x = mv_input<1>(input + ulong(group.y) * columns, columns, 1);
    float acc[4];
    bf16_mv_core<4, 1>(is_b ? w_b : w_a, rows, row, x, columns, lane, acc);
    mv_store<4, 1>(acc, (is_b ? out_b : out_a) + ulong(group.y) * rows, rows, row, 1, lane);
}

// Gather token embeddings from BF16 or int8 table into F32 [tokens, width]
kernel void embed_gather_bf16(
    device const ushort *table [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &vocab [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tokens * width) return;
    const uint t = gid / width, c = gid % width;
    const uint id = token_ids[t];
    output[gid] = id < vocab ? as_type<float>(uint(table[ulong(id) * width + c]) << 16) : 0.0f;
}

kernel void embed_gather_int8(
    device const char *table [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const uint *token_ids [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tokens * width) return;
    const uint t = gid / width, c = gid % width;
    const uint id = token_ids[t];
    output[gid] = id < vocab ? float(table[ulong(id) * width + c]) * scales[id] : 0.0f;
}

static inline float ptq1_element(device const Ptq1Block &block, uint element) {
    uint byte, power;
    if (element < 80) {
        byte = element % 16;
        power = element / 16;
    } else if (element < 120) {
        byte = 16 + (element - 80) % 8;
        power = (element - 80) / 8;
    } else {
        byte = 24 + (element - 120) % 2;
        power = (element - 120) / 2;
    }
    const uchar code = byte < 24 ? block.qs[byte] : block.qh[byte - 24];
    const uchar pow3[5] = {1, 3, 9, 27, 81};
    const uchar q = uchar(code * pow3[power]);
    const float trit = float((ushort(q) * 3) >> 8) - 1.0f;
    return trit * float(block.d);
}

kernel void embed_gather_ptq1(
    device const Ptq1Block *table [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &vocab [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tokens * width) return;
    const uint t = gid / width, c = gid % width;
    const uint id = token_ids[t];
    if (id >= vocab) {
        output[gid] = 0.0f;
        return;
    }
    const uint blocks_per_row = width / 128;
    const ulong block_idx = ulong(id) * blocks_per_row + (c / 128);
    output[gid] = ptq1_element(table[block_idx], c % 128);
}

// Fused PTQ1_0 embedding gather + 512-block Signed Hadamard inverse rotation!
// Dispatch `tokens * (width / 512)` threadgroups of 64 threads.
kernel void embed_gather_ptq1_fwht512(
    device const Ptq1Block *table [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &blocks_per_row [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[512];
    const uint token = block / blocks_per_row;
    const uint b = block % blocks_per_row;
    const uint width = blocks_per_row * 512;
    const uint id = token_ids[token];
    if (id >= vocab) {
        const ulong base = ulong(block) * 512;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) output[base + i * 64 + tid] = 0.0f;
        return;
    }
    const uint ptq_blocks_per_row = width / 128;
    const ulong row_block_base = ulong(id) * ptq_blocks_per_row + b * 4;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint local = i * 64 + tid;
        values[i] = ptq1_element(table[row_block_base + (local / 128)], local % 128);
    }
    fwht_512_values<true>(values, signs, output, blocks_per_row, block, tid, shared);
}

// Gather token embeddings from INT4_G64 table (`codes` [vocab, width/2], `groups` [vocab, width/64] half2)
kernel void embed_gather_int4(
    device const uchar *codes [[buffer(0)]],
    device const half2 *groups [[buffer(1)]],
    device const uint *token_ids [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tokens * width) return;
    const uint t = gid / width, c = gid % width;
    const uint id = token_ids[t];
    if (id >= vocab) {
        output[gid] = 0.0f;
        return;
    }
    const uchar byte = codes[ulong(id) * (width / 2) + (c / 2)];
    const float q = float((c & 1u) ? (byte >> 4) : (byte & 0x0Fu));
    const float2 sb = float2(groups[ulong(id) * (width / 64) + (c / 64)]);
    output[gid] = fma(q, sb.x, sb.y);
}
