// MiniCPM-V 4.7 1B Vision Tower Kernels (SigLIP2 27-layer ViT + Intra-ViT 2x2 Window-Attention Merger + 2x2 DownsampleMLP Merger):
// - Width = 1152, 16 heads x 72, FFN = 4304 (gelu_tanh)
// - Layer 6 output (16x mode): 2x2 window self-attention (within each 4-patch window) + 2x2 concat (4608) + pre_norm + MLP (4608 -> 17216 -> 1152, gelu_tanh) + mean(2x2) residual
// - Post-Layer-26 Merger: 2x2 concat (4608) + pre_norm + MLP (4608 -> 4608 -> 1024, gelu_tanh)
#include "common.h"

constant constexpr uint BV_TILE = 32;
constant constexpr uint BV_PAD = 4;
constant constexpr uint BV_WIDTH = 1152;
constant constexpr uint BV_HEADS = 16;
constant constexpr uint BV_D = 72;

static inline float bv_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline float bv_gelu_tanh(float x) {
    const float c = clamp(x, -10.0f, 10.0f);
    const float k = 0.7978845608028654f; // sqrt(2 / pi)
    return 0.5f * x * (1.0f + precise::tanh(k * (c + 0.044715f * c * c * c)));
}

static inline float bv_gelu_erf(float x) {
    const float z = min(abs(x) * 0.7071067811865475f, 8.0f); // |x| / sqrt(2)
    const float t = 1.0f / (1.0f + 0.3275911f * z);
    const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    const float erf_z = 1.0f - poly * exp(-z * z);
    return 0.5f * x * (1.0f + (x < 0.0f ? -erf_z : erf_z));
}

static inline float bv_weight1(ushort w) { return bv_bf16(w); }

// 32x32 tiled simdgroup-matrix GEMM with bias and epilogue mode:
//   0: store W x + b
//   1: store gelu_tanh(W x + b)
//   3: accumulate output += W x + b
template <typename W>
static inline void bv_gemm_tile(
    device const W *weights, device const float *bias, device const float *input,
    device float *output, uint rows, uint columns, uint tokens, uint mode,
    uint2 group, uint tid, uint simd,
    threadgroup float (*xs)[BV_TILE + BV_PAD], threadgroup float (*ws)[BV_TILE + BV_PAD],
    threadgroup float (*cs)[BV_TILE + BV_PAD]
) {
    const uint row0 = group.x * BV_TILE, token0 = group.y * BV_TILE;
    const uint line = tid / 4, k8 = (tid % 4) * 8;
    const uint token = token0 + line, row = row0 + line;
    const bool token_live = token < tokens;
    device const float *x_line = input + ulong(min(token, tokens - 1)) * columns + k8;
    device const W *w_line = weights + ulong(min(row, rows - 1)) * columns + k8;
    const uint sm = (simd / 2) * 16, sn = (simd % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < columns; k0 += BV_TILE) {
        const uint k = k0 + k8;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 8; ++e) {
            const bool col_live = (k + e) < columns;
            xs[line][k8 + e] = (token_live && col_live) ? x_line[k0 + e] : 0.0f;
            ws[k8 + e][line] = (row < rows && col_live) ? bv_weight1(w_line[k0 + e]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BV_TILE; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(a[i], &xs[sm + 8 * i][kk], BV_TILE + BV_PAD);
                simdgroup_load(b[i], &ws[kk][sn + 8 * i], BV_TILE + BV_PAD);
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], &cs[sm + 8 * i][sn + 8 * j], BV_TILE + BV_PAD);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r8 = (tid % 4) * 8;
    if (!token_live) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 8; ++r) {
        const uint out_row = row0 + r8 + r;
        if (out_row >= rows) continue;
        const float value = cs[line][r8 + r] + bias[out_row];
        device float *o = output + ulong(token) * rows + out_row;
        switch (mode) {
            case 1: *o = bv_gelu_tanh(value); break;
            case 2: *o = bv_gelu_erf(value); break;
            case 3: *o = *o + value; break;
            default: *o = value; break;
        }
    }
}

kernel void bv_gemm(
    device const ushort *weights [[buffer(0)]], device const float *bias [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], constant uint &mode [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float xs[BV_TILE][BV_TILE + BV_PAD];
    threadgroup float ws[BV_TILE][BV_TILE + BV_PAD];
    threadgroup float cs[BV_TILE][BV_TILE + BV_PAD];
    bv_gemm_tile(weights, bias, input, output, rows, columns, tokens, mode, group, tid, simd, xs, ws, cs);
}

// Direct SIMD-group dot-product GEMM for F32 matrices with arbitrary column counts (e.g. 588 or 4304).
// One 32-lane SIMD group per `(row, token)`: grid `(rows, tokens)`, threadgroup `(32, 1, 1)`.
kernel void bv_gemm_f32(
    device const float *weights [[buffer(0)]], device const float *bias [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], constant uint &mode [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group.x, token = group.y;
    if (row >= rows || token >= tokens) return;
    device const float *w_row = weights + ulong(row) * columns;
    device const float *x_row = input + ulong(token) * columns;
    float sum = 0.0f;
    for (uint c = lane; c < columns; c += 32) {
        sum = fma(w_row[c], x_row[c], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        const float value = sum + bias[row];
        device float *o = output + ulong(token) * rows + row;
        switch (mode) {
            case 1: *o = bv_gelu_tanh(value); break;
            case 2: *o = bv_gelu_erf(value); break;
            case 3: *o = *o + value; break;
            default: *o = value; break;
        }
    }
}

// LayerNorm over `dimension` columns
static inline float bv_sum_128(float value, threadgroup float *partial, uint tid) {
    value = simd_sum(value);
    if (tid % 32 == 0) partial[tid / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void bv_layernorm(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &dimension [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device const float *x = input + ulong(row) * dimension;
    device float *y = output + ulong(row) * dimension;
    float sum = 0.0f;
    for (uint c = tid; c < dimension; c += 128) sum += x[c];
    const float mean = bv_sum_128(sum, partial, tid) / float(dimension);
    float squares = 0.0f;
    for (uint c = tid; c < dimension; c += 128) {
        const float d = x[c] - mean;
        squares += d * d;
    }
    const float variance = bv_sum_128(squares, partial, tid) / float(dimension);
    const float inverse = 1.0f / precise::sqrt(variance + epsilon);
    for (uint c = tid; c < dimension; c += 128) y[c] = (x[c] - mean) * inverse * weight[c] + bias[c];
}

constant constexpr uint BV_BQ = 64;
constant constexpr uint BV_BK = 32;
constant constexpr uint BV_LD = BV_D + 4; // padded row stride (76 floats)
constant constexpr uint BV_KLD = BV_BK + 4; // transposed key tile stride (36 floats)

// Bidirectional FlashAttention over one crop's `count` patches (`window == 0`):
// q, k, v, out are separate [count, 16, 72] F32 buffers.
// 256 threads (8 SIMD groups) per threadgroup own 64 queries of one head;
// keys stream through threadgroup memory 32 at a time using 8x8 simdgroup MMA
// and online base-2 softmax in registers. Grid: (ceil(count / 64), 16, 1).
kernel void bv_attention_flash(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    threadgroup float *ks_memory [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float (*ks)[BV_LD] = (threadgroup float (*)[BV_LD]) ks_memory;
    threadgroup float vs[BV_BK][BV_LD];
    threadgroup float (*kt_tile)[BV_KLD] = (threadgroup float (*)[BV_KLD]) &ks[0][0];

    const uint head = group.y;
    const uint q0 = group.x * BV_BQ;
    const ulong stride = BV_WIDTH;
    device const float *qb = q + head * BV_D;
    device const float *kb = k + head * BV_D;
    device const float *vb = v + head * BV_D;

    // Stage scaled Q (zero rows past the end) through ks and vs, then into registers.
    {
        const float qscale = 0.11785113019775793f * M_LOG2E_F;
        for (uint i = tid; i < BV_BQ * 18; i += 256) {
            const uint row = i / 18, c = (i % 18) * 4, qi = q0 + row;
            const float4 qv = qi < count
                ? *(device const float4 *)(qb + ulong(qi) * stride + c) * qscale : float4(0.0f);
            threadgroup float *t = row < BV_BK ? &ks[row][c] : &vs[row - BV_BK][c];
            *(threadgroup float4 *)t = qv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float (*qs)[BV_LD] = sg < 4 ? &ks[sg * 8] : &vs[(sg - 4) * 8];
    simdgroup_float8x8 qm[9], om[9];
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 9; ++d) {
        simdgroup_load(qm[d], &qs[0][d * 8], BV_LD);
        om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint quad = lane / 4;
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    float m = -1.0e30f, l = 0.0f;

    for (uint k0 = 0; k0 < count; k0 += BV_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BV_BK * 18; i += 256) {
            const uint row = i / 18, c = (i % 18) * 4, kj = k0 + row;
            const bool live = kj < count;
            const ulong at = ulong(kj) * stride + c;
            const float4 kv = live ? *(device const float4 *)(kb + at) : float4(0.0f);
            kt_tile[c][row] = kv.x;
            kt_tile[c + 1][row] = kv.y;
            kt_tile[c + 2][row] = kv.z;
            kt_tile[c + 3][row] = kv.w;
            *(threadgroup float4 *)&vs[row][c] = live ? *(device const float4 *)(vb + at) : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s[4];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            s[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 9; ++d) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, &kt_tile[d * 8][j * 8], BV_KLD);
                simdgroup_multiply_accumulate(s[j], qm[d], kt, s[j]);
            }
        }

        const bool tail = k0 + BV_BK > count;
        float block_max = -1.0e30f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                if (tail && k0 + j * 8 + fn + e >= count) s[j].thread_elements()[e] = -1.0e30f;
                block_max = max(block_max, s[j].thread_elements()[e]);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float m_new = max(m, block_max);
        const float correction = exp2(m - m_new);
        float row_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                const float p = exp2(s[j].thread_elements()[e] - m_new);
                s[j].thread_elements()[e] = p;
                row_sum += p;
            }
        }
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 8);
        l = fma(l, correction, row_sum);
        m = m_new;

        #pragma clang loop unroll(full)
        for (uint d = 0; d < 9; ++d) {
            om[d].thread_elements()[0] *= correction;
            om[d].thread_elements()[1] *= correction;
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 9; ++d) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, &vs[j * 8][d * 8], BV_LD);
                simdgroup_multiply_accumulate(om[d], s[j], vt, om[d]);
            }
        }
    }

    const float inv = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 9; ++d) {
        om[d].thread_elements()[0] *= inv;
        om[d].thread_elements()[1] *= inv;
        simdgroup_store(om[d], &qs[0][d * 8], BV_LD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BV_BQ * 18; i += 256) {
        const uint row = i / 18, c = (i % 18) * 4, qi = q0 + row;
        if (qi < count) {
            threadgroup float *t = row < BV_BK ? &ks[row][c] : &vs[row - BV_BK][c];
            *(device float4 *)(out + ulong(qi) * BV_WIDTH + head * BV_D + c) = *(threadgroup float4 *)t;
        }
    }
}

// Bidirectional attention over one crop's `count` patches:
// q [count, 16, 72], k [count, 16, 72], v [count, 16, 72], out [count, 16, 72].
// Also supports `window_size == 4` (for `vit_merger.self_attn` where each 4-token window attends only to its 4 tokens).
kernel void bv_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &window [[buffer(5)]], // 0 = full crop attention, 4 = 2x2 window attention
    uint2 group [[threadgroup_position_in_grid]], // (token, head)
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint qi = group.x, head = group.y;
    if (qi >= count || head >= BV_HEADS) return;
    const float scale = 0.11785113019775793f; // 1 / sqrt(72)
    device const float *qp = q + (ulong(qi) * BV_HEADS + head) * BV_D;
    // Each of 32 lanes holds up to 3 elements of the 72-d head (lane + 32*e < 72)
    float qv[3], acc[3] = {0.0f, 0.0f, 0.0f};
    #pragma clang loop unroll(full)
    for (uint e = 0; e < 3; ++e) {
        const uint d = lane + 32 * e;
        qv[e] = d < BV_D ? qp[d] * scale : 0.0f;
    }
    const uint start = window == 0 ? 0 : (qi / window) * window;
    const uint end = window == 0 ? count : min(start + window, count);
    float m = 0.0f, l = 0.0f;
    for (uint kj = start; kj < end; ++kj) {
        device const float *kp = k + (ulong(kj) * BV_HEADS + head) * BV_D;
        device const float *vp = v + (ulong(kj) * BV_HEADS + head) * BV_D;
        float s = 0.0f;
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 3; ++e) {
            const uint d = lane + 32 * e;
            if (d < BV_D) s = fma(qv[e], kp[d], s);
        }
        s = simd_sum(s);
        const float m_new = (kj == start) ? s : max(m, s);
        const float corr = (kj == start) ? 0.0f : exp(m - m_new);
        const float w = exp(s - m_new);
        l = fma(l, corr, w);
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 3; ++e) {
            const uint d = lane + 32 * e;
            if (d < BV_D) acc[e] = fma(w, vp[d], acc[e] * corr);
        }
        m = m_new;
    }
    device float *op = out + (ulong(qi) * BV_HEADS + head) * BV_D;
    const float inv = 1.0f / l;
    #pragma clang loop unroll(full)
    for (uint e = 0; e < 3; ++e) {
        const uint d = lane + 32 * e;
        if (d < BV_D) op[d] = acc[e] * inv;
    }
}

// Gather 2x2 patch windows from raster grid `[grid_h, grid_w, 1152]` into
// window-ordered `[grid_h * grid_w, 1152]` and compute their 2x2 mean `[grid_h * grid_w / 4, 1152]`.
kernel void bv_window_gather(
    device const float *input [[buffer(0)]],
    device float *window_out [[buffer(1)]],
    device float *mean_out [[buffer(2)]],
    constant uint &grid_h [[buffer(3)]],
    constant uint &grid_w [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint windows_w = grid_w / 2;
    const uint total_windows = (grid_h / 2) * windows_w;
    if (gid >= total_windows * BV_WIDTH) return;
    const uint win = gid / BV_WIDTH, c = gid % BV_WIDTH;
    const uint wy = win / windows_w, wx = win % windows_w;
    float sum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint dy = 0; dy < 2; ++dy) {
        #pragma clang loop unroll(full)
        for (uint dx = 0; dx < 2; ++dx) {
            const uint py = wy * 2 + dy, px = wx * 2 + dx;
            const uint src = (py * grid_w + px) * BV_WIDTH + c;
            const uint dst = (win * 4 + dy * 2 + dx) * BV_WIDTH + c;
            const float v = input[src];
            window_out[dst] = v;
            sum += v;
        }
    }
    (void)mean_out;
    (void)sum;
}

// Compute 2x2 window mean `[total_windows, 1152]` from window-ordered `[total_windows * 4, 1152]`.
kernel void bv_window_mean(
    device const float *window_in [[buffer(0)]],
    device float *mean_out [[buffer(1)]],
    constant uint &total_windows [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_windows * BV_WIDTH) return;
    const uint win = gid / BV_WIDTH, c = gid % BV_WIDTH;
    float sum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 4; ++k) {
        sum += window_in[(win * 4 + k) * BV_WIDTH + c];
    }
    mean_out[gid] = sum * 0.25f;
}

// Apply bias [rows] and epilogue mode (0 = store, 1 = gelu_tanh, 3 = accumulate into residual)
// over `tokens` rows of `rows` columns.
kernel void bv_bias_epilogue(
    device float *data [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *residual [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &tokens [[buffer(4)]],
    constant uint &mode [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tokens * rows) return;
    const uint r = gid % rows;
    const float val = data[gid] + bias[r];
    if (mode == 1) {
        data[gid] = bv_gelu_tanh(val);
    } else if (mode == 2) {
        data[gid] = bv_gelu_erf(val);
    } else if (mode == 3) {
        residual[gid] += val;
    } else {
        data[gid] = val;
    }
}

// 2-tile (16 rows x 8 tokens) simdgroup-matrix INT4_G64 GEMM with fused bias and epilogue mode:
//   0: store W x + b
//   1: store gelu_tanh(W x + b)
//   2: store gelu_erf(W x + b)
//   3: accumulate output += W x + b
// Grid: (ceil(rows / 16), ceil(tokens / 8), 1), threadgroup: (256, 1, 1).
static inline void bv_int4_epilogue_impl(
    device const uchar *weights,
    device const half2 *groups,
    device const float *input,
    device const float *bias,
    device float *output,
    uint rows,
    uint columns,
    uint tokens,
    uint mode,
    uint2 group,
    uint simd,
    uint lane,
    threadgroup float *shared
) {
    constexpr uint TILES = 2;
    constexpr uint split = 8;
    const uint row_group = group.x;
    const uint token_offset = group.y * 8;
    const uint chunk_tokens = min(8u, tokens - token_offset);
    input += ulong(token_offset) * columns;
    output += ulong(token_offset) * rows;
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
    if (simd >= TILES || fm >= chunk_tokens) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint s = 0; s < split; ++s) total += shared[(simd * split + s) * 64 + lane * 2 + r];
        const uint row = first_row + 8 * simd + fn + r;
        if (row < rows) {
            const float val = total + bias[row];
            const ulong out_idx = ulong(fm) * rows + row;
            if (mode == 1) {
                output[out_idx] = bv_gelu_tanh(val);
            } else if (mode == 2) {
                output[out_idx] = bv_gelu_erf(val);
            } else if (mode == 3) {
                output[out_idx] += val;
            } else {
                output[out_idx] = val;
            }
        }
    }
}

kernel void bv_int4_matmul_epilogue(
    device const uchar *weights [[buffer(0)]],
    device const half2 *groups [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device const float *bias [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    constant uint &mode [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[2 * 8 * 64];
    bv_int4_epilogue_impl(weights, groups, input, bias, output, rows, columns, tokens, mode, group, simd, lane, shared);
}

// Fused Q/K/V INT4_G64 GEMM + bias for SigLIP2 & vit_merger self-attention (3x fewer dispatches):
kernel void bv_int4_matmul_qkv_bias(
    device const uchar *wq [[buffer(0)]], device const half2 *gq [[buffer(1)]], device const float *bq [[buffer(2)]],
    device const uchar *wk [[buffer(3)]], device const half2 *gk [[buffer(4)]], device const float *bk [[buffer(5)]],
    device const uchar *wv [[buffer(6)]], device const half2 *gv [[buffer(7)]], device const float *bv [[buffer(8)]],
    device const float *input [[buffer(9)]],
    device float *out_q [[buffer(10)]], device float *out_k [[buffer(11)]], device float *out_v [[buffer(12)]],
    constant uint &rows [[buffer(13)]], constant uint &columns [[buffer(14)]], constant uint &tokens [[buffer(15)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[2 * 8 * 64];
    const uint g_cnt = (rows + 15) / 16;
    if (group.x < g_cnt) {
        bv_int4_epilogue_impl(wq, gq, input, bq, out_q, rows, columns, tokens, 0, group, simd, lane, shared);
    } else if (group.x < 2 * g_cnt) {
        bv_int4_epilogue_impl(wk, gk, input, bk, out_k, rows, columns, tokens, 0, uint2(group.x - g_cnt, group.y), simd, lane, shared);
    } else {
        bv_int4_epilogue_impl(wv, gv, input, bv, out_v, rows, columns, tokens, 0, uint2(group.x - 2 * g_cnt, group.y), simd, lane, shared);
    }
}

// Add 70x70 BF16 position embeddings (`floor(70.0 * i / cur_h)` bucket lookup) directly on GPU!
// Eliminates 22.58 MB `positions_f32` CPU heap allocation and removes the CPU-stalling `commit_and_wait` after patch_embedding.
kernel void bv_add_positions_bf16(
    device float *hidden [[buffer(0)]],
    device const ushort *pos_bf16 [[buffer(1)]],
    constant uint &cur_h [[buffer(2)]],
    constant uint &cur_w [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint count = cur_h * cur_w;
    if (gid >= count * BV_WIDTH) return;
    const uint patch = gid / BV_WIDTH;
    const uint c = gid % BV_WIDTH;
    const uint py = patch / cur_w;
    const uint px = patch % cur_w;
    const uint sy = (70u * py) / cur_h;
    const uint sx = (70u * px) / cur_w;
    const ushort bits = pos_bf16[ulong(sy * 70u + sx) * BV_WIDTH + c];
    hidden[gid] += as_type<float>(uint(bits) << 16);
}
