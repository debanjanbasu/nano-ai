// Core operations for MiniCPM-V 4.7 1B language backbone (Qwen3.5-0.8B 24-layer hybrid):
// - Width = 1024, FFN = 3584, Vocab = 248,094
// - 18 Gated DeltaNet recurrent layers: 16 K heads x 128, 16 V heads x 128, conv1d [6144, 4]
// - 6 Full-Attention layers: 8 Q heads x 256 (with gate), 2 KV heads x 256, partial_rotary_factor=0.25 (64 dims, M-RoPE 11/11/10)
// - Rotated Q8 KV cache (272 B/token/layer) and F16 KV cache (1024 B/token/layer)
#include "common.h"

constant constexpr uint MC_WIDTH = 1024;
constant constexpr uint MC_GDN_HEADS = 16;
constant constexpr uint MC_CONV_CHANNELS = 6144;
constant constexpr uint MC_GDN_OUT = 2048;

// ----------------------------------------------------------------------------
// 1. RMSNorm, Fused Residual-Add + RMSNorm (+ optional 512-block FWHT), SwiGLU
// ----------------------------------------------------------------------------

kernel void mc_rms(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dimension [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &stride [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    if (row >= rows) return;
    float square_sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float value = input[ulong(row) * stride + i];
        square_sum = fma(value, value, square_sum);
    }
    const float sum = nano_group_sum<128>(square_sum, tid, partial);
    const float inverse = rsqrt(sum / float(dimension) + epsilon);
    for (uint i = tid; i < dimension; i += 128) {
        const ulong index = ulong(row) * stride + i;
        output[index] = input[index] * inverse * weights[i];
    }
}

// Fused RMSNorm + 512-block Signed Hadamard forward rotation for 1024-wide rows (`2 * 512`).
// Eliminates a separate `fwht512_forward` dispatch and global-memory round-trip before `lm_head`!
kernel void mc_rms_fwht512(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &dimension [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    threadgroup float shared[512];
    if (row >= rows) return;
    const ulong row_base = ulong(row) * dimension;
    float square_sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float value = input[row_base + i];
        square_sum = fma(value, value, square_sum);
    }
    const float sum = nano_group_sum<128>(square_sum, tid, partial);
    const float inverse = rsqrt(sum / float(dimension) + epsilon);
    const uint blocks_per_row = dimension / 512;
    if (tid < 64) {
        for (uint b = 0; b < blocks_per_row; ++b) {
            const uint block = row * blocks_per_row + b;
            float values[8];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                const uint col = b * 512 + i * 64 + tid;
                values[i] = input[row_base + col] * inverse * weights[col];
            }
            fwht_512_values<false>(values, signs, output, blocks_per_row, block, tid, shared);
        }
    }
}

kernel void mc_add(
    device const float *input [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < count) output[i] = input[i] + residual[i];
}

// Fused residual add (`hidden + branch` -> `sum_out`) + RMSNorm (`normalized`)
// One 128-thread threadgroup per 1024-wide row (8 elements per thread, all in registers).
kernel void mc_add_rms(
    device const float *hidden [[buffer(0)]],
    device const float *branch [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float *sum_out [[buffer(3)]],
    device float *normalized [[buffer(4)]],
    constant uint &dimension [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device const float *row_h = hidden + ulong(row) * dimension;
    device const float *row_b = branch + ulong(row) * dimension;
    device float *row_s = sum_out + ulong(row) * dimension;
    device float *row_n = normalized + ulong(row) * dimension;
    float square_sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float s = row_h[i] + row_b[i];
        row_s[i] = s;
        square_sum = fma(s, s, square_sum);
    }
    const float sum = nano_group_sum<128>(square_sum, tid, partial);
    const float inverse = rsqrt(sum / float(dimension) + epsilon);
    for (uint i = tid; i < dimension; i += 128) {
        row_n[i] = row_s[i] * inverse * weights[i];
    }
}

// Fused residual add (`hidden + branch` -> `sum_out`) + RMSNorm + 512-block Signed Hadamard forward rotation (`output`).
kernel void mc_add_rms_fwht512(
    device const float *hidden [[buffer(0)]],
    device const float *branch [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device const float *signs [[buffer(3)]],
    device float *sum_out [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant uint &dimension [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant float &epsilon [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    threadgroup float shared[512];
    if (row >= rows) return;
    const ulong row_base = ulong(row) * dimension;
    float square_sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float s = hidden[row_base + i] + branch[row_base + i];
        sum_out[row_base + i] = s;
        square_sum = fma(s, s, square_sum);
    }
    const float sum = nano_group_sum<128>(square_sum, tid, partial);
    const float inverse = rsqrt(sum / float(dimension) + epsilon);
    const uint blocks_per_row = dimension / 512;
    if (tid < 64) {
        for (uint b = 0; b < blocks_per_row; ++b) {
            const uint block = row * blocks_per_row + b;
            float values[8];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                const uint col = b * 512 + i * 64 + tid;
                values[i] = sum_out[row_base + col] * inverse * weights[col];
            }
            fwht_512_values<false>(values, signs, output, blocks_per_row, block, tid, shared);
        }
    }
}

// Fused residual add (`hidden + branch` -> `sum_out`) + optional tap copy + RMSNorm (`normalized`)
// When `tap_slot < 5`, also copies `sum_out` row into `taps[(row * 5 + tap_slot) * dimension]`.
kernel void mc_add_rms_tap(
    device const float *hidden [[buffer(0)]],
    device const float *branch [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float *sum_out [[buffer(3)]],
    device float *normalized [[buffer(4)]],
    device float *taps [[buffer(5)]],
    constant uint &dimension [[buffer(6)]],
    constant float &epsilon [[buffer(7)]],
    constant uint &tap_slot [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device const float *row_h = hidden + ulong(row) * dimension;
    device const float *row_b = branch + ulong(row) * dimension;
    device float *row_s = sum_out + ulong(row) * dimension;
    device float *row_n = normalized + ulong(row) * dimension;
    device float *row_t = taps + (ulong(row) * 5 + tap_slot) * dimension;
    const bool write_tap = tap_slot < 5;
    float square_sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float s = row_h[i] + row_b[i];
        row_s[i] = s;
        if (write_tap) row_t[i] = s;
        square_sum = fma(s, s, square_sum);
    }
    const float sum = nano_group_sum<128>(square_sum, tid, partial);
    const float inverse = rsqrt(sum / float(dimension) + epsilon);
    for (uint i = tid; i < dimension; i += 128) {
        row_n[i] = row_s[i] * inverse * weights[i];
    }
}

kernel void mc_swiglu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < count) output[i] = nano_silu(gate[i]) * up[i];
}

// ----------------------------------------------------------------------------
// 2. Gated DeltaNet (GDN) Convolution, Q/K L2 Norm, Decay/Beta, Recurrence
// ----------------------------------------------------------------------------

// 6144-channel causal conv1d (kernel=4, history=3) + SiLU
static inline void mc_conv_impl(
    device const float *input,
    device const float *weights,
    device const float *history,
    device float *output,
    device float *final_history,
    uint tokens,
    uint commit_tokens,
    uint channel
) {
    if (channel >= MC_CONV_CHANNELS) return;
    const uint h = channel * 3, w = channel * 4;
    float h0 = history[h], h1 = history[h + 1], h2 = history[h + 2];
    for (uint token = 0; token < tokens; ++token) {
        const uint index = token * MC_CONV_CHANNELS + channel;
        float sum = weights[w] * h0;
        sum = fma(weights[w + 1], h1, sum);
        sum = fma(weights[w + 2], h2, sum);
        sum = fma(weights[w + 3], input[index], sum);
        h0 = h1;
        h1 = h2;
        h2 = input[index];
        output[index] = nano_silu(sum);
        if (token + 1 == commit_tokens) {
            final_history[h] = h0;
            final_history[h + 1] = h1;
            final_history[h + 2] = h2;
        }
    }
}

static inline void mc_l2_qk_impl(
    device float *qkv,
    float epsilon,
    uint head,
    uint tid,
    threadgroup float *partial
) {
    // 32 Q/K heads of 128 each = first 4096 floats of the 6144-wide row
    const uint index = (head / 32) * MC_CONV_CHANNELS + (head % 32) * 128 + tid;
    const float value = qkv[index];
    const float sum = nano_group_sum<128>(value * value, tid, partial);
    qkv[index] = value / max(sqrt(sum), epsilon);
}

static inline void mc_decay_impl(
    device const float *a,
    device const float *alpha,
    device const float *dt,
    device const float *raw_beta,
    device float *decay,
    device float *beta,
    uint tokens,
    uint head
) {
    if (head >= MC_GDN_HEADS * tokens) return;
    const uint h = head % MC_GDN_HEADS;
    const float x = alpha[head] + dt[h];
    const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
    decay[head] = exp(a[h] * softplus);
    beta[head] = nano_sigmoid(raw_beta[head]);
}

// Fused conv + L2 Q/K norm + decay/beta over a block of tokens.
// Threadgroups 0..47 convolve 128 channels each (48 * 128 = 6144).
// Threadgroups 0..31 then L2-normalize their Q/K head for each token.
// Threadgroup 48 computes decay and beta for all 16 heads * tokens.
kernel void mc_conv_l2_decay(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *history [[buffer(2)]],
    device float *output [[buffer(3)]],
    device const float *a [[buffer(4)]],
    device const float *alpha [[buffer(5)]],
    device const float *dt [[buffer(6)]],
    device const float *raw_beta [[buffer(7)]],
    device float *decay [[buffer(8)]],
    device float *beta [[buffer(9)]],
    device float *final_history [[buffer(10)]],
    constant uint &tokens [[buffer(11)]],
    constant float &epsilon [[buffer(12)]],
    constant uint &commit_tokens [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (group == 48) {
        for (uint head = tid; head < MC_GDN_HEADS * tokens; head += 128) {
            mc_decay_impl(a, alpha, dt, raw_beta, decay, beta, tokens, head);
        }
        return;
    }
    if (tokens <= 8) {
        threadgroup float partial8[8][4];
        const uint channel = group * 128 + tid;
        float conv_vals[8];
        if (channel < MC_CONV_CHANNELS) {
            const uint h = channel * 3, w = channel * 4;
            float h0 = history[h], h1 = history[h + 1], h2 = history[h + 2];
            for (uint token = 0; token < tokens; ++token) {
                const uint index = token * MC_CONV_CHANNELS + channel;
                const float x = input[index];
                float sum = weights[w] * h0;
                sum = fma(weights[w + 1], h1, sum);
                sum = fma(weights[w + 2], h2, sum);
                sum = fma(weights[w + 3], x, sum);
                h0 = h1;
                h1 = h2;
                h2 = x;
                const float c = nano_silu(sum);
                conv_vals[token] = c;
                if (group >= 32) {
                    output[index] = c;
                }
                if (token + 1 == commit_tokens) {
                    final_history[h] = h0;
                    final_history[h + 1] = h1;
                    final_history[h + 2] = h2;
                }
            }
        }
        if (group >= 32) return;
        const uint simd = tid / 32;
        for (uint token = 0; token < tokens; ++token) {
            const float v = conv_vals[token];
            const float s = simd_sum(v * v);
            if (tid % 32 == 0) partial8[token][simd] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint token = 0; token < tokens; ++token) {
            const float sum = ((partial8[token][0] + partial8[token][1]) + partial8[token][2]) + partial8[token][3];
            output[token * MC_CONV_CHANNELS + channel] = conv_vals[token] / max(sqrt(sum), epsilon);
        }
        return;
    }
    threadgroup float partial[4];
    mc_conv_impl(input, weights, history, output, final_history, tokens, commit_tokens, group * 128 + tid);
    if (group >= 32) return;
    for (uint token = 0; token < tokens; ++token) {
        mc_l2_qk_impl(output, epsilon, token * 32 + group, tid, partial);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// GDN recurrence: 16 heads (1:1 key:value heads), state [16 * 128][128] stored as F16 (`half`).
// Arithmetic runs in F32 registers; state rounds to F16 once per 128 tokens and at block end.
static inline void mc_store_state(device half &state, float value) {
    state = half(clamp(value, -65504.0f, 65504.0f));
}

static inline float mc_round_state(float value) {
    return float(half(clamp(value, -65504.0f, 65504.0f)));
}

// Single-row or multi-row GDN recurrence, 4 value rows per SIMD group (512 SIMD groups total).
// Software-pipelined: each token's Q, K, V, decay, and beta are prefetched from device memory
// while the previous token's recurrence chain and simd_sum reductions execute on the ALU.
kernel void mc_gdn_rows_4_f16(
    device const float *qkv [[buffer(0)]],
    device const float *decay [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device const half *initial_state [[buffer(3)]],
    device float *output [[buffer(4)]],
    device half *final_state [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    constant uint &commit_tokens [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tokens == 0) return;
    constexpr uint rows = 4;
    const uint first_row = group * rows;
    const uint head = first_row / 128; // 0..15, key_group == head (16 K heads, 16 V heads)
    float values[rows][4];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < rows; ++r) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            values[r][i] = float(initial_state[(first_row + r) * 128 + lane + i * 32]);
        }
    }

    float next_q[4], next_k[4], next_v[rows], next_d, next_b;
    auto load = [&](uint token) {
        const uint token_offset = token * MC_CONV_CHANNELS;
        next_d = decay[token * MC_GDN_HEADS + head];
        next_b = beta[token * MC_GDN_HEADS + head];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint column = head * 128 + lane + i * 32;
            next_q[i] = qkv[token_offset + column];
            next_k[i] = qkv[token_offset + 2048 + column];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < rows; ++r) {
            next_v[r] = qkv[token_offset + 4096 + first_row + r];
        }
    };

    load(0);
    for (uint token = 0; token < tokens; ++token) {
        float q[4], k[4], v[rows];
        const float d = next_d, b = next_b;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            q[i] = next_q[i];
            k[i] = next_k[i];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < rows; ++r) {
            v[r] = next_v[r];
        }
        if (token + 1 < tokens) {
            load(token + 1);
        }

        float prediction[rows] = {};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < rows; ++r) {
                values[r][i] *= d;
                prediction[r] = fma(values[r][i], k[i], prediction[r]);
            }
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < rows; ++r) {
            const float correction = (v[r] - simd_sum(prediction[r])) * b;
            float result = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                values[r][i] = fma(k[i], correction, values[r][i]);
                result = fma(values[r][i], q[i], result);
            }
            result = simd_sum(result);
            if (lane == 0) output[token * MC_GDN_OUT + first_row + r] = result * 0.08838834764831845f; // 1 / sqrt(128)
        }
        if (token + 1 == commit_tokens) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < rows; ++r) {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    mc_store_state(final_state[(first_row + r) * 128 + lane + i * 32], values[r][i]);
                }
            }
        }
        if ((token + 1) % 128 == 0 && token + 1 < tokens) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < rows; ++r) {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    values[r][i] = mc_round_state(values[r][i]);
                }
            }
        }
    }
}

// Shift `kept_tokens` (`1..=8`) rows of raw `qkv` (`[kept_tokens, 6144]`) into `final_history` (`[6144 * 3]`)
// starting from `initial_history` (`[6144 * 3]`), without re-running conv1d, L2 norm, or decay!
kernel void mc_conv_history_shift(
    device const float *qkv [[buffer(0)]],
    device const float *initial_history [[buffer(1)]],
    device float *final_history [[buffer(2)]],
    constant uint &kept_tokens [[buffer(3)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= MC_CONV_CHANNELS) return;
    const uint h = channel * 3;
    float h0 = initial_history[h];
    float h1 = initial_history[h + 1];
    float h2 = initial_history[h + 2];
    for (uint t = 0; t < kept_tokens; ++t) {
        h0 = h1;
        h1 = h2;
        h2 = qkv[t * MC_CONV_CHANNELS + channel];
    }
    final_history[h] = h0;
    final_history[h + 1] = h1;
    final_history[h + 2] = h2;
}

// State-only GDN recurrence for partial speculative acceptance replay:
// skips `q` loads, dot-product with `q`, `simd_sum(result)`, and global `output` stores.
kernel void mc_gdn_state_only_f16(
    device const float *qkv [[buffer(0)]],
    device const float *decay [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device const half *initial_state [[buffer(3)]],
    device half *final_state [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tokens == 0) return;
    constexpr uint rows = 4;
    const uint first_row = group * rows;
    const uint head = first_row / 128;
    float values[rows][4];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < rows; ++r) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            values[r][i] = float(initial_state[(first_row + r) * 128 + lane + i * 32]);
        }
    }

    float next_k[4], next_v[rows], next_d, next_b;
    auto load = [&](uint token) {
        const uint token_offset = token * MC_CONV_CHANNELS;
        next_d = decay[token * MC_GDN_HEADS + head];
        next_b = beta[token * MC_GDN_HEADS + head];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint column = head * 128 + lane + i * 32;
            next_k[i] = qkv[token_offset + 2048 + column];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < rows; ++r) {
            next_v[r] = qkv[token_offset + 4096 + first_row + r];
        }
    };

    load(0);
    for (uint token = 0; token < tokens; ++token) {
        float k[4], v[rows];
        const float d = next_d, b = next_b;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            k[i] = next_k[i];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < rows; ++r) {
            v[r] = next_v[r];
        }
        if (token + 1 < tokens) {
            load(token + 1);
        }

        float prediction[rows] = {};
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < rows; ++r) {
                values[r][i] *= d;
                prediction[r] = fma(values[r][i], k[i], prediction[r]);
            }
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < rows; ++r) {
            const float correction = (v[r] - simd_sum(prediction[r])) * b;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                values[r][i] = fma(k[i], correction, values[r][i]);
            }
        }
        if ((token + 1) % 128 == 0 && token + 1 < tokens) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < rows; ++r) {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    values[r][i] = mc_round_state(values[r][i]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < rows; ++r) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            mc_store_state(final_state[(first_row + r) * 128 + lane + i * 32], values[r][i]);
        }
    }
}

// GDN output RMSNorm (per 128-d head) + SiLU gate (`z`).
// Since num_key_heads == num_value_heads == 16, head regrouping is identity.
// Dispatch 16 * tokens threadgroups of 128 threads.
kernel void mc_gdn_post(
    device const float *input [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    const uint index = head * 128 + tid;
    const float value = input[index];
    const float sum = nano_group_sum<128>(value * value, tid, partial);
    output[index] = value * rsqrt(sum / 128.0f + epsilon) * weights[tid] * nano_silu(gate[index]);
}

// ----------------------------------------------------------------------------
// 3. Full-Attention Prep (Q/K RMSNorm + Canvas 3D M-RoPE + Q8/F16 KV Store)
// ----------------------------------------------------------------------------
// Full-attention in MiniCPM-V 4.7 1B:
// - 8 Q heads x 256 (q_proj outputs 4096: per head 256 Q + 256 gate)
// - 2 KV heads x 256 (k_proj 512, v_proj 512)
// - partial_rotary_factor = 0.25 -> first 64 dims (32 pairs), interleaved M-RoPE sections [11, 11, 10]
// - KV cache token row: 2 KV heads x 256 dims:
//   - F16 (format 0): 1024 B/token (512 halves)
//   - Q8  (format 1): 544 B/token  (512 int8 + 16 half scales, Hadamard-256 rotated per head)
//   - Q4  (format 2): 288 B/token  (256 packed int4 nibbles + 16 half scales, Hadamard-256 rotated per head)

constant constexpr uint MC_KV_F16 = 0;
constant constexpr uint MC_KV_Q8 = 1;
constant constexpr uint MC_KV_Q4 = 2;
constant constexpr uint MC_KV_Q8_TOKEN_BYTES = 2 * 256 + 2 * 8 * 2; // 544 bytes
constant constexpr uint MC_KV_Q4_TOKEN_BYTES = 2 * 128 + 2 * 8 * 2; // 288 bytes
static inline float mc_text_rope(
    device const float *input,
    device const float *weights,
    float inverse,
    uint coordinate,
    uint3 position,
    float base
) {
    const float value = input[coordinate] * inverse * weights[coordinate];
    if (coordinate >= 64) return value;
    const uint pair = coordinate % 32, other = coordinate < 32 ? coordinate + 32 : coordinate - 32;
    const float other_value = input[other] * inverse * weights[other];
    const uint section = pair % 3;
    const uint rotation = section == 1 ? position.y : (section == 2 ? position.z : position.x);
    const float theta = float(rotation) * pow(base, -float(pair) / 32.0f);
    return coordinate < 32 ? value * cos(theta) - other_value * sin(theta)
                           : other_value * sin(theta) + value * cos(theta);
}

static inline uint3 mc_rope_position(
    device const uint4 *rope_rows, uint rope_mode, uint rope_delta, uint position, uint token
) {
    if (rope_mode != 0) return rope_rows[token].xyz;
    return uint3(uint(int(position + token) + as_type<int>(rope_delta)));
}

static inline void mc_kv_store(
    device uchar *cache, uint format, ulong token, uint head, uint block, uint lane, float value
) {
    if (format == MC_KV_F16) {
        device half *rows = reinterpret_cast<device half *>(cache);
        rows[(token * 2 + head) * 256 + block * 32 + lane] = half(value);
        return;
    }
    const float peak = simd_max(abs(value));
    if (format == MC_KV_Q8) {
        device uchar *row = cache + token * MC_KV_Q8_TOKEN_BYTES;
        const half scale = half(min(peak / 127.0f, 65504.0f));
        const float inverse = scale > half(0.0f) ? 1.0f / float(scale) : 0.0f;
        reinterpret_cast<device char *>(row)[head * 256 + block * 32 + lane] =
            char(clamp(rint(value * inverse), -127.0f, 127.0f));
        if (lane == 0) reinterpret_cast<device half *>(row + 512)[head * 8 + block] = scale;
        return;
    }
    if (format == MC_KV_Q4) {
        device uchar *row = cache + token * MC_KV_Q4_TOKEN_BYTES;
        const half scale = half(min(peak / 7.0f, 65504.0f));
        const float inverse = scale > half(0.0f) ? 1.0f / float(scale) : 0.0f;
        const uint q_own = uint(clamp(rint(value * inverse) + 8.0f, 0.0f, 15.0f));
        const uint q_pair = simd_shuffle_xor(q_own, 1u);
        if ((lane & 1u) == 0u) {
            row[head * 128 + block * 16 + (lane >> 1)] = uchar(q_own | (q_pair << 4));
        }
        if (lane == 0) reinterpret_cast<device half *>(row + 256)[head * 8 + block] = scale;
    }
}

// Dispatch 8 * tokens threadgroups of 256 threads.
kernel void mc_attn_prep(
    device const float *qg [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *q_weights [[buffer(3)]],
    device const float *k_weights [[buffer(4)]],
    device float *query [[buffer(5)]],
    device float *gate [[buffer(6)]],
    device uchar *k_cache [[buffer(7)]],
    device uchar *v_cache [[buffer(8)]],
    device const uint4 *rope_rows [[buffer(9)]],
    constant uint &position [[buffer(10)]],
    constant uint &kv_format [[buffer(11)]],
    constant uint &rope_mode [[buffer(12)]],
    constant uint &rope_delta [[buffer(13)]],
    constant float &epsilon [[buffer(14)]],
    constant float &base [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[8], scratch[256];
    const uint head = group % 8, token = group / 8, absolute_position = position + token;
    const uint3 rope = mc_rope_position(rope_rows, rope_mode, rope_delta, position, token);
    const bool rotate = kv_format != MC_KV_F16;
    device const float *qg_row = qg + token * 4096;
    device const float *key_row = key + token * 512;
    const float q = qg_row[head * 512 + tid];
    const float q_sum = nano_group_sum<256>(q * q, tid, partial);
    float roped = mc_text_rope(qg_row + head * 512, q_weights,
        rsqrt(q_sum / 256.0f + epsilon), tid, rope, base);
    if (rotate) roped = hadamard256(roped, tid, scratch);
    query[group * 256 + tid] = roped;
    gate[group * 256 + tid] = qg_row[head * 512 + 256 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (head < 2) {
        const float k = key_row[head * 256 + tid];
        const float k_sum = nano_group_sum<256>(k * k, tid, partial);
        float rotated = mc_text_rope(key_row + head * 256, k_weights,
            rsqrt(k_sum / 256.0f + epsilon), tid, rope, base);
        if (rotate) {
            rotated = hadamard256(rotated, tid, scratch);
        }
        mc_kv_store(k_cache, kv_format, absolute_position, head, tid / 32, tid % 32, rotated);
    } else if (head < 4) {
        const uint kv_head = head - 2;
        float v = value[token * 512 + kv_head * 256 + tid];
        if (rotate) {
            v = hadamard256(v, tid, scratch);
        }
        mc_kv_store(v_cache, kv_format, absolute_position, kv_head, tid / 32, tid % 32, v);
    }
}

// ----------------------------------------------------------------------------
// 4. Full attention: GQA-packed flash attention (decode, speculative verify, prefill)
// ----------------------------------------------------------------------------
// 8 Q heads share 2 KV heads (4 Q heads per KV head). A threadgroup serves one KV head, a tile
// of query rows and one key split. Each SIMD group owns one 8-row fragment of packed
// (query row, head) pairs (2 query rows x the 4 heads sharing the KV head), so every K/V tile
// staged in threadgroup memory feeds all of them. Q.K^T and P.V run on 8x8 simdgroup matrices
// (half operands, float accumulators) under an online softmax in the exp2 domain.

constant constexpr uint MC_FLASH_KEYS = 16;        // keys per staged K/V tile
constant constexpr uint MC_FLASH_STRIDE = 256 + 8; // padded tile row (halves) against bank conflicts

// Dequantize keys [first, first + 16) of `kv_head` into a [16, 256] half tile; keys >= `limit` are 0.
template <uint FORMAT>
static inline void mc_flash_stage(
    threadgroup half *tile, device const uchar *cache, uint kv_head, uint first, uint limit,
    uint tid, uint threads
) {
    for (uint item = tid; item < MC_FLASH_KEYS * 8; item += threads) {
        const uint k = item / 8, block = item % 8, token = first + k;
        threadgroup half4 *dst = reinterpret_cast<threadgroup half4 *>(tile + k * MC_FLASH_STRIDE + block * 32);
        if (token >= limit) {
            for (uint i = 0; i < 8; ++i) dst[i] = half4(0.0h);
        } else if (FORMAT == MC_KV_F16) {
            device const half4 *src = reinterpret_cast<device const half4 *>(cache)
                + ((ulong(token) * 2 + kv_head) * 256 + block * 32) / 4;
            for (uint i = 0; i < 8; ++i) dst[i] = src[i];
        } else if (FORMAT == MC_KV_Q8) {
            device const uchar *row = cache + ulong(token) * MC_KV_Q8_TOKEN_BYTES;
            const float scale = float(reinterpret_cast<device const half *>(row + 512)[kv_head * 8 + block]);
            device const char4 *src = reinterpret_cast<device const char4 *>(row + kv_head * 256 + block * 32);
            for (uint i = 0; i < 8; ++i) dst[i] = half4(float4(src[i]) * scale);
        } else {
            device const uchar *row = cache + ulong(token) * MC_KV_Q4_TOKEN_BYTES;
            const float scale = float(reinterpret_cast<device const half *>(row + 256)[kv_head * 8 + block]);
            device const uchar4 *src = reinterpret_cast<device const uchar4 *>(row + kv_head * 128 + block * 16);
            for (uint i = 0; i < 4; ++i) {
                // Byte j packs dims 2j (low nibble) and 2j + 1 (high nibble).
                const int4 b = int4(uint4(src[i]));
                const float4 lo = float4((b & 15) - 8) * scale, hi = float4((b >> 4) - 8) * scale;
                dst[2 * i] = half4(lo.x, hi.x, lo.y, hi.y);
                dst[2 * i + 1] = half4(lo.z, hi.z, lo.w, hi.w);
            }
        }
    }
}

// Row `row` sees keys < position + row + 1; split `s` covers keys [s * split_keys, (s + 1) * split_keys).
// Writes partials `[tokens * 8 heads, splits, 258]` = (max, denominator, numerator[256]) for `mc_attn_reduce`.
template <uint FORMAT>
static inline void mc_attn_flash_impl(
    device const float *query, device const uchar *key, device const uchar *value,
    device float *partials, uint position, uint tokens, uint splits, uint split_keys,
    uint3 group, uint simds, uint simd, uint lane, threadgroup half *k_tile, threadgroup half *v_tile
) {
    const uint kv_head = group.y, split = group.z, tid = simd * 32 + lane, threads = simds * 32;
    const uint row0 = group.x * simds * 2, row_end = min(row0 + simds * 2, tokens);
    const uint first = split * split_keys, last = min(first + split_keys, position + row_end);
    // This thread's element pair in every 8x8 fragment: row `fm`, columns `fn` and `fn + 1`.
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + (lane / 2) % 4, fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint packed = simd * 8 + fm, row = row0 + packed / 4, head = kv_head * 4 + packed % 4;
    const bool active = row < tokens;
    const uint visible = min(position + row + 1, last);

    // Q pre-scaled by log2(e) / sqrt(256) so the softmax runs on exp2.
    simdgroup_half8x8 q[32];
    device const float2 *q_row = reinterpret_cast<device const float2 *>(query + (ulong(row) * 8 + head) * 256 + fn);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 32; ++j) {
        const float2 v = active ? q_row[j * 4] * (M_LOG2E_F / 16.0f) : float2(0.0f);
        q[j].thread_elements()[0] = half(v.x);
        q[j].thread_elements()[1] = half(v.y);
    }
    simdgroup_float8x8 o[32];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 32; ++j) o[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float maximum = -INFINITY, denominator = 0.0f;

    for (uint start = first; start < last; start += MC_FLASH_KEYS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mc_flash_stage<FORMAT>(k_tile, key, kv_head, start, last, tid, threads);
        mc_flash_stage<FORMAT>(v_tile, value, kv_head, start, last, tid, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s[2] = {
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f), make_filled_simdgroup_matrix<float, 8, 8>(0.0f)
        };
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) {
            #pragma clang loop unroll(full)
            for (uint h = 0; h < 2; ++h) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, k_tile + h * 8 * MC_FLASH_STRIDE + j * 8, MC_FLASH_STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[h], q[j], kt, s[h]);
            }
        }
        // Causal and split mask, then the online-softmax update of this thread's row (4 lanes share it).
        float2 score[2];
        float tile_max = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint h = 0; h < 2; ++h) {
            const uint k0 = start + h * 8 + fn;
            score[h] = float2(k0 < visible ? s[h].thread_elements()[0] : -INFINITY,
                k0 + 1 < visible ? s[h].thread_elements()[1] : -INFINITY);
            tile_max = max(tile_max, max(score[h].x, score[h].y));
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float next = max(maximum, tile_max);
        const float shift = next == -INFINITY ? 0.0f : next; // no visible key yet: weights stay 0
        const float rescale = exp2(maximum - shift);
        float tile_sum = 0.0f;
        simdgroup_half8x8 p[2];
        #pragma clang loop unroll(full)
        for (uint h = 0; h < 2; ++h) {
            const float2 w = exp2(score[h] - shift);
            tile_sum += w.x + w.y;
            p[h].thread_elements()[0] = half(w.x);
            p[h].thread_elements()[1] = half(w.y);
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        denominator = fma(denominator, rescale, tile_sum);
        maximum = next;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) {
            o[j].thread_elements()[0] *= rescale;
            o[j].thread_elements()[1] *= rescale;
            #pragma clang loop unroll(full)
            for (uint h = 0; h < 2; ++h) {
                simdgroup_half8x8 v;
                simdgroup_load(v, v_tile + h * 8 * MC_FLASH_STRIDE + j * 8, MC_FLASH_STRIDE);
                simdgroup_multiply_accumulate(o[j], p[h], v, o[j]);
            }
        }
    }
    if (!active) return;
    // Splits past this row's causal prefix write an empty partial (weight 0 in the reduce).
    device float *out = partials + (ulong(row * 8 + head) * splits + split) * 258;
    if (fn == 0) {
        out[0] = maximum == -INFINITY ? -1e30f : maximum * M_LN2_F;
        out[1] = denominator;
    }
    device float2 *numerator = reinterpret_cast<device float2 *>(out + 2 + fn);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 32; ++j) {
        numerator[j * 4] = float2(o[j].thread_elements()[0], o[j].thread_elements()[1]);
    }
}

// Grid: (ceil(tokens / (2 * simds)), 2 KV heads, splits) threadgroups of `simds` (1..8) SIMD groups.
kernel void mc_attn_flash(
    device const float *query [[buffer(0)]],
    device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]],
    device float *partials [[buffer(3)]],
    constant uint &position [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &splits [[buffer(6)]],
    constant uint &split_keys [[buffer(7)]],
    constant uint &kv_format [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simds [[simdgroups_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup half4 tiles[2 * MC_FLASH_KEYS * MC_FLASH_STRIDE / 4];
    threadgroup half *k_tile = reinterpret_cast<threadgroup half *>(tiles);
    threadgroup half *v_tile = k_tile + MC_FLASH_KEYS * MC_FLASH_STRIDE;
    if (kv_format == MC_KV_F16) {
        mc_attn_flash_impl<MC_KV_F16>(query, key, value, partials, position, tokens, splits, split_keys,
            group, simds, simd, lane, k_tile, v_tile);
    } else if (kv_format == MC_KV_Q4) {
        mc_attn_flash_impl<MC_KV_Q4>(query, key, value, partials, position, tokens, splits, split_keys,
            group, simds, simd, lane, k_tile, v_tile);
    } else {
        mc_attn_flash_impl<MC_KV_Q8>(query, key, value, partials, position, tokens, splits, split_keys,
            group, simds, simd, lane, k_tile, v_tile);
    }
}

// Codes of 8 consecutive dims (`dim`, a multiple of 8) of `token`/`kv_head` as floats in `a`, `b`;
// returns the scale that turns them into values.
template <uint FORMAT>
static inline float mc_kv_codes8(
    device const uchar *cache, ulong token, uint kv_head, uint dim, thread float4 &a, thread float4 &b
) {
    if (FORMAT == MC_KV_F16) {
        device const half4 *src = reinterpret_cast<device const half4 *>(cache) + ((token * 2 + kv_head) * 256 + dim) / 4;
        a = float4(src[0]);
        b = float4(src[1]);
        return 1.0f;
    }
    if (FORMAT == MC_KV_Q8) {
        device const uchar *row = cache + token * MC_KV_Q8_TOKEN_BYTES;
        device const char4 *src = reinterpret_cast<device const char4 *>(row + kv_head * 256 + dim);
        a = float4(src[0]);
        b = float4(src[1]);
        return float(reinterpret_cast<device const half *>(row + 512)[kv_head * 8 + dim / 32]);
    }
    device const uchar *row = cache + token * MC_KV_Q4_TOKEN_BYTES;
    const int4 bytes = int4(uint4(*reinterpret_cast<device const uchar4 *>(row + kv_head * 128 + dim / 2)));
    const float4 lo = float4((bytes & 15) - 8), hi = float4((bytes >> 4) - 8);
    a = float4(lo.x, hi.x, lo.y, hi.y);
    b = float4(lo.z, hi.z, lo.w, hi.w);
    return float(reinterpret_cast<device const half *>(row + 256)[kv_head * 8 + dim / 32]);
}

// Single-row decode, the bandwidth-bound shape: one SIMD group serves a KV head's 4 query heads
// over one key split. Each lane scores its own key against all 4 heads (queries staged in
// threadgroup memory and read as broadcasts), then lanes switch to owning 8 output dims each
// and accumulate P.V over the chunk's 32 keys. Writes the same partials as `mc_attn_flash`.
template <uint FORMAT>
static inline void mc_attn_decode_impl(
    device const float *query, device const uchar *key, device const uchar *value,
    device float *partials, uint visible, uint splits, uint split_keys, uint split, uint kv_head,
    uint lane, threadgroup float4 *q4
) {
    for (uint i = lane; i < 256; i += 32) {
        q4[i] = reinterpret_cast<device const float4 *>(query + kv_head * 1024)[i] * (M_LOG2E_F / 16.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint first = split * split_keys, last = min(first + split_keys, visible);
    float maximum[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, denominator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o[4][2];
    for (uint h = 0; h < 4; ++h) o[h][0] = o[h][1] = float4(0.0f);
    for (uint chunk = first; chunk < last; chunk += 32) {
        const uint token = chunk + lane;
        float score[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        if (token < last) {
            float dot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint dim = 0; dim < 256; dim += 32) {
                float block[4] = {0.0f, 0.0f, 0.0f, 0.0f}, scale = 1.0f;
                for (uint part = 0; part < 32; part += 8) {
                    float4 a, b;
                    scale = mc_kv_codes8<FORMAT>(key, token, kv_head, dim + part, a, b);
                    for (uint h = 0; h < 4; ++h) {
                        block[h] += dot(a, q4[h * 64 + (dim + part) / 4]) + dot(b, q4[h * 64 + (dim + part) / 4 + 1]);
                    }
                }
                for (uint h = 0; h < 4; ++h) dot4[h] = fma(block[h], scale, dot4[h]);
            }
            for (uint h = 0; h < 4; ++h) score[h] = dot4[h];
        }
        float p[4];
        for (uint h = 0; h < 4; ++h) {
            const float next = max(maximum[h], simd_max(score[h])); // finite: lane 0 always has a key
            const float rescale = exp2(maximum[h] - next);
            p[h] = exp2(score[h] - next);
            denominator[h] = fma(denominator[h], rescale, simd_sum(p[h]));
            o[h][0] *= rescale;
            o[h][1] *= rescale;
            maximum[h] = next;
        }
        const uint count = min(32u, last - chunk);
        for (uint j = 0; j < count; ++j) {
            float4 a, b;
            const float scale = mc_kv_codes8<FORMAT>(value, chunk + j, kv_head, lane * 8, a, b);
            for (uint h = 0; h < 4; ++h) {
                const float weight = simd_shuffle(p[h], ushort(j)) * scale;
                o[h][0] = fma(a, weight, o[h][0]);
                o[h][1] = fma(b, weight, o[h][1]);
            }
        }
    }
    for (uint h = 0; h < 4; ++h) {
        device float *out = partials + (ulong(kv_head * 4 + h) * splits + split) * 258;
        if (lane == 0) {
            out[0] = maximum[h] == -INFINITY ? -1e30f : maximum[h] * M_LN2_F;
            out[1] = denominator[h];
        }
        device float2 *numerator = reinterpret_cast<device float2 *>(out + 2 + lane * 8);
        numerator[0] = o[h][0].xy;
        numerator[1] = o[h][0].zw;
        numerator[2] = o[h][1].xy;
        numerator[3] = o[h][1].zw;
    }
}

// Grid: (splits, 2 KV heads) threadgroups of one SIMD group; row 0 sees keys < visible.
kernel void mc_attn_decode(
    device const float *query [[buffer(0)]],
    device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]],
    device float *partials [[buffer(3)]],
    constant uint &visible [[buffer(4)]],
    constant uint &splits [[buffer(5)]],
    constant uint &split_keys [[buffer(6)]],
    constant uint &kv_format [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 q4[256];
    if (kv_format == MC_KV_F16) {
        mc_attn_decode_impl<MC_KV_F16>(query, key, value, partials, visible, splits, split_keys, group.x, group.y, lane, q4);
    } else if (kv_format == MC_KV_Q4) {
        mc_attn_decode_impl<MC_KV_Q4>(query, key, value, partials, visible, splits, split_keys, group.x, group.y, lane, q4);
    } else {
        mc_attn_decode_impl<MC_KV_Q8>(query, key, value, partials, visible, splits, split_keys, group.x, group.y, lane, q4);
    }
}
// Grid: `rows * 8` threadgroups (one per (row, head)) of 256 threads.
kernel void mc_attn_reduce(
    device const float *partials [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &splits [[buffer(3)]],
    constant uint &gated [[buffer(4)]],
    constant uint &rotated [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float scratch[256], scales[256], weights[256], maxima[8];
    device const float *base = partials + ulong(head) * splits * 258;
    float local = -1e30f;
    for (uint i = tid; i < splits; i += 256) local = max(local, base[ulong(i) * 258]);
    local = simd_max(local);
    if (lane == 0) maxima[simd] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    for (uint s = 1; s < 8; ++s) maximum = max(maximum, maxima[s]);
    float denominator = 0.0f, numerator = 0.0f;
    for (uint chunk = 0; chunk < splits; chunk += 256) {
        const uint count = min(256u, splits - chunk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < count) {
            device const float *part = base + ulong(chunk + tid) * 258;
            scales[tid] = exp(part[0] - maximum);
            weights[tid] = part[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float *column = base + ulong(chunk) * 258 + 2 + tid;
        for (uint j = 0; j < count; ++j) {
            denominator = fma(weights[j], scales[j], denominator);
            numerator = fma(column[ulong(j) * 258], scales[j], numerator);
        }
    }
    const uint index = head * 256 + tid;
    if (rotated) {
        const float value = hadamard256(numerator / denominator, tid, scratch);
        output[index] = value * (gated ? nano_sigmoid(gate[index]) : 1.0f);
    } else {
        output[index] = (numerator / denominator) * (gated ? nano_sigmoid(gate[index]) : 1.0f);
    }
}

// Copy `tokens` 1024-wide hidden rows into slot `tap_slot` (0..5) of `[tokens, 5120]` taps buffer.
kernel void mc_copy_tap(
    device const float *hidden [[buffer(0)]],
    device float *taps [[buffer(1)]],
    constant uint &tap_slot [[buffer(2)]],
    constant uint &tokens [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tokens * MC_WIDTH) return;
    const uint t = gid / MC_WIDTH, c = gid % MC_WIDTH;
    taps[(ulong(t) * 5 + tap_slot) * MC_WIDTH + c] = hidden[gid];
}
