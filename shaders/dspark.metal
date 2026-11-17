// DSpark 5-Layer Semi-Autoregressive Drafter Kernels for MiniCPM-V 4.7 1B:
// - Width = 1024, 5 full-attention layers (8 Q heads x 256, 2 KV heads x 256, FFN = 3584)
// - Tap projection `fc` [1024, 5120] from target layers [3, 7, 11, 15, 19], followed by `hidden_norm` [1024]
// - Non-causal block attention over `block_size` (up to 15) draft query tokens + context KV cache
// - Low-rank (r = 256) Markov head: `markov_w1[prev_token]` -> [256], `markov_w2` [vocab, 256] transition bias added to base logits
// - Confidence head: `σ(w^T [h_k; markov_w1[x_{k-1}]] + b)` -> prefix survival probability for hardware-aware prefix scheduling
#include "common.h"

constant constexpr uint DS_WIDTH = 1024;
constant constexpr uint DS_HEAD_DIM = 256;
constant constexpr uint DS_Q_HEADS = 8;
constant constexpr uint DS_KV_HEADS = 2;
constant constexpr uint DS_RANK = 256;
constant constexpr uint DS_MAX_SPLITS = 32;
constant constexpr uint DS_FULL_VOCAB = 248094;
constant constexpr uint DS_HEAD_CUT = 98304;
constant constexpr uint DS_SPECIAL_FIRST = 248044;
constant constexpr uint DS_SPECIAL_COUNT = 50;
constant constexpr uint DS_KV_Q8_TOKEN_BYTES = 2 * 256 + 2 * 8 * 2; // 544 bytes/token
// The drafter only attends to its last 1024 context keys plus its block, so its KV is a ring of
// this many positions (`DSPARK_KV_RING` in dspark.rs); position p lives in slot p % ring.
constant constexpr ulong DS_KV_RING = 2048;

static inline void ds_kv_store_q8(
    device uchar *cache, ulong token, uint kv_head, uint tid, float value
) {
    const uint block = tid / 32, lane = tid % 32;
    const float peak = simd_max(abs(value));
    device uchar *row = cache + (token % DS_KV_RING) * DS_KV_Q8_TOKEN_BYTES;
    const half scale = half(min(peak / 127.0f, 65504.0f));
    const float inv = scale > half(0.0f) ? 1.0f / float(scale) : 0.0f;
    reinterpret_cast<device char *>(row)[kv_head * 256 + block * 32 + lane] =
        char(clamp(rint(value * inv), -127.0f, 127.0f));
    if (lane == 0) reinterpret_cast<device half *>(row + 512)[kv_head * 8 + block] = scale;
}

static inline float ds_kv_load_q8(
    device const uchar *cache, ulong token, uint kv_head, uint dim
) {
    device const uchar *row = cache + (token % DS_KV_RING) * DS_KV_Q8_TOKEN_BYTES;
    const uint block = dim / 32;
    const float scale = float(reinterpret_cast<device const half *>(row + 512)[kv_head * 8 + block]);
    return float(reinterpret_cast<device const char *>(row)[kv_head * 256 + dim]) * scale;
}

// Fused Q/K per-head RMSNorm (256-d) + full 256-d RoPE + Q8 K/V cache store (`544 B/token`) for `draft_block`.
// Dispatch (8 + 2, tokens) threadgroups of 256 threads.
kernel void ds_qk_norm_rope_kv_store_f16(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_norm [[buffer(2)]],
    device const float *k_norm [[buffer(3)]],
    device const uint *positions [[buffer(4)]],
    device const float *v [[buffer(5)]],
    device uchar *k_cache [[buffer(6)]],
    device uchar *v_cache [[buffer(7)]],
    constant uint &start_pos [[buffer(8)]],
    constant float &epsilon [[buffer(9)]],
    constant float &rope_base [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]], // (head_slot 0..9, token)
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[8], shared[256];
    const uint slot = group.x, token = group.y;
    const uint pos = positions[token];
    const bool is_q = slot < DS_Q_HEADS;
    const uint kv_head = slot - DS_Q_HEADS;
    const ulong k_src_idx = (ulong(token) * DS_KV_HEADS + kv_head) * DS_HEAD_DIM;
    device float *vec = is_q
        ? (q + (ulong(token) * DS_Q_HEADS + slot) * DS_HEAD_DIM)
        : (k + k_src_idx);
    device const float *w = is_q ? q_norm : k_norm;
    const float val = vec[tid];
    const float sum = nano_group_sum<256>(val * val, tid, partial);
    const float normed = val * rsqrt(sum / 256.0f + epsilon) * w[tid];
    shared[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pair = tid % 128;
    const float other = tid < 128 ? -shared[tid + 128] : shared[tid - 128];
    const float angle = float(pos) * pow(rope_base, -float(2 * pair) / 256.0f);
    const float rot = normed * cos(angle) + other * sin(angle);
    vec[tid] = rot;
    if (!is_q) {
        const ulong abs_tok = ulong(start_pos + token);
        ds_kv_store_q8(k_cache, abs_tok, kv_head, tid, rot);
        ds_kv_store_q8(v_cache, abs_tok, kv_head, tid, v[k_src_idx + tid]);
    }
}

// Fused K-only RMSNorm (256-d) + full 256-d RoPE + Q8 K/V cache store (`544 B/token`) for `append_context`.
// Token row `token` sits at cache position (and RoPE position) `start_pos + token`.
// Dispatch (2, tokens) threadgroups of 256 threads.
kernel void ds_k_norm_rope_kv_store_f16(
    device const float *k [[buffer(0)]],
    device const float *v [[buffer(1)]],
    device const float *k_norm [[buffer(2)]],
    device uchar *k_cache [[buffer(3)]],
    device uchar *v_cache [[buffer(4)]],
    constant uint &start_pos [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    constant float &rope_base [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], // (kv_head 0..1, token)
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[8], shared[256];
    const uint kv_head = group.x, token = group.y;
    const uint pos = start_pos + token;
    const ulong src_idx = (ulong(token) * DS_KV_HEADS + kv_head) * DS_HEAD_DIM + tid;
    const float val = k[src_idx];
    const float sum = nano_group_sum<256>(val * val, tid, partial);
    const float normed = val * rsqrt(sum / 256.0f + epsilon) * k_norm[tid];
    shared[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pair = tid % 128;
    const float other = tid < 128 ? -shared[tid + 128] : shared[tid - 128];
    const float angle = float(pos) * pow(rope_base, -float(2 * pair) / 256.0f);
    const float rot_k = normed * cos(angle) + other * sin(angle);
    const ulong abs_tok = ulong(pos);
    ds_kv_store_q8(k_cache, abs_tok, kv_head, tid, rot_k);
    ds_kv_store_q8(v_cache, abs_tok, kv_head, tid, v[src_idx]);
}

// DSpark non-causal GQA block attention (short context `total_keys <= 128`):
kernel void ds_block_attention(
    device const float *q [[buffer(0)]],
    device const uchar *k_cache [[buffer(1)]],
    device const uchar *v_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &total_keys [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], // (query_idx, kv_head 0..1)
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float k_sh[DS_HEAD_DIM], v_sh[DS_HEAD_DIM];
    const uint qi = group.x, kv_head = group.y;
    const uint head = kv_head * 4 + simd;
    device const float *qp = q + (ulong(qi) * DS_Q_HEADS + head) * DS_HEAD_DIM;
    float qv[8], acc[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        qv[i] = qp[lane + i * 32] * (1.0f / 16.0f); // 1 / sqrt(256)
        acc[i] = 0.0f;
    }
    float m = -1e30f, l = 0.0f;
    for (uint kj = 0; kj < total_keys; ++kj) {
        k_sh[tid] = ds_kv_load_q8(k_cache, kj, kv_head, tid);
        k_sh[tid + 128] = ds_kv_load_q8(k_cache, kj, kv_head, tid + 128);
        v_sh[tid] = ds_kv_load_q8(v_cache, kj, kv_head, tid);
        v_sh[tid + 128] = ds_kv_load_q8(v_cache, kj, kv_head, tid + 128);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) s = fma(qv[i], k_sh[lane + i * 32], s);
        s = simd_sum(s);
        const float m_new = (kj == 0) ? s : max(m, s);
        const float corr = (kj == 0) ? 0.0f : exp(m - m_new);
        const float w = exp(s - m_new);
        l = fma(l, corr, w);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            acc[i] = fma(w, v_sh[lane + i * 32], acc[i] * corr);
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *op = out + (ulong(qi) * DS_Q_HEADS + head) * DS_HEAD_DIM;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        op[lane + i * 32] = acc[i] * inv;
    }
}

// DSpark Split-K Flash-Decoding Stage 1 over keys [first_key, total_keys):
kernel void ds_block_attention_split(
    device const float *q [[buffer(0)]],
    device const uchar *k_cache [[buffer(1)]],
    device const uchar *v_cache [[buffer(2)]],
    device float *attn_partials [[buffer(3)]],
    constant uint &total_keys [[buffer(4)]],
    constant uint &splits [[buffer(5)]],
    constant uint &first_key [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], // (split, qh_idx)
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint split = group.x, qh_idx = group.y;
    const uint head = qh_idx % DS_Q_HEADS;
    const uint kv_head = head / 4;
    device const float *qp = q + ulong(qh_idx) * DS_HEAD_DIM;
    float qv[8], acc[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        qv[i] = qp[lane + i * 32] * (1.0f / 16.0f); // 1 / sqrt(256)
        acc[i] = 0.0f;
    }
    const uint count = total_keys - first_key;
    const uint start = first_key + (count * split) / splits;
    const uint end = first_key + (count * (split + 1)) / splits;
    float m = -1e30f, l = 0.0f;
    for (uint kj = start; kj < end; ++kj) {
        float s = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            s = fma(qv[i], ds_kv_load_q8(k_cache, kj, kv_head, lane + i * 32), s);
        }
        s = simd_sum(s);
        const float m_new = (kj == start) ? s : max(m, s);
        const float corr = (kj == start) ? 0.0f : exp(m - m_new);
        const float w = exp(s - m_new);
        l = fma(l, corr, w);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            acc[i] = fma(w, ds_kv_load_q8(v_cache, kj, kv_head, lane + i * 32), acc[i] * corr);
        }
        m = m_new;
    }
    device float *part = attn_partials + (ulong(qh_idx) * DS_MAX_SPLITS + split) * 258;
    if (lane == 0) {
        part[0] = m;
        part[1] = l;
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        part[2 + lane + i * 32] = acc[i];
    }
}

// DSpark Split-K Flash-Decoding Stage 2:
// Grid: (block_tokens * DS_Q_HEADS, 1, 1) with 256 threads (1 thread per head dimension `tid = 0..255`).
kernel void ds_block_attention_reduce(
    device const float *attn_partials [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &splits [[buffer(2)]],
    uint qh_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float scales[DS_MAX_SPLITS], weights[DS_MAX_SPLITS], shared_max = -1e30f;
    device const float *base = attn_partials + ulong(qh_idx) * DS_MAX_SPLITS * 258;
    if (tid < 32) {
        const float local = (lane < splits) ? base[ulong(lane) * 258] : -1e30f;
        const float m = simd_max(local);
        if (lane == 0) shared_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = shared_max;
    if (tid < splits) {
        device const float *part = base + ulong(tid) * 258;
        scales[tid] = exp(part[0] - maximum);
        weights[tid] = part[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float num = 0.0f, den = 0.0f;
    device const float *col = base + 2 + tid;
    for (uint s = 0; s < splits; ++s) {
        const float sc = scales[s];
        den = fma(weights[s], sc, den);
        num = fma(col[ulong(s) * 258], sc, num);
    }
    out[ulong(qh_idx) * DS_HEAD_DIM + tid] = (den > 0.0f) ? (num / den) : 0.0f;
}

struct DsCandidate {
    uint id;
    float logit;
};

static inline int ds_total_key(float value) {
    int key = as_type<int>(value);
    return key ^ int(uint(key >> 31) >> 1);
}

static inline bool ds_better(float a, uint a_id, float b, uint b_id) {
    if (a_id == UINT_MAX) return false;
    if (b_id == UINT_MAX) return true;
    const int ak = ds_total_key(a), bk = ds_total_key(b);
    return ak > bk || (ak == bk && a_id < b_id);
}

static inline float ds_eval_w2_int4(
    device const uchar *wc,
    device const half2 *wg,
    threadgroup const float *m_shared,
    threadgroup const float *g_sums
) {
    float bias = 0.0f;
    #pragma clang loop unroll(full)
    for (uint g = 0; g < 4; ++g) {
        const float2 sb = float2(wg[g]);
        device const uint4 *p = reinterpret_cast<device const uint4 *>(wc + g * 32);
        const uint4 b0 = p[0], b1 = p[1];
        float d = 0.0f;
        #pragma clang loop unroll(full)
        for (uint q = 0; q < 4; ++q) {
            const uint base_c = g * 64 + q * 8;
            const uchar4 u0 = as_type<uchar4>(b0[q]);
            const float4 m0 = *reinterpret_cast<threadgroup const float4 *>(m_shared + base_c);
            const float4 m1 = *reinterpret_cast<threadgroup const float4 *>(m_shared + base_c + 4);
            d += float(u0.x & 0x0Fu) * m0.x + float(u0.x >> 4) * m0.y
               + float(u0.y & 0x0Fu) * m0.z + float(u0.y >> 4) * m0.w
               + float(u0.z & 0x0Fu) * m1.x + float(u0.z >> 4) * m1.y
               + float(u0.w & 0x0Fu) * m1.z + float(u0.w >> 4) * m1.w;

            const uint base_c2 = g * 64 + 32 + q * 8;
            const uchar4 u1 = as_type<uchar4>(b1[q]);
            const float4 m2 = *reinterpret_cast<threadgroup const float4 *>(m_shared + base_c2);
            const float4 m3 = *reinterpret_cast<threadgroup const float4 *>(m_shared + base_c2 + 4);
            d += float(u1.x & 0x0Fu) * m2.x + float(u1.x >> 4) * m2.y
               + float(u1.y & 0x0Fu) * m2.z + float(u1.y >> 4) * m2.w
               + float(u1.z & 0x0Fu) * m3.x + float(u1.z >> 4) * m3.y
               + float(u1.w & 0x0Fu) * m3.z + float(u1.w >> 4) * m3.w;
        }
        bias = fma(sb.x, d, fma(sb.y, g_sums[g], bias));
    }
    return bias;
}


// ============================================================================
// Kapitsa-style Two-Stage Top-K Candidate Extraction + Parallel Markov Walk
// (`nh_cand1` -> `nh_cand2` -> `nh_dsp_walk_dots` -> `nh_dsp_walk_pick`)
// Eliminates 7 sequential full-vocabulary `markov_w2` scans per draft pass!
// ============================================================================

constant constexpr uint DS_CAND_K = 16;
constant constexpr uint DS_CHUNK_VOCAB = 4096;

// Stage 1: For each (chunk, step), find the top-32 candidates in `base_logits[step, chunk*4096..(chunk+1)*4096]`
// (plus special tokens in chunk 0 when `vocab == DS_HEAD_CUT`).
// Grid: (nchunks, steps, 1), threadgroup: (256, 1, 1).
kernel void ds_cand_stage1(
    device const float *base_logits [[buffer(0)]],
    device const float *special_logits [[buffer(1)]],
    device DsCandidate *chunk_cands [[buffer(2)]],
    constant uint &vocab [[buffer(3)]],
    constant uint &nchunks [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], // (chunk, step)
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint s_ids[8];
    threadgroup float s_vals[8];
    const uint chunk = group.x, step = group.y;
    device const float *row = base_logits + ulong(step) * vocab;
    const uint base = chunk * DS_CHUNK_VOCAB + tid * 16;
    float v[16];
    uint id[16];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 16; ++j) {
        const uint idx = base + j;
        if (idx < vocab) {
            v[j] = row[idx];
            id[j] = idx;
        } else {
            v[j] = -1e30f;
            id[j] = UINT_MAX;
        }
    }
    if (vocab == DS_HEAD_CUT && chunk == 0 && tid < DS_SPECIAL_COUNT) {
        const float sv = special_logits[step * DS_SPECIAL_COUNT + tid];
        const uint sid = DS_SPECIAL_FIRST + tid;
        if (ds_better(sv, sid, v[15], id[15])) {
            v[15] = sv;
            id[15] = sid;
        }
    }
    device DsCandidate *out = chunk_cands + (ulong(step) * nchunks + chunk) * DS_CAND_K;
    for (uint r = 0; r < DS_CAND_K; ++r) {
        float bv = -1e30f;
        uint bi = UINT_MAX;
        uint b_slot = 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 16; ++j) {
            if (ds_better(v[j], id[j], bv, bi)) {
                bv = v[j];
                bi = id[j];
                b_slot = j;
            }
        }
        float wv = bv;
        uint wi = bi;
        for (ushort d = 16; d > 0; d >>= 1) {
            const float ov = simd_shuffle_xor(wv, d);
            const uint oi = simd_shuffle_xor(wi, d);
            if (ds_better(ov, oi, wv, wi)) {
                wv = ov;
                wi = oi;
            }
        }
        if (lane == 0) {
            s_vals[simd] = wv;
            s_ids[simd] = wi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float gv = s_vals[0];
            uint gi = s_ids[0];
            for (uint s = 1; s < 8; ++s) {
                if (ds_better(s_vals[s], s_ids[s], gv, gi)) {
                    gv = s_vals[s];
                    gi = s_ids[s];
                }
            }
            s_vals[0] = gv;
            s_ids[0] = gi;
            out[r] = DsCandidate{gi, gv};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (bi != UINT_MAX && bi == s_ids[0]) {
            v[b_slot] = -1e30f;
            id[b_slot] = UINT_MAX;
        }
    }
}

// Stage 2: Merge `nchunks` (<= 64) sorted top-32 candidate lists per step on a single 32-thread SIMD group!
// Grid: (steps, 1, 1), threadgroup: (32, 1, 1).
kernel void ds_cand_stage2(
    device const DsCandidate *chunk_cands [[buffer(0)]],
    device uint *cand_ids [[buffer(1)]],
    device float *cand_logits [[buffer(2)]],
    constant uint &nchunks [[buffer(3)]],
    uint step [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const DsCandidate *in_step = chunk_cands + ulong(step) * nchunks * DS_CAND_K;
    uint h0 = 0, h1 = 0;
    for (uint r = 0; r < DS_CAND_K; ++r) {
        float bv = -1e30f;
        uint bi = UINT_MAX;
        uint bc = UINT_MAX;
        if (lane < nchunks && h0 < DS_CAND_K) {
            const DsCandidate c0 = in_step[lane * DS_CAND_K + h0];
            bv = c0.logit;
            bi = c0.id;
            bc = lane;
        }
        if (lane + 32 < nchunks && h1 < DS_CAND_K) {
            const DsCandidate c1 = in_step[(lane + 32) * DS_CAND_K + h1];
            if (ds_better(c1.logit, c1.id, bv, bi)) {
                bv = c1.logit;
                bi = c1.id;
                bc = lane + 32;
            }
        }
        for (ushort d = 16; d > 0; d >>= 1) {
            const float ov = simd_shuffle_xor(bv, d);
            const uint oi = simd_shuffle_xor(bi, d);
            const uint oc = simd_shuffle_xor(bc, d);
            if (ds_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bc = oc;
            }
        }
        if (bc == lane) {
            ++h0;
        } else if (bc == lane + 32) {
            ++h1;
        }
        if (lane == 0) {
            cand_ids[step * DS_CAND_K + r] = bi;
            cand_logits[step * DS_CAND_K + r] = bv;
        }
    }
}

// Stage 3: Compute all pairwise Markov transition dots `D[step, j, c] = W2[cand[step, c]] . W1[prev]`
// in parallel across all `steps` and all `K = 32` candidates!
// Grid: (steps, DS_CAND_K, 1), threadgroup: (256, 1, 1).
kernel void ds_markov_walk_dots_int4(
    device const uint *cand_ids [[buffer(0)]],
    device const uchar *w1_codes [[buffer(1)]],
    device const half2 *w1_groups [[buffer(2)]],
    device const uchar *w2_codes [[buffer(3)]],
    device const half2 *w2_groups [[buffer(4)]],
    device const uint *draft_tokens [[buffer(5)]],
    device float *walk_dots [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    const uint step = group.x, j = group.y;
    if (step == 0 && j > 0) return;
    threadgroup float m_shared[DS_RANK];
    threadgroup float g_sums[4];
    threadgroup float s_red[8];
    const uint prev = (step == 0) ? draft_tokens[0] : cand_ids[(step - 1) * DS_CAND_K + j];
    float m = 0.0f;
    if (prev < DS_FULL_VOCAB) {
        const uchar byte = w1_codes[ulong(prev) * (DS_RANK / 2) + (tid / 2)];
        const float code = float((tid & 1u) ? (byte >> 4) : (byte & 0x0Fu));
        const float2 sb = float2(w1_groups[ulong(prev) * (DS_RANK / 64) + (tid / 64)]);
        m = fma(code, sb.x, sb.y);
    }
    m_shared[tid] = m;
    const float s32 = simd_sum(m);
    if (lane == 0) s_red[simd] = s32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) {
        g_sums[tid] = s_red[2 * tid] + s_red[2 * tid + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < DS_CAND_K) {
        const uint cid = cand_ids[step * DS_CAND_K + tid];
        float bias = 0.0f;
        if (cid < DS_FULL_VOCAB) {
            bias = ds_eval_w2_int4(
                w2_codes + ulong(cid) * (DS_RANK / 2),
                w2_groups + ulong(cid) * 4,
                m_shared,
                g_sums
            );
        }
        walk_dots[(step * DS_CAND_K + j) * DS_CAND_K + tid] = bias;
    }
}

// Stage 4: Walk the `steps` slots on a single 256-thread threadgroup (in registers/shared memory)
// and compute `conf_out[0..steps]`!
// Grid: (1, 1, 1), threadgroup: (256, 1, 1).
kernel void ds_markov_walk_pick_int4(
    device const uint *cand_ids [[buffer(0)]],
    device const float *cand_logits [[buffer(1)]],
    device const float *walk_dots [[buffer(2)]],
    device const uchar *w1_codes [[buffer(3)]],
    device const half2 *w1_groups [[buffer(4)]],
    device const float *hidden [[buffer(5)]],
    device const float *conf_w [[buffer(6)]],
    device const float *conf_b [[buffer(7)]],
    device uint *draft_tokens [[buffer(8)]],
    device float *conf_out [[buffer(9)]],
    constant uint &steps [[buffer(10)]],
    device const uint2 *rep_pairs [[buffer(11)]],
    constant uint &rep_len [[buffer(12)]],
    constant float2 &penalties [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    threadgroup uchar cand_counts[8][DS_CAND_K];
    threadgroup uint shared_prev_j = 0;
    threadgroup uint shared_prev_tok = 0;

    const uint total_cands = steps * DS_CAND_K;
    for (uint idx = tid; idx < total_cands; idx += 256) {
        const uint cid = cand_ids[idx];
        uint cnt = 0;
        if (penalties.x > 1.0001f && rep_len > 0 && cid != UINT_MAX) {
            for (uint r = 0; r < rep_len; ++r) {
                const uint2 pair = rep_pairs[r];
                if ((pair.x >> 24) == 0xFFu && (pair.x & 0x00FFFFFFu) == cid) {
                    cnt = min(pair.y, 255u);
                    break;
                }
            }
        }
        cand_counts[idx / DS_CAND_K][idx % DS_CAND_K] = uchar(cnt);
    }
    if (tid == 0) {
        shared_prev_j = 0;
        shared_prev_tok = draft_tokens[0];
        draft_tokens[8] = UINT_MAX;
        conf_out[7] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < steps; ++s) {
        const uint prev_tok = shared_prev_tok;
        const uint prev_j = shared_prev_j;
        float m = 0.0f;
        if (prev_tok < DS_FULL_VOCAB) {
            const uchar byte = w1_codes[ulong(prev_tok) * (DS_RANK / 2) + (tid / 2)];
            const float code = float((tid & 1u) ? (byte >> 4) : (byte & 0x0Fu));
            const float2 sb = float2(w1_groups[ulong(prev_tok) * (DS_RANK / 64) + (tid / 64)]);
            m = fma(code, sb.x, sb.y);
        }
        device const float *h = hidden + ulong(s) * DS_WIDTH;
        float dot_sum = m * conf_w[DS_WIDTH + tid];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint c = i * 256 + tid;
            dot_sum = fma(h[c], conf_w[c], dot_sum);
        }
        const float total = nano_group_sum<256>(dot_sum, tid, partial);
        if (tid == 0) {
            conf_out[s] = nano_sigmoid(total + conf_b[0]);
        }
        if (simd == 0) {
            const bool active = lane < DS_CAND_K;
            const uint cid = active ? cand_ids[s * DS_CAND_K + lane] : UINT_MAX;
            float sc = active
                ? (cand_logits[s * DS_CAND_K + lane] + walk_dots[(s * DS_CAND_K + prev_j) * DS_CAND_K + lane])
                : -1e30f;
            if (active && cid != UINT_MAX && penalties.x > 1.0001f) {
                uint cnt = uint(cand_counts[s][lane]);
                for (uint p = 0; p <= s; ++p) {
                    if (draft_tokens[p] == cid) {
                        ++cnt;
                    }
                }
                if (cnt > 0) {
                    const float rep_p = penalties.x;
                    const float freq_p = penalties.y * float(cnt - 1u);
                    sc = (sc > 0.0f ? (sc / rep_p) : (sc * rep_p)) - freq_p;
                }
            }
            float bv = sc;
            uint bi = cid;
            uint bj = lane;
            for (ushort d = 8; d > 0; d >>= 1) {
                const float ov = simd_shuffle_xor(bv, d);
                const uint oi = simd_shuffle_xor(bi, d);
                const uint oj = simd_shuffle_xor(bj, d);
                if (ds_better(ov, oi, bv, bi)) {
                    bv = ov;
                    bi = oi;
                    bj = oj;
                }
            }
            if (s == 0) {
                float v2 = (active && lane != bj) ? sc : -1e30f;
                uint i2 = (active && lane != bj) ? cid : UINT_MAX;
                for (ushort d = 8; d > 0; d >>= 1) {
                    const float ov = simd_shuffle_xor(v2, d);
                    const uint oi = simd_shuffle_xor(i2, d);
                    if (ds_better(ov, oi, v2, i2)) {
                        v2 = ov;
                        i2 = oi;
                    }
                }
                if (lane == 0) {
                    draft_tokens[8] = i2;
                    conf_out[7] = (i2 != UINT_MAX) ? max(bv - v2, 0.0f) : 99.0f;
                }
            }
            if (lane == 0) {
                draft_tokens[s + 1] = bi;
                shared_prev_tok = bi;
                shared_prev_j = bj;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
