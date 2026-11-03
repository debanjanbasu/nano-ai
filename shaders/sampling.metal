// GPU-resident top-k and greedy argmax selection over vocabulary logits.
#include "common.h"

struct TopKCandidate { uint token_id; float logit; };

struct GreedyRowResult {
    uint best_id;
    float best_logit;
    uint nonfinite;
    float second_logit;
};

static inline int total_key(uint bits) {
    int key = as_type<int>(bits);
    return key ^ int(uint(key >> 31) >> 1);
}

static inline bool key_better(int a_key, uint a_id, int b_key, uint b_id) {
    if (a_id == UINT_MAX) return false;
    if (b_id == UINT_MAX) return true;
    return a_key > b_key || (a_key == b_key && a_id < b_id);
}

static inline void best_group(thread int &key, thread uint &id, threadgroup int *keys,
                              threadgroup uint *ids, uint lane, uint simd) {
    const int simd_key = simd_max(id == UINT_MAX ? INT_MIN : key);
    const uint simd_id = simd_min(id != UINT_MAX && key == simd_key ? id : UINT_MAX);
    if (lane == 0) { keys[simd] = simd_key; ids[simd] = simd_id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    key = keys[0];
    id = ids[0];
    for (uint i = 1; i < 8; ++i) {
        if (key_better(keys[i], ids[i], key, id)) { key = keys[i]; id = ids[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

constant constexpr uint TOPK_GROUP = 4096;
constant constexpr uint TOPK_PER_THREAD = TOPK_GROUP / 256;
constant constexpr uint MAX_TOP_K = 100;
constant constexpr uint TOPK_FINAL_PER_THREAD = 25;

kernel void topk_rows_partial(
    device const float *logits [[buffer(0)]],
    device TopKCandidate *partial [[buffer(1)]],
    constant uint &vocab [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup int shared_keys[8];
    threadgroup uint shared_ids[8];
    device const float *row = logits + ulong(group.y) * stride;
    const uint base = group.x * TOPK_GROUP + tid;
    int keys[TOPK_PER_THREAD];
    uint valid = 0;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TOPK_PER_THREAD; ++i) {
        const uint id = base + i * 256;
        const bool inside = id < vocab;
        const float value = inside ? row[id] : 0.0f;
        keys[i] = total_key(as_type<uint>(value));
        valid |= uint(inside) << i;
    }
    const uint slot = group.y * groups.x + group.x;
    for (uint round = 0; round < k; ++round) {
        int key = INT_MIN;
        uint id = UINT_MAX;
        uint index = 0;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < TOPK_PER_THREAD; ++i) {
            if ((valid & (1u << i)) && (id == UINT_MAX || keys[i] > key)) {
                key = keys[i];
                id = base + i * 256;
                index = i;
            }
        }
        const uint own = id;
        best_group(key, id, shared_keys, shared_ids, lane, simd);
        if (own != UINT_MAX && own == id) valid &= ~(1u << index);
        if (tid == 0) {
            partial[slot * MAX_TOP_K + round] =
                TopKCandidate{id, as_type<float>(uint(total_key(uint(key))))};
        }
    }
}

kernel void topk_rows_final(
    device const TopKCandidate *partial [[buffer(0)]],
    device TopKCandidate *output [[buffer(1)]],
    constant uint &groups [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup int shared_keys[8];
    threadgroup uint shared_ids[8];
    int keys[TOPK_FINAL_PER_THREAD];
    uint ids[TOPK_FINAL_PER_THREAD];
    const uint candidates = groups * k;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < TOPK_FINAL_PER_THREAD; ++j) {
        const uint c = tid + j * 256;
        ids[j] = UINT_MAX;
        keys[j] = INT_MIN;
        if (c < candidates) {
            const TopKCandidate item = partial[(row * groups + c / k) * MAX_TOP_K + c % k];
            ids[j] = item.token_id;
            keys[j] = total_key(as_type<uint>(item.logit));
        }
    }
    for (uint round = 0; round < k; ++round) {
        int key = INT_MIN;
        uint id = UINT_MAX;
        uint index = 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TOPK_FINAL_PER_THREAD; ++j) {
            if (key_better(keys[j], ids[j], key, id)) {
                key = keys[j];
                id = ids[j];
                index = j;
            }
        }
        const uint own = id;
        best_group(key, id, shared_keys, shared_ids, lane, simd);
        if (own != UINT_MAX && own == id) ids[index] = UINT_MAX;
        if (tid == 0) {
            output[row * MAX_TOP_K + round] =
                TopKCandidate{id, as_type<float>(uint(total_key(uint(key))))};
        }
    }
}

// Fast greedy argmax over `rows` rows of `vocab` logits (1024-logit groups)
static inline void best2_group(
    thread int &k1, thread uint &id1, thread int &k2,
    threadgroup int *s_k1, threadgroup uint *s_id1, threadgroup int *s_k2,
    uint lane, uint simd
) {
    for (ushort d = 16; d > 0; d >>= 1) {
        const int ok1 = simd_shuffle_xor(k1, d);
        const uint oid1 = simd_shuffle_xor(id1, d);
        const int ok2 = simd_shuffle_xor(k2, d);
        if (key_better(ok1, oid1, k1, id1)) {
            k2 = max(k1, ok2);
            k1 = ok1;
            id1 = oid1;
        } else if (oid1 != id1) {
            k2 = max(k2, ok1);
        } else {
            k2 = max(k2, ok2);
        }
    }
    if (lane == 0) {
        s_k1[simd] = k1;
        s_id1[simd] = id1;
        s_k2[simd] = k2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    k1 = s_k1[0];
    id1 = s_id1[0];
    k2 = s_k2[0];
    for (uint i = 1; i < 8; ++i) {
        const int ok1 = s_k1[i];
        const uint oid1 = s_id1[i];
        const int ok2 = s_k2[i];
        if (key_better(ok1, oid1, k1, id1)) {
            k2 = max(k1, ok2);
            k1 = ok1;
            id1 = oid1;
        } else if (oid1 != id1) {
            k2 = max(k2, ok1);
        } else {
            k2 = max(k2, ok2);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void greedy_rows_partial(
    device const float *logits [[buffer(0)]],
    device GreedyRowResult *partials [[buffer(1)]],
    constant uint &vocab [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup int shared_k1[8];
    threadgroup uint shared_ids[8];
    threadgroup int shared_k2[8];
    threadgroup uint shared_nf[8];
    device const float *row = logits + ulong(group.y) * vocab;
    const uint base = group.x * 1024;
    int k1 = INT_MIN;
    int k2 = INT_MIN;
    uint id = UINT_MAX;
    uint nf = 0;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        const uint idx = base + i * 256 + tid;
        if (idx < vocab) {
            const float val = row[idx];
            nf |= uint(!isfinite(val));
            const int k = total_key(as_type<uint>(val));
            if (key_better(k, idx, k1, id)) {
                k2 = k1;
                k1 = k;
                id = idx;
            } else if (k > k2) {
                k2 = k;
            }
        }
    }
    nf = simd_or(nf);
    if (lane == 0) shared_nf[simd] = nf;
    best2_group(k1, id, k2, shared_k1, shared_ids, shared_k2, lane, simd);
    if (tid == 0) {
        uint total_nf = 0;
        for (uint s = 0; s < 8; ++s) total_nf |= shared_nf[s];
        partials[group.y * groups.x + group.x] = GreedyRowResult{
            id,
            as_type<float>(uint(total_key(uint(k1)))),
            total_nf,
            as_type<float>(uint(total_key(uint(k2))))
        };
    }
}

kernel void greedy_rows_final(
    device const GreedyRowResult *partials [[buffer(0)]],
    device GreedyRowResult *results [[buffer(1)]],
    constant uint &groups [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup int shared_k1[8];
    threadgroup uint shared_ids[8];
    threadgroup int shared_k2[8];
    threadgroup uint shared_nf[8];
    int k1 = INT_MIN;
    int k2 = INT_MIN;
    uint id = UINT_MAX;
    uint nf = 0;
    for (uint g = tid; g < groups; g += 256) {
        const GreedyRowResult item = partials[row * groups + g];
        nf |= item.nonfinite;
        const int ik1 = total_key(as_type<uint>(item.best_logit));
        const int ik2 = total_key(as_type<uint>(item.second_logit));
        if (key_better(ik1, item.best_id, k1, id)) {
            k2 = max(k1, ik2);
            k1 = ik1;
            id = item.best_id;
        } else {
            k2 = max(k2, ik1);
        }
    }
    nf = simd_or(nf);
    if (lane == 0) shared_nf[simd] = nf;
    best2_group(k1, id, k2, shared_k1, shared_ids, shared_k2, lane, simd);
    if (tid == 0) {
        uint total_nf = 0;
        for (uint s = 0; s < 8; ++s) total_nf |= shared_nf[s];
        results[row] = GreedyRowResult{
            id,
            as_type<float>(uint(total_key(uint(k1)))),
            total_nf,
            as_type<float>(uint(total_key(uint(k2))))
        };
    }
}

// Apply transformer repetition penalty (`repetition_penalty > 1.0`) AND per-row DRY / frequency
// penalty (`frequency_penalty * count`) for `(row << 24 | token_id, count)` pairs in `history_pairs`.
// When `row_mask == 0xFF`, the entry applies to all `0..rows`; otherwise it applies to `row == row_mask`!
// Pairs arrive sorted by token with the all-rows entry first, and the first thread of each token run
// applies the whole run in order, so no logit sees concurrent read-modify-writes.
kernel void apply_repetition_penalty(
    device float *logits [[buffer(0)]],
    device const uint2 *history_pairs [[buffer(1)]],
    constant uint &history_len [[buffer(2)]],
    constant uint &vocab [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant float2 &penalties [[buffer(5)]], // (repetition_penalty, frequency_penalty)
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= history_len) return;
    const uint tok = history_pairs[gid].x & 0x00FFFFFFu;
    if (tok >= vocab || (gid > 0 && (history_pairs[gid - 1].x & 0x00FFFFFFu) == tok)) return;
    const float rep_p = penalties.x;
    const float inv_p = 1.0f / rep_p;
    for (uint i = gid; i < history_len && (history_pairs[i].x & 0x00FFFFFFu) == tok; ++i) {
        const uint row_mask = history_pairs[i].x >> 24;
        const uint cnt = history_pairs[i].y;
        if (cnt == 0) continue;
        if (row_mask == 0xFFu) {
            const float freq_p = penalties.y * float(cnt - 1u);
            for (uint r = 0; r < rows; ++r) {
                const ulong idx = ulong(r) * vocab + tok;
                const float v = logits[idx];
                const float scaled = v > 0.0f ? (v * inv_p) : (v * rep_p);
                logits[idx] = scaled - freq_p;
            }
        } else if (row_mask < rows) {
            logits[ulong(row_mask) * vocab + tok] -= penalties.y * float(cnt);
        } else if ((row_mask & 0x80u) != 0u && (row_mask & 0x7Fu) < rows) {
            logits[ulong(row_mask & 0x7Fu) * vocab + tok] += penalties.y * float(cnt);
        }
    }
}
