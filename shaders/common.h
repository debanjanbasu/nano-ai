// Shared math, reductions, and PTQ1_0 / Hadamard primitives for MiniCPM-V 4.7 1B.
#ifndef MICRO_COMMON_H
#define MICRO_COMMON_H

#include <metal_stdlib>
using namespace metal;

static inline float micro_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float micro_silu(float x) { return x * micro_sigmoid(x); }

template<uint threads>
static inline float micro_group_sum(float value, uint tid, threadgroup float *partial) {
    const float sum = simd_sum(value);
    if (tid % 32 == 0) partial[tid / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < threads / 32; ++i) result += partial[i];
    return result;
}

struct Ptq1Block {
    uchar qs[24];
    uchar qh[2];
    half d;
};
static_assert(sizeof(Ptq1Block) == 28, "PTQ1_0 block layout must be 28 bytes");

// Base-3 prefixes of a code byte from one half FMA and one subtraction each,
// avoiding quarter-rate floor and integer-to-float conversion on Apple GPUs.
static inline half ptq1_code(uint byte) {
    return as_type<half>(ushort(0x6400u | byte));
}

static inline half ptq1_prefix(half code, uint p) {
    const half powers[5] = {
        3.0h / 256.0h, 9.0h / 256.0h, 27.0h / 256.0h, 81.0h / 256.0h, 243.0h / 256.0h};
    const half shifts[5] = {1013.5h, 1001.5h, 929.5h, 821.5h, 173.5h};
    const half offsets[5] = {1026.0h, 1038.0h, 1038.0h, 1146.0h, 1146.0h};
    return fma(code, powers[p - 1], shifts[p - 1]) - offsets[p - 1];
}

static inline half ptq1_balanced_prefix(half code, uint p) {
    const half powers[5] = {
        3.0h / 256.0h, 9.0h / 256.0h, 27.0h / 256.0h, 81.0h / 256.0h, 243.0h / 256.0h};
    const half shifts[5] = {1013.5h, 1001.5h, 929.5h, 821.5h, 173.5h};
    const half offsets[5] = {1027.0h, 1042.0h, 1051.0h, 1186.0h, 1267.0h};
    return fma(code, powers[p - 1], shifts[p - 1]) - offsets[p - 1];
}

static inline void ptq1_trits(half code, thread half *trits) {
    half previous = 0.0h;
    #pragma clang loop unroll(full)
    for (uint n = 0; n < 5; ++n) {
        const half balanced = ptq1_balanced_prefix(code, n + 1);
        trits[n] = n == 0 ? balanced : fma(-3.0h, previous, balanced);
        previous = balanced;
    }
}

// Rotate one 512-wide block whose thread tid (0..64) holds elements i * 64 + tid in values.
// Supports both 512-wide and 1024-wide (2 blocks of 512) or 3584-wide (7 blocks of 512).
template<bool inverse, typename Signs, typename Output>
static inline void fwht_512_values(
    thread float *values,
    Signs signs,
    Output output,
    uint blocks_per_row,
    uint block,
    uint tid,
    threadgroup float *shared
) {
    const ulong base = ulong(block) * 512;
    const uint sign_base = (block % blocks_per_row) * 512;
    const float norm = 0.04419417382415922f; // 1 / sqrt(512)
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint local = i * 64 + tid;
        const float sign = inverse ? 1.0f : signs[sign_base + local];
        values[i] = values[i] * sign * norm;
    }
    for (uint distance = 1; distance < 32; distance <<= 1) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            const float a = values[i];
            const float b = simd_shuffle_xor(a, distance);
            values[i] = (tid & distance) == 0 ? a + b : b - a;
        }
    }
    // distance = 32 across the two 32-thread SIMD groups of the 64-thread threadgroup
    {
        const uint distance = 32;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) shared[i * 64 + tid] = values[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            const float a = values[i];
            const float b = shared[i * 64 + (tid ^ distance)];
            values[i] = (tid & distance) == 0 ? a + b : b - a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint step = 1; step < 8; step <<= 1) {
        for (uint i = 0; i < 8; i += 2 * step) {
            for (uint j = 0; j < step; ++j) {
                const float a = values[i + j];
                const float b = values[i + j + step];
                values[i + j] = a + b;
                values[i + j + step] = a - b;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint local = i * 64 + tid;
        const float sign = inverse ? signs[sign_base + local] : 1.0f;
        output[base + local] = values[i] * sign;
    }
}

// Orthonormal 256-point Walsh-Hadamard transform, one value per thread of a
// 256-thread group. Used for rotated Q8 KV cache (head_dim = 256).
static inline float hadamard256(float value, uint tid, threadgroup float *scratch) {
    for (ushort distance = 1; distance < 32; distance <<= 1) {
        const float other = simd_shuffle_xor(value, distance);
        value = (tid & distance) ? other - value : value + other;
    }
    for (uint distance = 32; distance < 256; distance <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float other = scratch[tid ^ distance];
        value = (tid & distance) ? other - value : value + other;
    }
    return value * (1.0f / 16.0f);
}

#endif
