/*
 * MIT License
 *
 * Copyright (c) 2026 kitamuratatuki
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma GCC optimize("O3,unroll-loops,omit-frame-pointer,inline")
#pragma GCC target("avx2,bmi,bmi2,popcnt,lzcnt")

#include <vector>
#include <limits>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <atomic>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include "io_struct.hpp"

using io_struct::input_t;
using io_struct::output_t;

struct GlobalState {
    alignas(64) std::atomic<long long> best_T{-1};
    alignas(64) std::atomic<uint64_t> best_mask{0};
    uint64_t adj[64];
    long long weights[64];
    alignas(64) int v_params[64][128];
    alignas(64) int r_params[128];
    int N;
};

struct Xoroshiro128Plus {
    uint64_t s[2];
    Xoroshiro128Plus(uint64_t seed1, uint64_t seed2) { s[0] = seed1; s[1] = seed2; }
    static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t s0 = s[0]; uint64_t s1 = s[1]; const uint64_t result = s0 + s1;
        s1 ^= s0; s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); s[1] = rotl(s1, 37); return result;
    }
};

inline void fast_add_sums(int* next_sum, const int* current_sum, const int* v_ptr) {
#if defined(__AVX2__)
    const __m256i* s_simd = (const __m256i*)current_sum;
    const __m256i* v_simd = (const __m256i*)v_ptr;
    __m256i* n_simd = (__m256i*)next_sum;
    for(int k=0; k<16; ++k) n_simd[k] = _mm256_add_epi32(s_simd[k], v_simd[k]);
#elif defined(__ARM_NEON)
    const int32x4_t* s_simd = (const int32x4_t*)current_sum;
    const int32x4_t* v_simd = (const int32x4_t*)v_ptr;
    int32x4_t* n_simd = (int32x4_t*)next_sum;
    for(int k=0; k<32; ++k) n_simd[k] = vaddq_s32(s_simd[k], v_simd[k]);
#else
    for(int k=0; k<128; ++k) next_sum[k] = current_sum[k] + v_ptr[k];
#endif
}

inline bool check_success_simd(const int* current_sum, const int* r_params) {
#if defined(__AVX2__)
    const __m256i* s_simd = (const __m256i*)current_sum;
    const __m256i* r_simd = (const __m256i*)r_params;
    for(int k=0; k<16; ++k) {
        __m256i cmp = _mm256_cmpgt_epi32(r_simd[k], s_simd[k]);
        if (!_mm256_testz_si256(cmp, cmp)) return false;
    }
#elif defined(__ARM_NEON)
    const int32x4_t* s_simd = (const int32x4_t*)current_sum;
    const int32x4_t* r_simd = (const int32x4_t*)r_params;
    for(int k=0; k<32; ++k) {
        uint32x4_t cmp = vcltq_s32(s_simd[k], r_simd[k]);
        if (vaddv_u32(vget_low_u32(cmp)) + vaddv_u32(vget_high_u32(cmp)) > 0) return false;
    }
#else
    for(int k=0; k<128; ++k) if (current_sum[k] < r_params[k]) return false;
#endif
    return true;
}

void fast_dfs_v6(
    uint64_t current_mask,
    uint64_t candidates,
    long long current_T,
    int current_sum[128],
    GlobalState& gs
) {
    if (check_success_simd(current_sum, gs.r_params)) {
        long long old_best = gs.best_T.load(std::memory_order_relaxed);
        while (current_T > old_best && !gs.best_T.compare_exchange_weak(old_best, current_T)) {}
        if (current_T >= old_best) gs.best_mask.store(current_mask, std::memory_order_relaxed);
    }

    if (candidates == 0) return;

    long long rem_max = 0;
    uint64_t tmp = candidates;
    while(tmp) {
        rem_max += gs.weights[__builtin_ctzll(tmp)];
        tmp &= tmp - 1;
    }
    if (current_T + rem_max <= gs.best_T.load(std::memory_order_relaxed)) return;

    while (candidates) {
        int v = __builtin_ctzll(candidates);
        uint64_t v_bit = 1ULL << v;
        alignas(64) int next_sum[128];
        fast_add_sums(next_sum, current_sum, gs.v_params[v]);
        fast_dfs_v6(current_mask | v_bit, candidates & gs.adj[v] & ~(v_bit | (v_bit - 1)), current_T + gs.weights[v], next_sum, gs);
        candidates &= ~v_bit;
        rem_max -= gs.weights[v];
        if (current_T + rem_max <= gs.best_T.load(std::memory_order_relaxed)) break;
    }
}

void solve(input_t &input, output_t &output) {
    io_struct::InitOutput(output);
    const int N = input.N;
    GlobalState gs;
    gs.N = N;
    gs.best_T.store(-1);
    gs.best_mask.store(0);
    
    struct Vertex { int id; long long w; };
    std::vector<Vertex> vs(N);
    for(int i=0; i<N; ++i) {
        vs[i].id = i; vs[i].w = 0;
        for(int k=0; k<128; ++k) vs[i].w += input.v[i][k];
    }
    std::sort(vs.begin(), vs.end(), [](const Vertex& a, const Vertex& b) { return a.w > b.w; });

    for(int i=0; i<N; ++i) {
        int orig = vs[i].id;
        gs.weights[i] = vs[i].w;
        memcpy(gs.v_params[i], input.v[orig].data(), 128 * sizeof(int));
    }
    memcpy(gs.r_params, input.r.data(), 128 * sizeof(int));
    memset(gs.adj, 0, sizeof(gs.adj));
    for(int i=0; i<N; ++i) {
        int orig_i = vs[i].id;
        for(int j=0; j<N; ++j) if(input.A[orig_i][vs[j].id]) gs.adj[i] |= (1ULL << j);
        gs.adj[i] |= (1ULL << i);
    }

    // Hybrid Part 1: Quick Greedy / SA (Very short time)
    #ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
    #else
    {
        int tid = 0;
    #endif
        Xoroshiro128Plus rng(12345 + 100 * (tid + 1), 67890);
        for(int iter=0; iter<50; ++iter) {
            uint64_t mask = 0;
            long long cur_T = 0;
            int cur_sum[128] = {0};
            uint64_t candidates = (N == 64) ? ~0ULL : (1ULL << N) - 1;
            
            // Randomly pick a few vertices, then greedy
            while(candidates) {
                uint64_t c = candidates;
                int pc = __builtin_popcountll(c);
                int r = (iter == 0) ? 0 : (int)(rng.next() % std::min(pc, 2)); 
                for(int i=0; i<r; ++i) c &= c - 1;
                int v = __builtin_ctzll(c);
                
                mask |= (1ULL << v);
                cur_T += gs.weights[v];
                fast_add_sums(cur_sum, cur_sum, gs.v_params[v]);
                candidates &= gs.adj[v];
                candidates &= ~((1ULL << (v + 1)) - 1);
            }
            if (check_success_simd(cur_sum, gs.r_params)) {
                long long old = gs.best_T.load(std::memory_order_relaxed);
                while (cur_T > old && !gs.best_T.compare_exchange_weak(old, cur_T)) {}
                if (cur_T >= old) gs.best_mask.store(mask, std::memory_order_relaxed);
            }
        }
    }

    // Hybrid Part 2: DFS
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for(int i=0; i<N; ++i) {
        int start_sum[128];
        memcpy(start_sum, gs.v_params[i], 128 * sizeof(int));
        uint64_t full = (N == 64) ? ~0ULL : (1ULL << N) - 1;
        uint64_t cands = gs.adj[i] & full & ~((1ULL << (i + 1)) - 1);
        fast_dfs_v6(1ULL << i, cands, gs.weights[i], start_sum, gs);
    }

    long long final_best_T = gs.best_T.load();
    uint64_t final_best_mask = gs.best_mask.load();
    if (final_best_T != -1) {
        output.T = final_best_T;
        output.K_size = __builtin_popcountll(final_best_mask);
        std::vector<int> res;
        for(int i=0; i<N; ++i) if(final_best_mask & (1ULL << i)) res.push_back(vs[i].id + 1);
        std::sort(res.begin(), res.end());
        for(int i=0; i<output.K_size; ++i) output.members[i] = res[i];
    }
}