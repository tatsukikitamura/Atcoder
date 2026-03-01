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
// SIMD codepath (AVX-512 vs AVX2 vs scalar) is selected by -march flag from CMake.
// Use -march=native locally (AVX2), -march=icelake-client for contest (AVX-512).

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include "io_struct.hpp"

using io_struct::input_t;
using io_struct::output_t;

// ============================================================================
// Global State
// ============================================================================
static constexpr int PDIM = 128;  // parameter dimensions

struct GlobalState {
    // Best solution: best_T is atomic for fast lock-free reads in pruning,
    // spinlock protects the combined T+mask write to keep them consistent.
    alignas(64) std::atomic_flag best_lock = ATOMIC_FLAG_INIT;
    alignas(64) std::atomic<int> best_T{-1};
    uint64_t best_mask = 0;

    uint64_t adj[64];          // adjacency bitmasks (reordered)
    int weights[64];           // total weight per vertex
    alignas(64) int vp[64][PDIM]; // parameters per vertex
    alignas(64) int rp[PDIM];    // BOSS requirements
    int reorder_map[64];       // reordered_id -> original_id
    int best_small = -1;       // single-threaded best score for N<=24 FPT path
};

struct FptTask {
    uint64_t mask;
    int cands_weight;
    alignas(64) int cands_sum[PDIM];
};
// ============================================================================
// SIMD 128-element operations
// ============================================================================
inline void v_add(int* __restrict__ d, const int* __restrict__ a, const int* __restrict__ b) {
#if defined(__AVX512F__)
    for (int i = 0; i < 8; ++i)
        _mm512_store_si512((__m512i*)(d + i*16),
            _mm512_add_epi32(_mm512_load_si512((const __m512i*)(a + i*16)),
                             _mm512_load_si512((const __m512i*)(b + i*16))));
#elif defined(__AVX2__)
    for (int i = 0; i < 16; ++i)
        _mm256_store_si256((__m256i*)(d + i*8),
            _mm256_add_epi32(_mm256_load_si256((const __m256i*)(a + i*8)),
                             _mm256_load_si256((const __m256i*)(b + i*8))));
#else
    for (int i = 0; i < PDIM; ++i) d[i] = a[i] + b[i];
#endif
}

inline void v_iadd(int* __restrict__ d, const int* __restrict__ s) {
#if defined(__AVX512F__)
    for (int i = 0; i < 8; ++i) {
        __m512i vd = _mm512_load_si512((__m512i*)(d + i*16));
        __m512i vs = _mm512_load_si512((const __m512i*)(s + i*16));
        _mm512_store_si512((__m512i*)(d + i*16), _mm512_add_epi32(vd, vs));
    }
#elif defined(__AVX2__)
    for (int i = 0; i < 16; ++i) {
        __m256i vd = _mm256_load_si256((__m256i*)(d + i*8));
        __m256i vs = _mm256_load_si256((const __m256i*)(s + i*8));
        _mm256_store_si256((__m256i*)(d + i*8), _mm256_add_epi32(vd, vs));
    }
#else
    for (int i = 0; i < PDIM; ++i) d[i] += s[i];
#endif
}

inline void v_isub(int* __restrict__ d, const int* __restrict__ s) {
#if defined(__AVX512F__)
    for (int i = 0; i < 8; ++i) {
        __m512i vd = _mm512_load_si512((__m512i*)(d + i*16));
        __m512i vs = _mm512_load_si512((const __m512i*)(s + i*16));
        _mm512_store_si512((__m512i*)(d + i*16), _mm512_sub_epi32(vd, vs));
    }
#elif defined(__AVX2__)
    for (int i = 0; i < 16; ++i) {
        __m256i vd = _mm256_load_si256((__m256i*)(d + i*8));
        __m256i vs = _mm256_load_si256((const __m256i*)(s + i*8));
        _mm256_store_si256((__m256i*)(d + i*8), _mm256_sub_epi32(vd, vs));
    }
#else
    for (int i = 0; i < PDIM; ++i) d[i] -= s[i];
#endif
}

// d = a - b
inline void v_sub(int* __restrict__ d, const int* __restrict__ a, const int* __restrict__ b) {
#if defined(__AVX512F__)
    for (int i = 0; i < 8; ++i)
        _mm512_store_si512((__m512i*)(d + i*16),
            _mm512_sub_epi32(_mm512_load_si512((const __m512i*)(a + i*16)),
                             _mm512_load_si512((const __m512i*)(b + i*16))));
#elif defined(__AVX2__)
    for (int i = 0; i < 16; ++i)
        _mm256_store_si256((__m256i*)(d + i*8),
            _mm256_sub_epi32(_mm256_load_si256((const __m256i*)(a + i*8)),
                             _mm256_load_si256((const __m256i*)(b + i*8))));
#else
    for (int i = 0; i < PDIM; ++i) d[i] = a[i] - b[i];
#endif
}

inline bool v_geq(const int* __restrict__ a, const int* __restrict__ b) {
#if defined(__AVX512F__)
    for (int i = 0; i < 8; ++i) {
        __mmask16 mask = _mm512_cmpgt_epi32_mask(
            _mm512_load_si512((const __m512i*)(b + i*16)),
            _mm512_load_si512((const __m512i*)(a + i*16)));
        if (mask) return false;
    }
    return true;
#elif defined(__AVX2__)
    for (int i = 0; i < 16; ++i) {
        __m256i cmp = _mm256_cmpgt_epi32(
            _mm256_load_si256((const __m256i*)(b + i*8)),
            _mm256_load_si256((const __m256i*)(a + i*8)));
        if (!_mm256_testz_si256(cmp, cmp)) return false;
    }
    return true;
#else
    for (int i = 0; i < PDIM; ++i)
        if (a[i] < b[i]) return false;
    return true;
#endif
}

// ============================================================================
// Update best solution (thread-safe via spinlock for T+mask consistency)
// ============================================================================
inline void update_best(GlobalState& gs, int T, uint64_t mask) {
    // Quick check without lock (atomic read)
    if (T <= gs.best_T.load(std::memory_order_relaxed)) return;

    // Acquire spinlock for combined T+mask write
    while (gs.best_lock.test_and_set(std::memory_order_acquire)) {}

    if (T > gs.best_T.load(std::memory_order_relaxed)) {
        gs.best_mask = mask;
        gs.best_T.store(T, std::memory_order_release);
    }

    gs.best_lock.clear(std::memory_order_release);
}

// Single-threaded (non-atomic) update for N==24 path.
// Avoids the LOCK XCHG spinlock overhead (≈20-30 cycles even uncontested)
// since fpt_dfs always runs on a single thread.
inline void update_best_st(GlobalState& gs, int T, uint64_t mask) {
    if (T > gs.best_T.load(std::memory_order_relaxed)) {
        gs.best_mask = mask;
        gs.best_T.store(T, std::memory_order_relaxed);
    }
}

// ============================================================================
// Incremental local coloring for BnB branch loop
// ============================================================================
// Stores the complete local coloring so we can update it O(|class|) when
// one vertex is removed, instead of recoloring from scratch in O(|cands|²).
struct LocalColorData {
    uint8_t   color[64];     // color[v] = color class assigned to v  (0..63 fits uint8_t)
    int       cmax[64];      // cmax[c]  = max weight in color class c
    uint64_t  members[64];   // members[c] = bitmask of still-present vertices in class c
    int       num_colors;
    int       ub;            // current upper bound = sum(cmax[c])
};

// Full local greedy coloring, filling an LCData struct.
inline LocalColorData lc_compute(uint64_t cands, const GlobalState& gs) {
    LocalColorData lc;
    memset(&lc, 0, sizeof(lc));
    if (cands == 0) return lc;

    // Sort vertices by degree within cands descending (DSATUR-style).
    // Highly-connected vertices get colored first → fewer color classes → tighter upper bound.
    // Skip sort when |cands| is tiny: overhead exceeds benefit.
    int order[64];
    int n = 0;
    { uint64_t tmp = cands; while (tmp) { order[n++] = __builtin_ctzll(tmp); tmp &= tmp - 1; } }
    // n==1: single vertex, color=0, ub=weights[v] — skip all overhead
    if (n == 1) {
        int v = order[0];
        lc.color[v] = 0; lc.members[0] = 1ULL << v;
        lc.cmax[0] = gs.weights[v]; lc.num_colors = 1; lc.ub = gs.weights[v];
        return lc;
    }
    // Bucket sort descending by degree in cands: O(N) vs std::sort's O(N log N).
    // deg values are in [0, n-1], so 65-element bucket array is sufficient.
    {
        uint64_t buckets[65] = {};
        for (int i = 0; i < n; ++i) {
            int v = order[i];
            int d = __builtin_popcountll(cands & gs.adj[v]);
            buckets[d] |= 1ULL << v;
        }
        int si = 0;
        for (int d = n; d >= 0; --d) {
            uint64_t m = buckets[d];
            while (m) { order[si++] = __builtin_ctzll(m); m &= m - 1; }
        }
    }

    // BBMC-style bit-parallel coloring (San Segundo et al. 2011).
    // For each vertex v, instead of iterating over its colored neighbors to
    // build a "used_colors" bitmask, iterate over existing color classes and
    // check lc.members[c] & adj[v] == 0 with a single AND.
    // For dense graphs (density ~0.95) the number of color classes is 2-5,
    // so the inner while loop runs only 2-5 times vs. ~N neighbor iterations.
    // Also eliminates the 512-byte local_color[] stack array and processed mask.
    for (int i = 0; i < n; ++i) {
        int v = order[i];
        uint64_t adj_v = gs.adj[v];

        // Find smallest color class c such that no member is adjacent to v.
        int c = 0;
        while (c < lc.num_colors && (lc.members[c] & adj_v)) c++;

        lc.color[v]    = (uint8_t)c;
        lc.members[c] |= (1ULL << v);
        if (c >= lc.num_colors) lc.num_colors = c + 1;
        if (gs.weights[v] > lc.cmax[c]) lc.cmax[c] = gs.weights[v];
    }
    for (int c = 0; c < lc.num_colors; ++c) lc.ub += lc.cmax[c];
    return lc;
}

// Remove vertex v from the coloring incrementally.
// If v was the max of its color class, scan remaining members for new max.
// Returns the updated upper bound.
inline int lc_remove(LocalColorData& lc, int v, const GlobalState& gs) {
    int c = lc.color[v];
    lc.members[c] &= ~(1ULL << v);

    if (gs.weights[v] < lc.cmax[c]) return lc.ub;  // v not the class max; ub unchanged

    // v was the max: find new max from remaining members
    lc.ub -= lc.cmax[c];
    int new_max = 0;
    uint64_t m = lc.members[c];
    while (m) {
        int u = __builtin_ctzll(m);
        if (gs.weights[u] > new_max) new_max = gs.weights[u];
        m &= m - 1;
    }
    lc.cmax[c] = new_max;
    lc.ub += new_max;
    return lc.ub;
}


// ============================================================================
// MCQ (Maximum-Clique-based coloring Bound) — tight per-vertex child UB
// ============================================================================
// For branch vertex v with color c_v in the coloring of `cands`:
//
//   child_cands = cands & adj[v]   (after removing v)
//
// Key invariant (MCQ): when candidates are processed from highest color to
// lowest and each processed vertex is removed from `cands` before the next
// iteration, vertex v at step bi has:
//
//   cands = {verts[0..bi]}   (all vertices with color ≤ c_v still present)
//   child_cands ⊆ {verts[0..bi-1]} ∩ N(v)
//
// Since color class c_v is an independent set, no class-c_v member is in
// N(v), so child_cands uses only colors 0..c_v-1 (at most c_v classes).
// Therefore:
//
//   max_weight_clique(child_cands) ≤ Σ_{j=0}^{c_v-1} max_weight(class j ∩ child_cands)
//
// This is tighter than a naive sum over ALL K classes
// by exactly the contributions from classes c_v..K-1 that are excluded here.
//
// The upper bound is computed in O(|child_cands|).
inline int mcq_child_ub(
    uint64_t child_cands, int c_v,   // c_v = color of branch vertex v (0-indexed)
    const LocalColorData& lc, const GlobalState& gs)
{
    if (c_v <= 0 || child_cands == 0) return 0;

    // Vertices are globally reordered by descending weight, so in any bitmask
    // the smallest index bit corresponds to the heaviest vertex.
    // This lets us get each class max in O(1) with ctz, making child UB O(c_v)
    // instead of O(|child_cands|).
    int ub = 0;
    for (int j = 0; j < c_v; ++j) {
        uint64_t m = lc.members[j] & child_cands;
        if (m) ub += gs.weights[__builtin_ctzll(m)];
    }
    return ub;
}

// ============================================================================
// Phase 3-A: FPT non-edge branching (N == 24)
// ============================================================================
// Branching scheme: "include u / exclude u" vertex branching on the complement.
//
//   Pick u = vertex with maximum complement-degree in current candidates.
//   Branch A: include u → all non-neighbors of u are removed at once
//             (candidates_A = candidates & adj[u]).
//   Branch B: exclude u → remove only u
//             (candidates_B = candidates \ {u}).
//
// When complement-degree d_u ≥ 2, Branch A removes d_u vertices simultaneously,
// giving a tighter recursion tree than the plain binary non-edge branch (which
// removes only 1 vertex per branch). For d_u = 1 it is identical to the old scheme.
//
// Per-branch coloring UB pre-pruning via lc_remove (branch B: O(|class|);
// branch A: O(d_u × |class|)).
void fpt_dfs(uint64_t candidates, int cands_weight, int* __restrict__ cands_sum, GlobalState& gs) {
    if (candidates == 0) return;

    // Cheap O(1) pruning first.
    if (cands_weight <= gs.best_small) return;

    // Constraint pruning: if all remaining candidates can't satisfy rp, prune (O(PDIM))
    if (!v_geq(cands_sum, gs.rp)) return;

    const int cands_size = __builtin_popcountll(candidates);

    // Find u = vertex with max non-neighbor count in candidates (max complement-degree).
    // non_adj is computed over candidates excluding the vertex itself (adj has self-bit set).
    int u = -1;
    int max_non_adj_cnt = 0;
    {
        uint64_t scan = candidates;
        while (scan) {
            int i = __builtin_ctzll(scan);
            int cnt = cands_size - __builtin_popcountll(candidates & gs.adj[i]);
            if (cnt > max_non_adj_cnt) { max_non_adj_cnt = cnt; u = i; }
            scan &= scan - 1;
        }
    }

    if (max_non_adj_cnt == 0) {
        // All candidates are mutually adjacent → valid clique (leaf: skip lc_compute)
        if (cands_weight > gs.best_small) {
            gs.best_small = cands_weight;
            gs.best_mask = candidates;
        }
        return;
    }

    uint64_t non_adj_mask = candidates & ~gs.adj[u]; // non-neighbors of u in candidates

    // Tighter coloring upper bound (only for internal nodes that actually branch)
    LocalColorData lc = lc_compute(candidates, gs);
    if (lc.ub <= gs.best_small) return;

    // Branch A: include u → remove all non-neighbors of u at once
    {
        LocalColorData lcA = lc;
        int ubA = lcA.ub;
        uint64_t rm = non_adj_mask;
        while (rm) {
            int j = __builtin_ctzll(rm);
            ubA = lc_remove(lcA, j, gs);
            if (ubA <= gs.best_small) break;
            rm &= rm - 1;
        }
        if (ubA > gs.best_small) {
            // Remove non-adj vertices from cands_sum and cands_weight
            uint64_t cands_A = candidates & gs.adj[u]; // u stays (self-adj bit is set)
            int cw_A = cands_weight;
            rm = non_adj_mask;
            while (rm) {
                int j = __builtin_ctzll(rm);
                v_isub(cands_sum, gs.vp[j]);
                cw_A -= gs.weights[j];
                rm &= rm - 1;
            }
            fpt_dfs(cands_A, cw_A, cands_sum, gs);
            // Restore cands_sum
            rm = non_adj_mask;
            while (rm) {
                int j = __builtin_ctzll(rm);
                v_iadd(cands_sum, gs.vp[j]);
                rm &= rm - 1;
            }
        }
    }

    // Branch B: exclude u — pre-prune with coloring bound (modify lc in-place)
    {
        int ubB = lc_remove(lc, u, gs);
        if (ubB > gs.best_small) {
            v_isub(cands_sum, gs.vp[u]);
            fpt_dfs(candidates & ~(1ULL << u), cands_weight - gs.weights[u], cands_sum, gs);
            v_iadd(cands_sum, gs.vp[u]);
        }
    }
}

// ============================================================================
// Phase 3-A helper: pre-split FPT tree into small independent subtasks
// ============================================================================
// Expands the FPT tree shallowly to produce sub-problems with smaller
// candidate sets.  Smaller sets yield tighter lc_compute upper bounds inside
// fpt_dfs, dramatically reducing node counts.
// target_tasks controls depth: 4 gives 2 levels of branching (~4 leaves).
static void fpt_collect_tasks(
    uint64_t candidates,
    int cands_weight,
    int* __restrict__ cands_sum,
    GlobalState& gs,
    std::vector<FptTask>& tasks,
    int target_tasks
) {
    if (candidates == 0) return;

    // Cheap O(1) pruning first.
    if (cands_weight <= gs.best_small) return;

    if (!v_geq(cands_sum, gs.rp)) return;

    if ((int)tasks.size() >= target_tasks) {
        tasks.push_back({});
        FptTask& t = tasks.back();
        t.mask = candidates;
        t.cands_weight = cands_weight;
        memcpy(t.cands_sum, cands_sum, PDIM * sizeof(int));
        return;
    }

    // Same pivot as fpt_dfs: pick u with max complement-degree in candidates
    int u = -1;
    {
        const int cands_size = __builtin_popcountll(candidates);
        int max_non_adj_cnt = 0;
        uint64_t scan = candidates;
        while (scan) {
            int i = __builtin_ctzll(scan);
            int cnt = cands_size - __builtin_popcountll(candidates & gs.adj[i]);
            if (cnt > max_non_adj_cnt) { max_non_adj_cnt = cnt; u = i; }
            scan &= scan - 1;
        }
    }

    if (u == -1) {
        tasks.push_back({});
        FptTask& t = tasks.back();
        t.mask = candidates;
        t.cands_weight = cands_weight;
        memcpy(t.cands_sum, cands_sum, PDIM * sizeof(int));
        return;
    }

    // Branch A: include u (remove non-neighbors)
    {
        uint64_t non_adj_mask = candidates & ~gs.adj[u];
        uint64_t cands_A = candidates & gs.adj[u];
        int cw_A = cands_weight;
        uint64_t rm = non_adj_mask;
        while (rm) { int j = __builtin_ctzll(rm); v_isub(cands_sum, gs.vp[j]); cw_A -= gs.weights[j]; rm &= rm - 1; }
        fpt_collect_tasks(cands_A, cw_A, cands_sum, gs, tasks, target_tasks);
        rm = non_adj_mask;
        while (rm) { int j = __builtin_ctzll(rm); v_iadd(cands_sum, gs.vp[j]); rm &= rm - 1; }
    }

    // Branch B: exclude u
    v_isub(cands_sum, gs.vp[u]);
    fpt_collect_tasks(candidates & ~(1ULL << u), cands_weight - gs.weights[u], cands_sum, gs, tasks, target_tasks);
    v_iadd(cands_sum, gs.vp[u]);
}

// ============================================================================
// Phase 3-B: MCQ Branch & Bound  (main large-N solver, N > 24)
// ============================================================================
//
// Algorithm: Maximum-Clique-based coloring Bound (MCQ-W), weighted variant.
//   Originally described by Tomita & Seki (2003) for unweighted clique;
//   extended here to weighted maximum clique.
//
// Core idea vs. plain DSATUR bound
// ----------------------------------
// DSATUR assigns each candidate a color 0..K-1 and bounds the child clique
// weight by Σ_c cmax[c] (sum of max weights over ALL K color classes present
// in the child candidate set).
//
// MCQ tightens this: branch vertices are sorted by color *ascending* and
// processed in *descending* order (highest color first).  At iteration bi,
// the invariant holds:
//
//   cands = {verts[0], ..., verts[bi]}   (vertices with color ≤ color(verts[bi]))
//
// When branching on v = verts[bi] with color c_v:
//
//   child_cands = cands ∩ N(v) ∩ ¬{v}
//              ⊆ {verts[0..bi-1]}           (after removing v, colors < c_v or = c_v
//                                             but class c_v is independent ⇒ ≠ c_v)
//
// Therefore child_cands uses only colors 0..c_v-1 (at most c_v classes), and:
//
//   max_weight_clique(child_cands)  ≤  Σ_{j=0}^{c_v-1} max_weight(class j ∩ child_cands)
//
// vs. the DSATUR bound which sums up to K classes (K ≥ c_v in general).
// Gain = contributions from excluded classes c_v..K-1 → 30-50% fewer nodes.
//
// Additional pruning:
//   • After each recursion, lc_remove() updates the parent coloring UB.
//     Once T + parent_lc_ub ≤ best_T, ALL remaining (lower-color) branches
//     are pruned in a single break — the classic MCQ early-exit.
//   • Constraint-aware pruning (O(PDIM)) via cands_sum.
//
// Branches on ALL candidates (no BK pivot), so branching factor = |cands|,
// but with tight per-vertex bounds the effective factor is much smaller.
//
void dfs_mcq(
    uint64_t mask,
    uint64_t cands,
    int T,
    int* __restrict__ sum,
    int* __restrict__ cands_sum,
    GlobalState& gs
) {
    // ---- Feasibility update ----
    if (v_geq(sum, gs.rp)) {
        update_best_st(gs, T, mask);
    }
    if (cands == 0) return;

    // ---- Single-candidate shortcut (skip all sorting/coloring overhead) ----
    if (__builtin_popcountll(cands) == 1) {
        int v = __builtin_ctzll(cands);
        if (T + gs.weights[v] <= gs.best_T.load(std::memory_order_relaxed)) return;
        alignas(64) int new_sum[PDIM];
        v_add(new_sum, sum, gs.vp[v]);
        if (v_geq(new_sum, gs.rp)) update_best(gs, T + gs.weights[v], mask | (1ULL << v));
        return;
    }

    // ---- Constraint pruning: O(PDIM) via precomputed cands_sum ----
    {
        alignas(64) int max_sum[PDIM];
        v_add(max_sum, sum, cands_sum);
        if (!v_geq(max_sum, gs.rp)) return;
    }

    // ---- Greedy coloring of all candidates (DSATUR-style, degree-desc order) ----
    LocalColorData lc = lc_compute(cands, gs);

    // Quick UB check before sorting/iterating
    if (T + lc.ub <= gs.best_T.load(std::memory_order_relaxed)) return;

    // ---- Sort ALL candidates by color ascending (bucket sort, O(N)) ----
    // verts[0] has smallest color, verts[n-1] has largest color.
    // We will iterate from n-1 down to 0 (highest color processed first).
    uint8_t verts[64];
    int n = 0;
    {
        uint64_t buckets[64] = {};
        uint64_t tmp = cands;
        while (tmp) {
            int u = __builtin_ctzll(tmp);
            buckets[lc.color[u]] |= 1ULL << u;
            tmp &= tmp - 1;
        }
        for (int c = 0; c < lc.num_colors; ++c) {
            uint64_t m = buckets[c];
            // Iterate from highest bit to lowest so that within each color class
            // the last-added vertex (= verts[last_of_class], processed FIRST in
            // the MCQ loop) has the smallest index = highest weight.  This makes
            // the first recursive descent pick higher-weight vertices, establishing
            // a tighter best_T early and enabling more pruning in subsequent branches.
            while (m) { int b = 63 - __builtin_clzll(m); verts[n++] = b; m ^= 1ULL << b; }
        }
        // n == __builtin_popcountll(cands)
    }

    alignas(64) int new_cands_sum[PDIM];

    // Cache best_T for the entire loop iteration; refresh only after recursion.
    // Avoids repeated atomic loads that block compiler hoisting/CSE.
    int cur_best = gs.best_T.load(std::memory_order_relaxed);

    // ---- Main MCQ loop: process from verts[n-1] (highest color) down to verts[0] ----
    for (int bi = n - 1; bi >= 0; --bi) {
        int v    = verts[bi];
        int c_v  = lc.color[v];     // color of v (0-indexed)
        uint64_t vb = 1ULL << v;

        // child_cands = cands ∩ N(v) \ {v}
        // By the MCQ invariant (cands = {verts[0..bi]}), child_cands ⊆ {verts[0..bi-1]},
        // which uses only colors 0..c_v-1.  Class c_v is independent ⇒ no class-c_v
        // vertex in N(v), so the restriction is strict: colors ∈ [0, c_v-1].
        uint64_t child_cands = cands & gs.adj[v] & ~vb;

        // MCQ child upper bound: Σ_{j=0}^{c_v-1} max_weight(class j ∩ child_cands)
        // Strictly tighter than DSATUR (which would sum classes 0..K-1 ≥ c_v).
        int child_mcq_ub = mcq_child_ub(child_cands, c_v, lc, gs);

        if (T + gs.weights[v] + child_mcq_ub > cur_best) {
            // Compute incremental new_cands_sum for child:
            //   new_cands_sum = cands_sum − vp[v] − Σ_{u ∈ cands \ N(v)} vp[u]
            v_sub(new_cands_sum, cands_sum, gs.vp[v]);
            {
                uint64_t non_adj = cands & ~gs.adj[v];
                while (non_adj) {
                    v_isub(new_cands_sum, gs.vp[__builtin_ctzll(non_adj)]);
                    non_adj &= non_adj - 1;
                }
            }

            v_iadd(sum, gs.vp[v]);
            dfs_mcq(mask | vb, child_cands, T + gs.weights[v], sum, new_cands_sum, gs);
            v_isub(sum, gs.vp[v]);
            cur_best = gs.best_T.load(std::memory_order_relaxed);  // refresh after recursion
        }

        // Remove v from cands AFTER recursion — maintains the MCQ invariant for next bi.
        cands &= ~vb;
        v_isub(cands_sum, gs.vp[v]);

        // Incremental parent UB update via lc_remove (O(|class|)).
        // MCQ early-exit: once parent UB ≤ best_T, all remaining (lower-color)
        // branches have the same or smaller bound → prune them all at once.
        int parent_ub = lc_remove(lc, v, gs);
        if (T + parent_ub <= cur_best) break;
    }
}

// ============================================================================
// Main solve function
// ============================================================================
void solve(input_t &input, output_t &output) {
    io_struct::InitOutput(output);
    const int N = input.N;

    // OpenMP thread policy (contest server aligned: 4C/8T):
    // - If OMP_NUM_THREADS is explicitly set, respect it.
    // - Otherwise default to 8 threads.
    #ifdef _OPENMP
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(8);
    }
    #endif

    GlobalState gs;
    gs.best_T.store(-1, std::memory_order_relaxed);
    gs.best_mask = 0;
    gs.best_small = -1;

    // ---- Sort vertices by weight descending ----
    struct VInfo { int id; int w; };
    VInfo vs[64];
    for (int i = 0; i < N; ++i) {
        vs[i].id = i; vs[i].w = 0;
        for (int k = 0; k < PDIM; ++k) vs[i].w += input.v[i][k];
    }
    std::sort(vs, vs + N, [](const VInfo& a, const VInfo& b) { return a.w > b.w; });

    for (int i = 0; i < N; ++i) {
        int orig = vs[i].id;
        gs.weights[i] = vs[i].w;
        gs.reorder_map[i] = orig;
        memcpy(gs.vp[i], input.v[orig].data(), PDIM * sizeof(int));
    }
    memcpy(gs.rp, input.r.data(), PDIM * sizeof(int));

    // Build adjacency bitmasks
    memset(gs.adj, 0, sizeof(gs.adj));
    for (int i = 0; i < N; ++i) {
        int oi = vs[i].id;
        for (int j = 0; j < N; ++j)
            if (input.A[oi][vs[j].id]) gs.adj[i] |= (1ULL << j);
        gs.adj[i] |= (1ULL << i); // self-adjacent
    }

    // ---- Execution strategy ----
    if (N == 24) {
        // === N==24 FPT PATH ===
        // N==24, p=0.95: k = N - max_clique ≈ 4, so FPT tree has ~2^4 = 16 leaves.
        // Non-edge branching exploits the extreme density (complement is very sparse).
        uint64_t all_cands = (1ULL << N) - 1;

        // --- Greedy warm-start: set best_T before FPT search ---
        // Vertices are sorted by weight descending. Single greedy pass: O(N) cost,
        // negligible overhead. Tightens best_T before FPT tree starts.
        {
            uint64_t gc_mask = 0;
            int gc_T = 0;
            alignas(64) int gc_sum[PDIM];
            memset(gc_sum, 0, sizeof(gc_sum));
            uint64_t gc_adj = all_cands;
            for (int i = 0; i < N; ++i) {
                if (gc_mask == 0 || (gc_adj & (1ULL << i))) {
                    gc_mask |= (1ULL << i);
                    gc_T    += gs.weights[i];
                    v_iadd(gc_sum, gs.vp[i]);
                    gc_adj  &= gs.adj[i];
                }
            }
            if (v_geq(gc_sum, gs.rp) && gc_T > gs.best_small) {
                gs.best_small = gc_T;
                gs.best_mask = gc_mask;
            }
        }

        alignas(64) int init_cs[PDIM];
        memset(init_cs, 0, sizeof(init_cs));
        int init_cw = 0;
        { uint64_t tmp = all_cands; while (tmp) { int b = __builtin_ctzll(tmp); v_iadd(init_cs, gs.vp[b]); init_cw += gs.weights[b]; tmp &= tmp - 1; } }

        // Pre-split into small subtasks so each fpt_dfs call gets a smaller
        // candidate set → tighter lc_compute upper bounds → fewer nodes.
        // target_tasks=2: one shallow split level before handing off to fpt_dfs.
        std::vector<FptTask> tasks;
        tasks.reserve(4);
        fpt_collect_tasks(all_cands, init_cw, init_cs, gs, tasks, 2);

        for (const FptTask& task : tasks) {
            alignas(64) int cs[PDIM];
            memcpy(cs, task.cands_sum, PDIM * sizeof(int));
            fpt_dfs(task.mask, task.cands_weight, cs, gs);
        }

        gs.best_T.store(gs.best_small, std::memory_order_relaxed);
    } else {
        // === FALLBACK PATH (N != 24) ===
        // MCQ (Maximum-Clique-based coloring Bound) solver.
        // Outer symmetry-breaking loop: fix vertex i as the lexicographically
        // smallest clique member and restrict candidates to higher-indexed
        // neighbours only.

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int i = 0; i < N; ++i) {
            alignas(64) int ss[PDIM];
            memcpy(ss, gs.vp[i], PDIM * sizeof(int));

            uint64_t full = (N == 64) ? ~0ULL : (1ULL << N) - 1;
            uint64_t c = gs.adj[i] & full & (i == 63 ? 0 : ~((1ULL << (i + 1)) - 1));

            alignas(64) int cs[PDIM];
            memset(cs, 0, sizeof(cs));
            { uint64_t tmp = c; while (tmp) { v_iadd(cs, gs.vp[__builtin_ctzll(tmp)]); tmp &= tmp - 1; } }

            dfs_mcq(1ULL << i, c, gs.weights[i], ss, cs, gs);
        }
    }

    // ---- Output ----
    uint64_t fM = gs.best_mask;
    if (fM != 0) {
        // Recalculate T from mask to guarantee correctness
        int fT = 0;
        { uint64_t tmp = fM; while (tmp) { fT += gs.weights[__builtin_ctzll(tmp)]; tmp &= tmp - 1; } }

        output.T = fT;
        output.K_size = __builtin_popcountll(fM);
        int res[64];
        int cnt = 0;
        { uint64_t tmp = fM; while (tmp) { int i = __builtin_ctzll(tmp); res[cnt++] = gs.reorder_map[i] + 1; tmp &= tmp - 1; } }
        std::sort(res, res + cnt);
        for (int i = 0; i < output.K_size; ++i)
            output.members[i] = res[i];
    }
}