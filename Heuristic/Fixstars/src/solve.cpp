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
#include <limits>
#include <cstring>
#include <iostream>
#include <atomic>
#include <numeric>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <climits>
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
    alignas(64) std::atomic<long long> best_T{-1};
    uint64_t best_mask = 0;

    uint64_t adj[64];          // adjacency bitmasks (reordered)
    long long weights[64];     // total weight per vertex
    alignas(64) int vp[64][PDIM]; // parameters per vertex
    alignas(64) int rp[PDIM];    // BOSS requirements
    int N;
    int reorder_map[64];       // reordered_id -> original_id
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
inline void update_best(GlobalState& gs, long long T, uint64_t mask) {
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

// ============================================================================
// Incremental local coloring for BnB branch loop
// ============================================================================
// Stores the complete local coloring so we can update it O(|class|) when
// one vertex is removed, instead of recoloring from scratch in O(|cands|²).
struct LocalColorData {
    int       color[64];     // color[v] = color class assigned to v
    long long cmax[64];      // cmax[c]  = max weight in color class c
    uint64_t  members[64];   // members[c] = bitmask of still-present vertices in class c
    int       num_colors;
    long long ub;            // current upper bound = sum(cmax[c])
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
    if (n > 1) {
        std::sort(order, order + n, [&](int a, int b) {
            return __builtin_popcountll(cands & gs.adj[a]) > __builtin_popcountll(cands & gs.adj[b]);
        });
    }

    int local_color[64];
    memset(local_color, -1, sizeof(local_color));
    uint64_t processed = 0;

    for (int i = 0; i < n; ++i) {
        int v = order[i];

        uint64_t used_colors = 0;
        uint64_t colored_nbrs = processed & gs.adj[v];
        while (colored_nbrs) {
            int u = __builtin_ctzll(colored_nbrs);
            used_colors |= (1ULL << local_color[u]);
            colored_nbrs &= colored_nbrs - 1;
        }

        int c = __builtin_ctzll(~used_colors);
        local_color[v] = c;
        lc.color[v]     = c;
        lc.members[c]  |= (1ULL << v);
        if (c >= lc.num_colors) lc.num_colors = c + 1;
        if (gs.weights[v] > lc.cmax[c]) lc.cmax[c] = gs.weights[v];

        processed |= (1ULL << v);
    }
    for (int c = 0; c < lc.num_colors; ++c) lc.ub += lc.cmax[c];
    return lc;
}

// Remove vertex v from the coloring incrementally.
// If v was the max of its color class, scan remaining members for new max.
// Returns the updated upper bound.
inline long long lc_remove(LocalColorData& lc, int v, const GlobalState& gs) {
    int c = lc.color[v];
    lc.members[c] &= ~(1ULL << v);

    if (gs.weights[v] < lc.cmax[c]) return lc.ub;  // v not the class max; ub unchanged

    // v was the max: find new max from remaining members
    lc.ub -= lc.cmax[c];
    long long new_max = 0;
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



// Compute upper bound for child_cands using the parent's LocalColorData in O(|child_cands|).
// child_cands ⊆ cands, so the parent's coloring is still a valid coloring for child_cands
// (removing vertices never introduces new edges). The bound is slightly looser than a
// fresh local recoloring but costs O(n) instead of O(n²).
inline long long child_ub_from_parent_lc(
    uint64_t child_cands, const LocalColorData& lc, const GlobalState& gs)
{
    long long color_max[64] = {};
    uint64_t tmp = child_cands;
    while (tmp) {
        int u = __builtin_ctzll(tmp);
        int c = lc.color[u];
        if (gs.weights[u] > color_max[c]) color_max[c] = gs.weights[u];
        tmp &= tmp - 1;
    }
    long long ub = 0;
    for (int c = 0; c < lc.num_colors; ++c) ub += color_max[c];
    return ub;
}

// Variant of lc_compute that uses precomputed deg[] (= |cands & adj[v]| per vertex).
// Avoids recomputing popcounts already done during pivot selection.
inline LocalColorData lc_compute_with_deg(uint64_t cands, const int* deg, const GlobalState& gs) {
    LocalColorData lc;
    memset(&lc, 0, sizeof(lc));
    if (cands == 0) return lc;

    int order[64];
    int n = 0;
    { uint64_t tmp = cands; while (tmp) { order[n++] = __builtin_ctzll(tmp); tmp &= tmp - 1; } }
    if (n == 1) {
        int v = order[0];
        lc.color[v] = 0; lc.members[0] = 1ULL << v;
        lc.cmax[0] = gs.weights[v]; lc.num_colors = 1; lc.ub = gs.weights[v];
        return lc;
    }
    std::sort(order, order + n, [&](int a, int b) { return deg[a] > deg[b]; });

    int local_color[64];
    memset(local_color, -1, sizeof(local_color));
    uint64_t processed = 0;
    for (int i = 0; i < n; ++i) {
        int v = order[i];
        uint64_t used_colors = 0;
        uint64_t colored_nbrs = processed & gs.adj[v];
        while (colored_nbrs) {
            int u = __builtin_ctzll(colored_nbrs);
            used_colors |= (1ULL << local_color[u]);
            colored_nbrs &= colored_nbrs - 1;
        }
        int c = __builtin_ctzll(~used_colors);
        local_color[v] = c;
        lc.color[v]    = c;
        lc.members[c] |= (1ULL << v);
        if (c >= lc.num_colors) lc.num_colors = c + 1;
        if (gs.weights[v] > lc.cmax[c]) lc.cmax[c] = gs.weights[v];
        processed |= (1ULL << v);
    }
    for (int c = 0; c < lc.num_colors; ++c) lc.ub += lc.cmax[c];
    return lc;
}

// ============================================================================
// Each non-edge (u,v) in G corresponds to an edge in complement G'.
// At least one of u or v must be excluded from the clique.
// Branching heuristic: pick u with fewest neighbors (most constrained),
// then v = heaviest non-neighbor.
void fpt_dfs(uint64_t candidates, GlobalState& gs) {
    if (candidates == 0) return;

    // Weight upper bound: sum of weights of all candidates
    long long ub = 0;
    {
        uint64_t tmp = candidates;
        while (tmp) { ub += gs.weights[__builtin_ctzll(tmp)]; tmp &= tmp - 1; }
    }
    if (ub <= gs.best_T.load(std::memory_order_relaxed)) return;

    // Constraint upper bound: if even ALL candidates together fail rp, prune
    {
        alignas(64) int max_sum[PDIM] = {};
        uint64_t tmp = candidates;
        while (tmp) { v_iadd(max_sum, gs.vp[__builtin_ctzll(tmp)]); tmp &= tmp - 1; }
        if (!v_geq(max_sum, gs.rp)) return;
    }

    // Find a non-edge (u,v) in candidates
    int u = -1, v = -1;
    {
        int min_adj = INT_MAX;
        uint64_t scan = candidates;
        while (scan) {
            int i = __builtin_ctzll(scan);
            uint64_t non_adj = candidates & ~gs.adj[i];
            if (non_adj) {
                int adj_cnt = __builtin_popcountll(candidates & gs.adj[i]);
                if (adj_cnt < min_adj) {
                    min_adj = adj_cnt;
                    u = i;
                    v = -1;
                    long long best_w = -1;
                    uint64_t na = non_adj;
                    while (na) {
                        int j = __builtin_ctzll(na);
                        if (gs.weights[j] > best_w) { best_w = gs.weights[j]; v = j; }
                        na &= na - 1;
                    }
                }
            }
            scan &= scan - 1;
        }
    }

    if (u == -1) {
        // All candidates are mutually adjacent → valid clique
        update_best(gs, ub, candidates);
        return;
    }

    // Branch 1: exclude v
    fpt_dfs(candidates & ~(1ULL << v), gs);
    // Branch 2: exclude u
    fpt_dfs(candidates & ~(1ULL << u), gs);
}

// Expand the FPT tree shallowly to collect independent subtasks for parallel execution.
// Stops expanding a node when:
//   (a) no non-edge remains (leaf clique) → add to tasks, OR
//   (b) tasks.size() >= target_tasks → add as-is for fpt_dfs to finish.
static void fpt_collect_tasks(
    uint64_t candidates,
    GlobalState& gs,
    std::vector<uint64_t>& tasks,
    int target_tasks
) {
    if (candidates == 0) return;

    // Prune by weight upper bound
    long long ub = 0;
    { uint64_t t = candidates; while (t) { ub += gs.weights[__builtin_ctzll(t)]; t &= t-1; } }
    if (ub <= gs.best_T.load(std::memory_order_relaxed)) return;

    // If we already have enough tasks, stop expanding and queue the rest for fpt_dfs
    if ((int)tasks.size() >= target_tasks) {
        tasks.push_back(candidates);
        return;
    }

    // Find a non-edge (same heuristic as fpt_dfs)
    int u = -1, v = -1;
    {
        int min_adj = INT_MAX;
        uint64_t scan = candidates;
        while (scan) {
            int i = __builtin_ctzll(scan);
            uint64_t non_adj = candidates & ~gs.adj[i];
            if (non_adj) {
                int adj_cnt = __builtin_popcountll(candidates & gs.adj[i]);
                if (adj_cnt < min_adj) {
                    min_adj = adj_cnt;
                    u = i;
                    v = -1; long long bw = -1;
                    uint64_t na = non_adj;
                    while (na) { int j = __builtin_ctzll(na); if (gs.weights[j] > bw) { bw = gs.weights[j]; v = j; } na &= na-1; }
                }
            }
            scan &= scan - 1;
        }
    }

    if (u == -1) {
        // Leaf: all candidates are mutually adjacent
        tasks.push_back(candidates);
        return;
    }

    fpt_collect_tasks(candidates & ~(1ULL << v), gs, tasks, target_tasks);
    fpt_collect_tasks(candidates & ~(1ULL << u), gs, tasks, target_tasks);
}

// ============================================================================
// Fast DFS for dense small graphs
// ============================================================================
void fast_dfs(
    uint64_t current_mask,
    uint64_t candidates,
    long long current_T,
    int* __restrict__ current_sum,
    GlobalState& gs
) {
    // 1. Feasibility check (SIMD)
    if (v_geq(current_sum, gs.rp)) {
        update_best(gs, current_T, current_mask);
    }

    if (candidates == 0) return;

    // 2. Local coloring upper bound (tighter than rem_max)
    LocalColorData lc = lc_compute(candidates, gs);
    if (current_T + lc.ub <= gs.best_T.load(std::memory_order_relaxed)) return;

    // 3. Constraint pruning: sum + all candidates still can't satisfy rp?
    {
        alignas(64) int max_sum[PDIM];
        memcpy(max_sum, current_sum, PDIM * sizeof(int));
        uint64_t tmp2 = candidates;
        while (tmp2) {
            v_iadd(max_sum, gs.vp[__builtin_ctzll(tmp2)]);
            tmp2 &= tmp2 - 1;
        }
        if (!v_geq(max_sum, gs.rp)) return;
    }

    while(candidates) {
        int v = __builtin_ctzll(candidates);
        uint64_t v_bit = 1ULL << v;
        alignas(64) int next_sum[PDIM];
        v_add(next_sum, current_sum, gs.vp[v]);

        fast_dfs(current_mask | v_bit, candidates & gs.adj[v] & ~(v_bit | (v_bit - 1)), current_T + gs.weights[v], next_sum, gs);

        candidates &= ~v_bit;
        long long parent_ub = lc_remove(lc, v, gs);
        if (current_T + parent_ub <= gs.best_T.load(std::memory_order_relaxed)) break;
    }
}

// ============================================================================
// Phase 3: DFS Branch & Bound with Bron-Kerbosch Pivot
// ============================================================================

// cands_sum: sum of vp[u] for all u in cands (maintained incrementally).
// This makes the feasibility pruning O(PDIM) instead of O(|cands|*PDIM).
//
// lc_ub: incremental upper bound from LocalColorData passed in by parent.
//   The parent already has a valid local coloring for `cands`; we pass it
//   down to avoid recomputing from scratch at the start of this call.
void dfs_bnb(
    uint64_t mask,
    uint64_t cands,
    long long T,
    int* __restrict__ sum,
    int* __restrict__ cands_sum,
    GlobalState& gs,
    long long lc_ub   // local_color upper bound already computed by parent
) {
    if (v_geq(sum, gs.rp)) {
        update_best(gs, T, mask);
    }
    if (cands == 0) return;

    // Weight upper bound: use lc_ub passed in from parent (already computed)
    if (T + lc_ub <= gs.best_T.load(std::memory_order_relaxed)) return;

    // Single candidate shortcut: skip pivot/sort/lc_compute machinery entirely
    if (__builtin_popcountll(cands) == 1) {
        int v = __builtin_ctzll(cands);
        if (T + gs.weights[v] <= gs.best_T.load(std::memory_order_relaxed)) return;
        alignas(64) int max_sum[PDIM];
        v_add(max_sum, sum, gs.vp[v]);
        if (v_geq(max_sum, gs.rp)) update_best(gs, T + gs.weights[v], mask | (1ULL << v));
        return;
    }

    // Constraint-aware pruning: O(PDIM) using precomputed cands_sum
    // max_sum = sum (current clique) + cands_sum (all candidate params)
    {
        alignas(64) int max_sum[PDIM];
        v_add(max_sum, sum, cands_sum);
        if (!v_geq(max_sum, gs.rp)) return;
    }

    // === Compute deg[] once: reused by both pivot selection and lc_compute ===
    // deg[u] = |cands ∩ adj[u]| for each u in cands
    int deg[64];
    int pivot = -1;
    int max_adj_count = -1;
    {
        uint64_t tmp = cands;
        while (tmp) {
            int u = __builtin_ctzll(tmp);
            int d = __builtin_popcountll(cands & gs.adj[u]);
            deg[u] = d;
            if (d > max_adj_count) { max_adj_count = d; pivot = u; }
            tmp &= tmp - 1;
        }
    }

    // Branch set: candidates NOT adjacent to pivot, PLUS the pivot itself
    uint64_t branch_set = (cands & ~gs.adj[pivot]) | (1ULL << pivot);

    // Collect and sort branch vertices by weight descending
    int branch_verts[64];
    int n_branch = 0;
    {
        uint64_t tmp = branch_set;
        while (tmp) {
            branch_verts[n_branch++] = __builtin_ctzll(tmp);
            tmp &= tmp - 1;
        }
        if (n_branch > 1) {
            std::sort(branch_verts, branch_verts + n_branch, [&](int a, int b) {
                return gs.weights[a] > gs.weights[b];
            });
        }
    }

    // Workspace for child cands_sum (allocated once per call, reused per branch)
    alignas(64) int new_cands_sum[PDIM];

    // Build LocalColorData once for `cands` using precomputed deg[] (no extra popcounts).
    // We will incrementally remove branch vertices as we iterate.
    LocalColorData lc = lc_compute_with_deg(cands, deg, gs);

    // Re-check weight upper bound with the tighter local coloring
    // (lc.ub <= lc_ub passed from parent, so this may prune more)
    if (T + lc.ub <= gs.best_T.load(std::memory_order_relaxed)) return;

    // Branch on each vertex in branch_set
    for (int bi = 0; bi < n_branch; ++bi) {
        int v = branch_verts[bi];
        uint64_t vb = 1ULL << v;

        // Compute new_cands_sum for the child call:
        //   new_cands = cands & adj[v] & ~{v}
        //   new_cands_sum = cands_sum - vp[v] - sum(vp[u] for u in cands & ~adj[v])
        // Use v_sub to fuse memcpy+v_isub into one SIMD pass.
        v_sub(new_cands_sum, cands_sum, gs.vp[v]);
        {
            uint64_t non_adj = cands & ~gs.adj[v];
            while (non_adj) {
                v_isub(new_cands_sum, gs.vp[__builtin_ctzll(non_adj)]);
                non_adj &= non_adj - 1;
            }
        }

        // Compute child's lc_ub using parent's coloring: O(|child_cands|) instead of O(|cands|²).
        // child_cands ⊆ cands so parent lc is a valid (possibly non-tight) coloring for it.
        long long child_lc_ub = child_ub_from_parent_lc(cands & gs.adj[v] & ~vb, lc, gs);

        // Add vertex v to clique and recurse
        v_iadd(sum, gs.vp[v]);
        dfs_bnb(mask | vb, cands & gs.adj[v] & ~vb, T + gs.weights[v], sum, new_cands_sum, gs, child_lc_ub);
        v_isub(sum, gs.vp[v]);

        // Remove v from candidates for subsequent branches and update cands_sum
        cands &= ~vb;
        v_isub(cands_sum, gs.vp[v]);

        // Incrementally remove v from the local coloring → O(|class|) instead of O(|cands|²)
        long long parent_ub = lc_remove(lc, v, gs);
        if (T + parent_ub <= gs.best_T.load(std::memory_order_relaxed)) break;
    }
}

// ============================================================================
// Main solve function
// ============================================================================
void solve(input_t &input, output_t &output) {
    io_struct::InitOutput(output);
    const int N = input.N;

    // Ensure OpenMP uses all available threads (4 cores × 2 HT = 8 threads)
    #ifdef _OPENMP
    omp_set_num_threads(8);
    #endif

    GlobalState gs;
    gs.N = N;
    gs.best_T.store(-1, std::memory_order_relaxed);
    gs.best_mask = 0;

    // ---- Sort vertices by weight descending ----
    struct VInfo { int id; long long w; };
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

    // ---- Execution strategy depends on problem size ----
    if (N <= 24) {
        // === SMALL FPT PATH ===
        // N<=24, p=0.95: k = N - max_clique ≈ 4, so FPT tree has ~2^4 = 16 leaves.
        // Expand the tree shallowly to collect >= 16 independent subtasks,
        // then dispatch them via omp parallel for (dynamic) across 8 threads.
        {
            uint64_t all_cands = (1ULL << N) - 1;

            #ifdef _OPENMP
            const int n_threads = 8;
            #else
            const int n_threads = 1;
            #endif
            // Collect 2x thread count tasks for good dynamic load balancing
            const int target_tasks = n_threads * 2;

            std::vector<uint64_t> tasks;
            tasks.reserve(target_tasks * 2);
            fpt_collect_tasks(all_cands, gs, tasks, target_tasks);

            const int ntasks = (int)tasks.size();
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
            #endif
            for (int ti = 0; ti < ntasks; ++ti) {
                fpt_dfs(tasks[ti], gs);
            }
        }
    } else {
        // === LARGE PATH (N > 24) ===
        // Strategy: minimal greedy for initial bound, then full BnB.
        // SA is removed - every millisecond goes to BnB with BK pivot.

        // Full BnB
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(8)
        #endif
        for (int i = 0; i < N; ++i) {

            alignas(64) int ss[PDIM];
            memcpy(ss, gs.vp[i], PDIM * sizeof(int));

            uint64_t full = (N == 64) ? ~0ULL : (1ULL << N) - 1;
            uint64_t c = gs.adj[i] & full & (i == 63 ? 0 : ~((1ULL << (i + 1)) - 1));

            alignas(64) int cs[PDIM];
            memset(cs, 0, sizeof(cs));
            { uint64_t tmp = c; while (tmp) { v_iadd(cs, gs.vp[__builtin_ctzll(tmp)]); tmp &= tmp-1; } }

            dfs_bnb(1ULL << i, c, gs.weights[i], ss, cs, gs, lc_compute(c, gs).ub);
        }
    }

    // ---- Output ----
    uint64_t fM = gs.best_mask;
    if (fM != 0) {
        // Recalculate T from mask to guarantee correctness
        long long fT = 0;
        for (int i = 0; i < N; ++i)
            if (fM & (1ULL << i)) fT += gs.weights[i];

        output.T = fT;
        output.K_size = __builtin_popcountll(fM);
        int res[64];
        int cnt = 0;
        for (int i = 0; i < N; ++i)
            if (fM & (1ULL << i)) res[cnt++] = gs.reorder_map[i] + 1;
        std::sort(res, res + cnt);
        for (int i = 0; i < output.K_size; ++i)
            output.members[i] = res[i];
    }
}