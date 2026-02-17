#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt")

/**
 * AHC061 - Multi-Player Territory Game
 *
 * Algorithm: Chokudai Search with opponent simulation (optimized)
 *
 * === Optimizations applied ===
 *   1. Incremental V*L: cached_vl/cached_ovl in BState, O(1) update in apply_move
 *   2. Time check reduction: elapsed_sec() called every 4 iterations
 *   3. Bitboard BFS: uint64_t[2] replaces bool[100] for visited array
 *   4. Tiered evaluate: skip pex/pst at depth >= 3
 *   5. nth_element: O(n) top-k selection instead of O(n log n) sort
 *   6. #pragma GCC optimize + target for compiler-level speedup
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
using namespace std;

const int DX[] = {-1, 1, 0, 0};
const int DY[] = {0, 0, -1, 1};

int N, M, T, U;
int VV[10][10];

// True game state (read from tester each turn)
int g_owner[10][10], g_lv[10][10];
int g_px[8], g_py[8];

chrono::high_resolution_clock::time_point G_START;

// ===== Top scorer tracking =====
int g_top_player = 1;     // AI player with highest V*L
double g_my_vl = 0;       // player 0's V*L (from actual game state)
double g_top_vl = 0;      // top AI player's V*L
double g_second_vl = 0;   // second-highest AI V*L (for threshold detection)

// ===== Tunable parameters (overridable via environment variables for Optuna) =====
// Round 4 best (100 cases, 14 params: 9 base + 5 M/U)
double P_W2_BASE      = 0.192222;
double P_W3_MULT      = 1.128094;
double P_W7_MAX       = 0.751249;
double P_RATIO_SCALE  = 0.845678;
double P_W4_BASE      = 0.127005;
double P_W5_BASE      = 0.015127;
double P_REACH_DECAY  = 0.495684;
double P_QE_CAPTURE   = 2.604286;
double P_QE_ATK_BONUS = 1.723014;
double P_QE_EMPTY_FUT = 0.213099;
double P_COL_NEAR     = 281.223607;
double P_COL_TARGET   = 80.594577;
double P_W7_PHASE_START    = 0.226949;
double P_RATIO_PHASE_START = 0.155327;
// M/U adaptation: param = base + coeff * (M-5 or U-3)
double P_W2_BASE_U         = -0.008985;
double P_W3_MULT_U         = -0.033488;
double P_W7_MAX_M          = -0.006833;
double P_W4_BASE_M         = 0.023649;
double P_W7_PHASE_START_M  = -0.078900;

void read_params() {
    auto get = [](const char* name, double def) -> double {
        const char* v = getenv(name);
        return v ? atof(v) : def;
    };
    // Base parameters
    P_W2_BASE      = get("P_W2_BASE",      P_W2_BASE);
    P_W3_MULT      = get("P_W3_MULT",      P_W3_MULT);
    P_W7_MAX       = get("P_W7_MAX",       P_W7_MAX);
    P_RATIO_SCALE  = get("P_RATIO_SCALE",  P_RATIO_SCALE);
    P_W4_BASE      = get("P_W4_BASE",      P_W4_BASE);
    P_W5_BASE      = get("P_W5_BASE",      P_W5_BASE);
    P_REACH_DECAY  = get("P_REACH_DECAY",  P_REACH_DECAY);
    P_QE_CAPTURE   = get("P_QE_CAPTURE",   P_QE_CAPTURE);
    P_QE_ATK_BONUS = get("P_QE_ATK_BONUS", P_QE_ATK_BONUS);
    P_QE_EMPTY_FUT = get("P_QE_EMPTY_FUT", P_QE_EMPTY_FUT);
    P_COL_NEAR     = get("P_COL_NEAR",     P_COL_NEAR);
    P_COL_TARGET   = get("P_COL_TARGET",   P_COL_TARGET);
    P_W7_PHASE_START    = get("P_W7_PHASE_START",    P_W7_PHASE_START);
    P_RATIO_PHASE_START = get("P_RATIO_PHASE_START", P_RATIO_PHASE_START);
    // M/U coefficients
    P_W2_BASE_U        = get("P_W2_BASE_U",        P_W2_BASE_U);
    P_W3_MULT_U        = get("P_W3_MULT_U",        P_W3_MULT_U);
    P_W7_MAX_M         = get("P_W7_MAX_M",         P_W7_MAX_M);
    P_W4_BASE_M        = get("P_W4_BASE_M",        P_W4_BASE_M);
    P_W7_PHASE_START_M = get("P_W7_PHASE_START_M", P_W7_PHASE_START_M);
}

// Apply M/U adaptation after M and U are known
void apply_mu_adaptation() {
    P_W2_BASE          += P_W2_BASE_U        * (U - 3);
    P_W3_MULT          += P_W3_MULT_U        * (U - 3);
    P_W7_MAX           += P_W7_MAX_M         * (M - 5);
    P_W4_BASE          += P_W4_BASE_M        * (M - 5);
    P_W7_PHASE_START   += P_W7_PHASE_START_M * (M - 5);
    // Clamp to reasonable ranges
    P_W2_BASE        = max(0.0, P_W2_BASE);
    P_W3_MULT        = max(0.01, P_W3_MULT);
    P_W7_MAX         = max(0.0, P_W7_MAX);
    P_W4_BASE        = max(0.0, P_W4_BASE);
    P_W7_PHASE_START = max(0.05, min(0.9, P_W7_PHASE_START));
}

void compute_scores() {
    double scores[8] = {};
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (g_owner[i][j] >= 0)
                scores[g_owner[i][j]] += VV[i][j] * g_lv[i][j];

    g_my_vl = scores[0];
    g_top_player = 1;
    g_top_vl = 0;
    g_second_vl = 0;
    for (int p = 1; p < M; p++) {
        if (scores[p] > g_top_vl) {
            g_second_vl = g_top_vl;
            g_top_vl = scores[p];
            g_top_player = p;
        } else if (scores[p] > g_second_vl) {
            g_second_vl = scores[p];
        }
    }
}

inline double elapsed_sec() {
    return chrono::duration<double>(chrono::high_resolution_clock::now() - G_START).count();
}

void read_initial() {
    cin >> N >> M >> T >> U;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cin >> VV[i][j];
    for (int p = 0; p < M; p++)
        cin >> g_px[p] >> g_py[p];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            g_owner[i][j] = -1;
    memset(g_lv, 0, sizeof(g_lv));
    for (int p = 0; p < M; p++) {
        g_owner[g_px[p]][g_py[p]] = p;
        g_lv[g_px[p]][g_py[p]] = 1;
    }
}

void read_turn() {
    int a, b;
    for (int p = 0; p < M; p++) cin >> a >> b;
    for (int p = 0; p < M; p++) cin >> g_px[p] >> g_py[p];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cin >> g_owner[i][j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cin >> g_lv[i][j];
}

// ===== Beam State =====
struct BState {
    int8_t own[100];     // owner[r*10+c]: -1 or 0..M-1
    int8_t lev[100];     // level[r*10+c]: 0..U
    int8_t px[8], py[8]; // all player positions
    double score;
    double cached_vl;    // [OPT1] sum of V*L for player 0's cells
    double cached_ovl;   // [OPT1] sum of V*L for opponents' cells
    double cached_top_vl; // V*L of g_top_player specifically
    int8_t fx, fy;       // first move of player 0 from root

    bool operator<(const BState& o) const { return score < o.score; }
};

// ===== [OPT4] Bitboard helpers =====
inline bool btest(const uint64_t vis[2], int id) {
    return (vis[id >> 6] >> (id & 63)) & 1;
}
inline void bset(uint64_t vis[2], int id) {
    vis[id >> 6] |= (1ULL << (id & 63));
}

// ===== BFS: reachable candidates for a player =====
int get_cands(const BState& bs, int player, pair<int8_t,int8_t>* out) {
    uint64_t vis[2] = {0, 0};  // [OPT4] bitboard instead of bool[100]
    pair<int8_t,int8_t> bq[100];
    int qh = 0, qt = 0, cnt = 0;

    int sx = bs.px[player], sy = bs.py[player];
    bset(vis, sx * 10 + sy);
    bq[qt++] = {(int8_t)sx, (int8_t)sy};

    while (qh < qt) {
        int x = bq[qh].first, y = bq[qh].second;
        qh++;

        bool blk = false;
        for (int p = 0; p < M; p++) {
            if (p != player && bs.px[p] == x && bs.py[p] == y) {
                blk = true;
                break;
            }
        }
        if (!blk) out[cnt++] = {(int8_t)x, (int8_t)y};

        if (bs.own[x * 10 + y] == player) {
            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d], ny = y + DY[d];
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                    int nid = nx * 10 + ny;
                    if (!btest(vis, nid)) {
                        bset(vis, nid);
                        bq[qt++] = {(int8_t)nx, (int8_t)ny};
                    }
                }
            }
        }
    }
    return cnt;
}

// ===== [OPT1] Apply move with incremental V*L cache update =====
inline void apply_move(BState& ns, int player, int tx, int ty) {
    int id = tx * 10 + ty;
    int v = VV[tx][ty];

    if (ns.own[id] == -1) {
        // Claim empty cell
        ns.own[id] = player;
        ns.lev[id] = 1;
        ns.px[player] = tx;
        ns.py[player] = ty;
        if (player == 0) ns.cached_vl += v;
        else             ns.cached_ovl += v;
        if (player == g_top_player) ns.cached_top_vl += v;
    } else if (ns.own[id] == player) {
        // Strengthen own cell
        if (ns.lev[id] < U) {
            ns.lev[id]++;
            if (player == 0) ns.cached_vl += v;
            else             ns.cached_ovl += v;
            if (player == g_top_player) ns.cached_top_vl += v;
        }
        ns.px[player] = tx;
        ns.py[player] = ty;
    } else {
        // Attack enemy cell
        int old_owner = ns.own[id];
        ns.lev[id]--;
        if (ns.lev[id] == 0) {
            // Captured! Remove old owner's V*L, add new owner's V*1
            if (old_owner == 0) ns.cached_vl -= v;
            else                ns.cached_ovl -= v;
            if (old_owner == g_top_player) ns.cached_top_vl -= v;
            ns.own[id] = player;
            ns.lev[id] = 1;
            if (player == 0) ns.cached_vl += v;
            else             ns.cached_ovl += v;
            if (player == g_top_player) ns.cached_top_vl += v;
            ns.px[player] = tx;
            ns.py[player] = ty;
        } else {
            // Failed attack: level decreased by 1 → V*L decreased by V
            if (old_owner == 0) ns.cached_vl -= v;
            else                ns.cached_ovl -= v;
            if (old_owner == g_top_player) ns.cached_top_vl -= v;
            // Piece recalled to pre-move position (unchanged)
        }
    }
}

// ===== Simulate opponent moves (greedy) =====
void sim_opponents(BState& ns) {
    pair<int8_t,int8_t> opcands[100];

    for (int p = 1; p < M; p++) {
        int nc = get_cands(ns, p, opcands);
        if (nc == 0) continue;

        double best = -1e18;
        int bx = ns.px[p], by = ns.py[p];

        for (int i = 0; i < nc; i++) {
            int x = opcands[i].first, y = opcands[i].second;
            int id = x * 10 + y;
            double v = VV[x][y];
            double ev = 0;

            if (ns.own[id] == -1)          ev = v * 0.65;
            else if (ns.own[id] == p)      ev = (ns.lev[id] < U) ? v * 0.65 : 0;
            else                           ev = (ns.lev[id] == 1) ? v * 0.65 : v * 0.30;

            if (ev > best) { best = ev; bx = x; by = y; }
        }
        apply_move(ns, p, bx, by);
    }
}

// ===== Quick eval for candidate pre-filtering =====
inline double quick_eval(const BState& bs, int tx, int ty, int turn) {
    double v = VV[tx][ty];
    double phase = (double)turn / T;
    int id = tx * 10 + ty;

    if (bs.own[id] == -1) {
        return v * (1.0 + P_QE_EMPTY_FUT * (U - 1)) * (1.5 - 0.5 * phase);
    }
    if (bs.own[id] == 0) {
        if (bs.lev[id] < U) {
            int safe = 0;
            for (int d = 0; d < 4; d++) {
                int nx = tx + DX[d], ny = ty + DY[d];
                if (nx < 0 || nx >= N || ny < 0 || ny >= N) safe++;
                else if (bs.own[nx * 10 + ny] == 0) safe++;
            }
            return v * (0.8 + 0.6 * phase) * (0.5 + 0.5 * safe / 4.0);
        }
        return -1e9;
    }

    // ===== Enemy territory: attack evaluation =====
    int lev = bs.lev[id];
    int rem = T - turn;
    bool is_top = (bs.own[id] == g_top_player);

    // Level 1: one-hit capture (always valuable)
    if (lev == 1) {
        double base = v * P_QE_CAPTURE;
        // [ATTACK] Bonus for capturing top scorer's cells in late game
        // This directly reduces S_A (denominator of S_0/S_A)
        if (is_top && phase > 0.4)
            base += v * P_QE_ATK_BONUS * min(1.0, (phase - 0.4) * 2.0);
        return base;
    }

    // Level 2: only consider for top scorer in very late game
    if (is_top && lev == 2 && phase > 0.75 && rem >= 3) {
        return v * 0.8;
    }

    // Default: discourage level 2+ attacks (recall penalty too costly)
    if (rem <= 5) return v * 0.8 - max(0.0, (rem - 1.0) * 20.0);
    return v * 0.05 - 200;
}

// ===== [OPT1+OPT5] Evaluate with cached V*L and depth-based tiering =====
double evaluate(const BState& bs, int turn, int depth) {
    double phase = (double)turn / T;
    int rem = T - turn;

    // [OPT1] Use cached values instead of re-accumulating
    double vl = bs.cached_vl;
    // Note: cached_ovl no longer used; replaced by ratio-based top_vl evaluation
    double fut = 0, fld = 0, pex = 0, pst = 0;

    int p0x = bs.px[0], p0y = bs.py[0];
    // [OPT5] Skip pex/pst at deep depths (small weights, inaccurate anyway)
    bool calc_prox = (depth < 3);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int id = i * 10 + j;

            if (bs.own[id] == 0) {
                double v = VV[i][j];
                int l = bs.lev[id];

                // C2: Future strengthen potential
                if (l < U) {
                    int dist = abs(i - p0x) + abs(j - p0y);
                    double reach = 1.0 / (1.0 + dist * P_REACH_DECAY);
                    int gap = U - l;
                    double feas = min(1.0, (double)rem / max(1, gap + dist));
                    fut += v * gap * reach * feas;

                    // C5: Proximity to strengthen (mid-late, shallow only)
                    if (calc_prox && phase > 0.3 && dist > 0 && dist <= 4)
                        pst += v * gap / (1.0 + dist);
                }

                // [NEW2] Frontier Level Differential (replaces old C3: saf)
                // For each adjacent enemy cell, compute (my_level - enemy_level) * V
                // Positive = defensively strong, Negative = vulnerable
                // Wall/own-cell neighbors contribute +my_level (safe direction)
                for (int d = 0; d < 4; d++) {
                    int nx = i + DX[d], ny = j + DY[d];
                    if (nx < 0 || nx >= N || ny < 0 || ny >= N) {
                        fld += v * l * 0.25; // wall = safe
                    } else {
                        int nid = nx * 10 + ny;
                        if (bs.own[nid] == 0) {
                            fld += v * l * 0.25; // own cell = safe
                        } else if (bs.own[nid] > 0) {
                            // Enemy cell: level differential matters
                            fld += v * (l - bs.lev[nid]) * 0.25;
                        }
                        // Unowned (-1): neutral, fld += 0
                    }
                }

            } else if (calc_prox && phase < 0.7) {
                // C4: Expansion proximity (early-mid, shallow only)
                int dist = abs(i - p0x) + abs(j - p0y);
                if (dist > 0 && dist <= 5) {
                    double v = VV[i][j];
                    if (bs.own[id] == -1)
                        pex += v / (1.0 + dist);
                    else if (bs.own[id] > 0 && bs.lev[id] == 1)
                        pex += v * 0.5 / (1.0 + dist);
                }
            }
        }
    }

    // Phase-adaptive weights
    double w1 = 1.0;
    double w2 = P_W2_BASE * (1.0 - phase * 0.5);
    double w3 = P_W3_MULT * (0.3 + phase * 0.7); // now for fld (frontier level diff)

    // [ATTACK] Penalty for opponents' V*L
    double top_vl = bs.cached_top_vl;
    double w7 = (phase > P_W7_PHASE_START) ? P_W7_MAX * min(1.0, (phase - P_W7_PHASE_START) / 0.3) : 0.0;

    // [NEW1] Score Ratio Approximation
    // In the late game, transition from linear to ratio-based evaluation
    // ratio_bonus = vl / max(1, top_vl) directly approximates the scoring function
    double ratio_w = (phase > P_RATIO_PHASE_START) ? min(1.0, (phase - P_RATIO_PHASE_START) / 0.4) : 0.0;
    double ratio_bonus = 0;
    if (ratio_w > 0 && top_vl > 1.0) {
        // Normalize so it's in a similar scale to vl
        // target: higher vl/top_vl is better
        ratio_bonus = (vl / top_vl) * vl * P_RATIO_SCALE;
    }

    double result = w1 * vl + w2 * fut + w3 * fld - w7 * top_vl + ratio_w * ratio_bonus;

    if (calc_prox) {
        double w4 = P_W4_BASE * max(0.0, 1.0 - phase * 1.4);
        double w5 = P_W5_BASE * max(0.0, phase - 0.3) / 0.7;
        result += w4 * pex + w5 * pst;
    }

    return result;
}

// ===== AI prediction (current game state, for collision avoidance) =====
pair<int,int> predict_ai_now(int player) {
    uint64_t vis[2] = {0, 0};
    pair<int8_t,int8_t> bq[100], cds[100];
    int qh = 0, qt = 0, cnt = 0;

    int sx = g_px[player], sy = g_py[player];
    bset(vis, sx * 10 + sy);
    bq[qt++] = {(int8_t)sx, (int8_t)sy};

    while (qh < qt) {
        int x = bq[qh].first, y = bq[qh].second;
        qh++;

        bool blk = false;
        for (int p = 0; p < M; p++) {
            if (p != player && g_px[p] == x && g_py[p] == y) {
                blk = true;
                break;
            }
        }
        if (!blk) cds[cnt++] = {(int8_t)x, (int8_t)y};

        if (g_owner[x][y] == player) {
            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d], ny = y + DY[d];
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                    int nid = nx * 10 + ny;
                    if (!btest(vis, nid)) {
                        bset(vis, nid);
                        bq[qt++] = {(int8_t)nx, (int8_t)ny};
                    }
                }
            }
        }
    }

    double best = -1e18;
    int bx = sx, by = sy;
    for (int i = 0; i < cnt; i++) {
        int x = cds[i].first, y = cds[i].second;
        double v = VV[x][y];
        double ev = 0;
        if (g_owner[x][y] == -1)          ev = v * 0.65;
        else if (g_owner[x][y] == player) ev = (g_lv[x][y] < U) ? v * 0.65 : 0;
        else                              ev = (g_lv[x][y] == 1) ? v * 0.65 : v * 0.30;
        if (ev > best) { best = ev; bx = x; by = y; }
    }
    return {bx, by};
}

// ===== [OPT6] Scored candidate for nth_element =====
struct ScoredCand {
    double eval;
    int8_t cx, cy;
};

// ===== Chokudai Search =====
pair<int,int> choose_move(int turn) {
    const int MAX_D = min(6, T - turn);
    const int MAX_C = 8;
    const int MAX_C0 = 20;

    // Time management
    // Last ~3 turns have shallow search (MAX_D=1~3) and finish instantly,
    // so discount them from the budget to give more time to earlier turns.
    double t0 = elapsed_sec();
    double t_rem = 1.97 - t0;
    int t_left = T - turn;
    int effective_left = max(1, t_left - 3);
    double t_per = t_rem / max(1, effective_left);
    double phase = (double)turn / T;
    if (phase < 0.5) t_per *= 1.2;
    double deadline = t0 + max(t_per, 0.002);

    // [ATTACK] Identify top scorer for this turn
    compute_scores();

    // Priority queues
    vector<priority_queue<BState>> beams(MAX_D + 1);

    // Initialize root state with cached V*L values
    BState root;
    root.cached_vl = 0;
    root.cached_ovl = 0;
    root.cached_top_vl = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int id = i * 10 + j;
            root.own[id] = (int8_t)g_owner[i][j];
            root.lev[id] = (int8_t)g_lv[i][j];
            if (g_owner[i][j] == 0)
                root.cached_vl += VV[i][j] * g_lv[i][j];
            else if (g_owner[i][j] > 0) {
                root.cached_ovl += VV[i][j] * g_lv[i][j];
                if (g_owner[i][j] == g_top_player)
                    root.cached_top_vl += VV[i][j] * g_lv[i][j];
            }
        }
    for (int p = 0; p < M; p++) {
        root.px[p] = (int8_t)g_px[p];
        root.py[p] = (int8_t)g_py[p];
    }
    root.fx = -1;
    root.fy = -1;
    root.score = evaluate(root, turn, 0);
    beams[0].push(root);

    // Chokudai iterations
    pair<int8_t,int8_t> cds[100];
    ScoredCand sc[100];

    int iter = 0;
    for (;;) {
        // [OPT3] Check time every 4 iterations
        if (!(iter & 3) && elapsed_sec() >= deadline) break;
        iter++;

        bool any = false;
        for (int d = 0; d < MAX_D; d++) {
            if (beams[d].empty()) continue;

            BState bs = beams[d].top();
            beams[d].pop();

            int nc = get_cands(bs, 0, cds);
            if (nc == 0) continue;

            // Score candidates with quick_eval
            int nsc = 0;
            for (int i = 0; i < nc; i++) {
                int cx = cds[i].first, cy = cds[i].second;
                int id = cx * 10 + cy;
                if (bs.own[id] == 0 && bs.lev[id] >= U) continue;
                sc[nsc++] = {quick_eval(bs, cx, cy, turn + d), (int8_t)cx, (int8_t)cy};
            }

            // [OPT6] nth_element for O(n) top-k selection
            int lim = min(nsc, (d == 0) ? MAX_C0 : MAX_C);
            if (nsc > lim) {
                nth_element(sc, sc + lim, sc + nsc,
                    [](const ScoredCand& a, const ScoredCand& b) {
                        return a.eval > b.eval;
                    });
            }

            for (int i = 0; i < lim; i++) {
                int cx = sc[i].cx, cy = sc[i].cy;
                BState ns = bs;

                apply_move(ns, 0, cx, cy);

                // Opponent simulation at all depths (accuracy > speed)
                sim_opponents(ns);

                ns.fx = (d == 0) ? (int8_t)cx : bs.fx;
                ns.fy = (d == 0) ? (int8_t)cy : bs.fy;

                // [OPT5] Depth-aware evaluation
                ns.score = evaluate(ns, turn + d + 1, d + 1);

                beams[d + 1].push(ns);
                any = true;
            }
        }
        if (!any) break;
    }

    // Extract best from deepest non-empty beam
    BState best;
    best.fx = -1;
    best.fy = -1;
    best.score = -1e18;

    for (int d = MAX_D; d >= 1; d--) {
        if (beams[d].empty()) continue;

        vector<BState> top;
        while (!beams[d].empty() && (int)top.size() < 30) {
            top.push_back(beams[d].top());
            beams[d].pop();
        }

        // Collision avoidance
        pair<int,int> ai_tgt[8];
        for (int p = 1; p < M; p++)
            ai_tgt[p] = predict_ai_now(p);

        for (auto& bs : top) {
            for (int p = 1; p < M; p++) {
                int dist = abs(g_px[p] - (int)bs.fx) + abs(g_py[p] - (int)bs.fy);
                if (dist <= 1) bs.score -= P_COL_NEAR;
                else if (dist == 2) bs.score -= 60;
                if (ai_tgt[p].first == (int)bs.fx && ai_tgt[p].second == (int)bs.fy)
                    bs.score -= P_COL_TARGET;
            }
        }

        for (const auto& bs : top) {
            if (bs.score > best.score) best = bs;
        }
        break;
    }

    if (best.fx >= 0) return {best.fx, best.fy};

    // Fallback: greedy
    pair<int8_t,int8_t> fb_cds[100];
    int fnc = get_cands(root, 0, fb_cds);
    double fb_best = -1e18;
    int fbx = g_px[0], fby = g_py[0];
    for (int i = 0; i < fnc; i++) {
        int cx = fb_cds[i].first, cy = fb_cds[i].second;
        double ev = quick_eval(root, cx, cy, turn);
        if (ev > fb_best) { fb_best = ev; fbx = cx; fby = cy; }
    }
    return {fbx, fby};
}

// ===== Main =====
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    G_START = chrono::high_resolution_clock::now();
    read_params();

    read_initial();
    apply_mu_adaptation();
    for (int turn = 0; turn < T; turn++) {
        auto [tx, ty] = choose_move(turn);
        cout << tx << " " << ty << endl;
        read_turn();
    }
    return 0;
}
