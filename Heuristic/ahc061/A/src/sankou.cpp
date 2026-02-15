#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt")

/**
 * AHC061 - Multi-Player Territory Game
 *
 * Algorithm: Chokudai Search (beam depth up to 6)
 *
 * Evaluation function (6 components):
 *   1. VL total (score numerator)
 *   2. Top player VL penalty (score denominator suppression)
 *   3. Approach bonus (proximity to top's high-value low-level cells)
 *   4. Frontier level differential (own level - enemy level at borders)
 *   5. Future strengthen potential (remaining gap * value * safety * reachability)
 *   6. Collision avoidance (distance & predicted target overlap penalty)
 *
 * Attack rule: only attack the top-scoring AI player.
 * Opponent sim: depth 0-1 all, depth 2 top only, depth 3+ skip.
 * All parameters overridable via environment variables for Optuna tuning.
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

// ===== Constants =====
const int DX[] = {-1, 1, 0, 0};
const int DY[] = {0, 0, -1, 1};

// ===== Game state =====
int N, M, T, U;
int VV[10][10];
int g_owner[10][10], g_lv[10][10];
int g_px[8], g_py[8];

chrono::high_resolution_clock::time_point G_START;

// ===== Top scorer tracking =====
int g_top_player = 1;
double g_my_vl = 0;
double g_top_vl = 0;

// ===== Tunable parameters (all env-var overridable for Optuna) =====

// ① VL total weight (fixed at 1.0, not tuned)
// ② Top player VL penalty
// ② Top player VL penalty
double P_W_FLD        = 1.712757;
double P_W_TOP        = 1.121621;
double P_TOP_PHASE    = 0.241533;
double P_TOP_RAMP     = 0.388142;
double P_DOMINANCE_W  = 1.030940;
double P_RATIO_SCALE  = 1.607460;
double P_RATIO_PHASE  = 0.506817;
double P_QE_CAPTURE   = 8.957569;
double P_QE_ATK_BONUS = 3.223586;
double P_QE_EMPTY_FUT = 0.353015;
double P_SAFE_MULT    = 0.149302;
double P_COL_NEAR     = 195.216619;
double P_COL_DIST2    = 88.848921;
double P_COL_TARGET   = 240.913531;
int    P_MAX_ITERS    = -1; // -1 means use time-based search




// ===== Parameter loading from environment =====
void read_params() {
    auto get = [](const char* name, double def) -> double {
        const char* v = getenv(name);
        return v ? atof(v) : def;
    };
    P_W_TOP           = get("P_W_TOP",           P_W_TOP);
    P_TOP_PHASE       = get("P_TOP_PHASE",       P_TOP_PHASE);
    P_TOP_RAMP        = get("P_TOP_RAMP",        P_TOP_RAMP);
    P_W_FLD           = get("P_W_FLD",           P_W_FLD);
    P_DOMINANCE_W     = get("P_DOMINANCE_W",     P_DOMINANCE_W);
    P_COL_NEAR        = get("P_COL_NEAR",        P_COL_NEAR);
    P_COL_DIST2       = get("P_COL_DIST2",       P_COL_DIST2);
    P_COL_TARGET      = get("P_COL_TARGET",      P_COL_TARGET);
    P_RATIO_SCALE     = get("P_RATIO_SCALE",     P_RATIO_SCALE);
    P_RATIO_PHASE     = get("P_RATIO_PHASE",     P_RATIO_PHASE);
    P_QE_CAPTURE      = get("P_QE_CAPTURE",      P_QE_CAPTURE);
    P_QE_ATK_BONUS    = get("P_QE_ATK_BONUS",    P_QE_ATK_BONUS);
    P_QE_EMPTY_FUT    = get("P_QE_EMPTY_FUT",    P_QE_EMPTY_FUT);
    P_SAFE_MULT       = get("P_SAFE_MULT",       P_SAFE_MULT);
    P_MAX_ITERS       = (int)get("P_MAX_ITERS",    (double)P_MAX_ITERS);
}



// ===== Compute current scores from game state =====
void compute_scores() {
    double scores[8] = {};
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (g_owner[i][j] >= 0)
                scores[g_owner[i][j]] += VV[i][j] * g_lv[i][j];

    g_my_vl = scores[0];
    g_top_player = 1;
    g_top_vl = 0;
    for (int p = 1; p < M; p++) {
        if (scores[p] > g_top_vl) {
            g_top_vl = scores[p];
            g_top_player = p;
        }
    }
}





inline double elapsed_sec() {
    return chrono::duration<double>(chrono::high_resolution_clock::now() - G_START).count();
}

// ===== I/O =====
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
    int8_t own[100];   // owner of each cell (-1=none, 0..M-1)
    int8_t lev[100];   // level of each cell
    int8_t px[8], py[8]; // player positions
    double score;
    double cached_vl;      // player 0's V*L sum (incremental)
    double cached_top_vl;  // top player's V*L sum (incremental)
    double cached_fld;     // Frontier Level Differential (incremental)
    int8_t fx, fy;         // first move (depth 0 choice)

    bool operator<(const BState& o) const { return score < o.score; }
};

// ===== Bitboard helpers for BFS visited =====
inline bool btest(const uint64_t vis[2], int id) {
    return (vis[id >> 6] >> (id & 63)) & 1;
}
inline void bset(uint64_t vis[2], int id) {
    vis[id >> 6] |= (1ULL << (id & 63));
}

// ===== BFS: get reachable candidate moves for a player =====
int get_cands(const BState& bs, int player, pair<int8_t,int8_t>* out) {
    uint64_t vis[2] = {0, 0};
    pair<int8_t,int8_t> bq[100];
    int qh = 0, qt = 0, cnt = 0;

    int sx = bs.px[player], sy = bs.py[player];
    bset(vis, sx * 10 + sy);
    bq[qt++] = {(int8_t)sx, (int8_t)sy};

    while (qh < qt) {
        int x = bq[qh].first, y = bq[qh].second;
        qh++;

        // Check if another player's piece blocks this cell
        bool blk = false;
        for (int p = 0; p < M; p++) {
            if (p != player && bs.px[p] == x && bs.py[p] == y) {
                blk = true;
                break;
            }
        }
        if (!blk) out[cnt++] = {(int8_t)x, (int8_t)y};

        // Expand through own territory
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

// ===== FLD Helper =====
inline double get_fld_part(const BState& bs, int x, int y) {
    int id = x * 10 + y;
    if (bs.own[id] != 0) return 0; // Only own cells contribute
    
    double v = VV[x][y];
    int l = bs.lev[id];
    double sum = 0;
    for (int d = 0; d < 4; d++) {
        int nx = x + DX[d], ny = y + DY[d];
        if (nx < 0 || nx >= N || ny < 0 || ny >= N) {
            sum += v * l * 0.25; // Wall
        } else {
            int nid = nx * 10 + ny;
            int o = bs.own[nid];
            if (o == 0) {
                sum += v * l * 0.25; // Own
            } else if (o > 0) {
                sum += v * (l - bs.lev[nid]) * 0.25; // Enemy
            }
        }
    }
    return sum;
}

// ===== Apply move with incremental cache update =====
inline void apply_move(BState& ns, int player, int tx, int ty) {
    // Calc FLD diff
    // Affected: (tx, ty) and neighbors
    int aff_x[5], aff_y[5];
    int aff_k = 0;
    aff_x[aff_k] = tx; aff_y[aff_k] = ty; aff_k++;
    for(int d=0; d<4; d++) {
        int nx = tx + DX[d], ny = ty + DY[d];
        if (nx>=0 && nx<N && ny>=0 && ny<N) {
             aff_x[aff_k] = nx; aff_y[aff_k] = ny; aff_k++;
        }
    }
    
    double fld_old = 0;
    for(int k=0; k<aff_k; k++) fld_old += get_fld_part(ns, aff_x[k], aff_y[k]);

    int id = tx * 10 + ty;
    int v = VV[tx][ty];

    if (ns.own[id] == -1) {
        // Capture empty cell
        ns.own[id] = player;
        ns.lev[id] = 1;
        ns.px[player] = tx;
        ns.py[player] = ty;
        if (player == 0) ns.cached_vl += v;
        if (player == g_top_player) ns.cached_top_vl += v;
    } else if (ns.own[id] == player) {
        // Strengthen own cell
        if (ns.lev[id] < U) {
            ns.lev[id]++;
            if (player == 0) ns.cached_vl += v;
            if (player == g_top_player) ns.cached_top_vl += v;
        }
        ns.px[player] = tx;
        ns.py[player] = ty;
    } else {
        // Attack enemy cell
        int old_owner = ns.own[id];
        ns.lev[id]--;
        if (ns.lev[id] == 0) {
            // Level drops to 0: capture
            if (old_owner == 0) ns.cached_vl -= v;
            if (old_owner == g_top_player) ns.cached_top_vl -= v;
            ns.own[id] = player;
            ns.lev[id] = 1;
            if (player == 0) ns.cached_vl += v;
            if (player == g_top_player) ns.cached_top_vl += v;
            ns.px[player] = tx;
            ns.py[player] = ty;
        } else {
            // Level reduced but not captured (attacker gets bounced)
            if (old_owner == 0) ns.cached_vl -= v;
            if (old_owner == g_top_player) ns.cached_top_vl -= v;
        }
    }
    
    // FLD update
    double fld_new = 0;
    for(int k=0; k<aff_k; k++) fld_new += get_fld_part(ns, aff_x[k], aff_y[k]);
    ns.cached_fld += (fld_new - fld_old);
}

// ===== Opponent simulation: all opponents (greedy) =====
void sim_opponents_full(BState& ns) {
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
            else                           ev = v * 0.65; // Expectation of [0.3, 1.0] is 0.65 for both wc and wd
            if (ev > best) { best = ev; bx = x; by = y; }
        }
        apply_move(ns, p, bx, by);
    }
}

// ===== Opponent simulation: top player only =====
void sim_top_only(BState& ns) {
    pair<int8_t,int8_t> opcands[100];
    int p = g_top_player;
    int nc = get_cands(ns, p, opcands);
    if (nc == 0) return;
    double best = -1e18;
    int bx = ns.px[p], by = ns.py[p];
    for (int i = 0; i < nc; i++) {
        int x = opcands[i].first, y = opcands[i].second;
        int id = x * 10 + y;
        double v = VV[x][y];
        double ev = 0;
        if (ns.own[id] == -1)          ev = v * 0.65;
        else if (ns.own[id] == p)      ev = (ns.lev[id] < U) ? v * 0.65 : 0;
        else                           ev = v * 0.65; // All equal expectation
        if (ev > best) { best = ev; bx = x; by = y; }
    }
    apply_move(ns, p, bx, by);
}


inline double quick_eval(const BState& bs, int tx, int ty, int turn) {
    double v = VV[tx][ty];
    double phase = (double)turn / T;
    int id = tx * 10 + ty;

    // Empty cell: claim it
    if (bs.own[id] == -1) {
        return v * (1.0 + P_QE_EMPTY_FUT * (U - 1)) * (1.5 - 0.5 * phase);
    }

    // Own cell: strengthen (with safety bonus)
    if (bs.own[id] == 0) {
        if (bs.lev[id] < U) {
            int safe = 0;
            for (int d = 0; d < 4; d++) {
                int nx = tx + DX[d], ny = ty + DY[d];
                if (nx < 0 || nx >= N || ny < 0 || ny >= N) safe++;
                else if (bs.own[nx * 10 + ny] == 0) safe++;
            }
            double safety_mult = 1.0 + safe * P_SAFE_MULT;
            return v * (0.8 + 0.6 * phase) * safety_mult;
        }
        return -1e9; // Already at max level
    }

    // Enemy territory: only attack top player
    if (bs.own[id] != g_top_player) {
        return -1e9; // Exclude non-top attacks
    }

    // Top player's cell: evaluate attack value
    int lev = bs.lev[id];
    int rem = T - turn;

    // Estimate capture time: roughly lev turns
    // If we have enough time and it's valuable, allow it.
    
    // Always allow Lv1 (high value)
    if (lev == 1) {
        double base = v * P_QE_CAPTURE;
        if (phase > 0.4)
            base += v * P_QE_ATK_BONUS * min(1.0, (phase - 0.4) * 2.0);
        return base;
    }

    // For Lv >= 2, check if we have enough turns
    // Heuristic: need `lev` turns to capture.
    if (rem >= lev + 2) { // +2 buffer for movement/mistakes
        // Base value: value * 1.0?
        // Devalue slightly because it takes time.
        // If v=16, lev=5, gaining 16 takes 5 turns. 3.2 per turn.
        // Only worth if V is high or it hurts opponent a lot.
        // For TOP player, hurting is worth double.
        // value 16 -> gain 16, opp lose 16 -> diff 32.
        // 32 / 5 = 6.4 per turn. Good.
        return v * 1.0 - (lev * 5.0); // Simple penalty for time cost
    }
    
    return -1e9; // Not enough time or not worth it
}

// ⑦ Dominance Bonus: V sum of empty cells reachable only by me (depth 4)
// Bitboard-based implementation (Optimized)
double calc_dominance(const BState& bs) {
    if (P_DOMINANCE_W < 0.001) return 0;
    
    using BB = unsigned __int128;
    BB my_trav = 0, top_trav = 0, empty_mask = 0;
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            int id = i*10 + j;
            int o = bs.own[id];
            BB bit = ((BB)1) << id;
            if (o == -1) {
                my_trav |= bit;
                top_trav |= bit;
                empty_mask |= bit;
            } else if (o == 0) {
                my_trav |= bit;
            } else if (o == g_top_player) {
                top_trav |= bit;
            }
        }
    }
    
    auto expand = [&](BB curr) -> BB {
        BB next = (curr >> 10) | (curr << 10);
        const BB col0 = ((BB)0x1004010040100401ULL) | (((BB)0x4010040ULL) << 64);
        const BB col9 = ((BB)0x802008020080200ULL) | (((BB)0x802008020ULL) << 64);
        next |= ((curr & ~col0) >> 1);
        next |= ((curr & ~col9) << 1);
        return next;
    };
    
    BB my_vis = ((BB)1) << (bs.px[0] * 10 + bs.py[0]);
    BB top_vis = ((BB)1) << (bs.px[g_top_player] * 10 + bs.py[g_top_player]);
    BB my_curr = my_vis, top_curr = top_vis;
    BB top_reached = top_vis;
    
    double score = 0;
    for(int d=0; d<4; d++) {
        my_curr = expand(my_curr) & my_trav & ~my_vis;
        my_vis |= my_curr;
        
        top_curr = expand(top_curr) & top_trav & ~top_vis;
        top_vis |= top_curr;
        top_reached |= top_curr;
        
        BB valid = my_curr & empty_mask & ~top_reached;
        uint64_t lo = (uint64_t)valid;
        uint64_t hi = (uint64_t)(valid >> 64);
        while(lo) {
            int id = __builtin_ctzll(lo);
            score += VV[id/10][id%10];
            lo &= (lo - 1);
        }
        while(hi) {
            int bit = __builtin_ctzll(hi);
            int id = bit + 64;
            score += VV[id/10][id%10];
            hi &= (hi - 1);
        }
    }
    return score;
}

// ===== Evaluate: 5-component weighted linear combination =====
// Components ①-④ + ratio computed here; ⑥ applied at beam extraction.
double evaluate(const BState& bs, int turn, int depth) {
    (void)depth;
    double phase = (double)turn / T;

    // ① VL total (from cache, no recomputation)
    double vl = bs.cached_vl;

    // ② Top player VL penalty (phase-gated with ramp)
    double top_vl = bs.cached_top_vl;
    double w_top = (phase > P_TOP_PHASE)
        ? P_W_TOP * min(1.0, (phase - P_TOP_PHASE) / max(0.01, P_TOP_RAMP))
        : 0.0;

    double w_fld = P_W_FLD * (0.3 + phase * 0.7);         // ④ grows late
    double fld = bs.cached_fld;

    // Score ratio bonus (S_0/S_A awareness)

    // Score ratio bonus (S_0/S_A awareness)
    double ratio_w = (phase > P_RATIO_PHASE)
        ? min(1.0, (phase - P_RATIO_PHASE) / 0.4) : 0.0;
    double ratio_bonus = 0;
    if (ratio_w > 0 && top_vl > 1.0) {
        ratio_bonus = (vl / top_vl) * vl * P_RATIO_SCALE;
    }

    // ⑦ Dominance
    double dominance = calc_dominance(bs);

    return vl                            // ① VL total
         - w_top * top_vl                // ② Top penalty
         + w_fld * fld                   // ④ Frontier diff
         + ratio_w * ratio_bonus         // Ratio bonus
         + dominance * P_DOMINANCE_W;    // ⑦ Dominance bonus
}

// ===== AI prediction for collision avoidance (⑥) =====
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

// ===== Chokudai Search =====
struct ScoredCand {
    double eval;
    int8_t cx, cy;
};

pair<int,int> choose_move(int turn) {
    const int MAX_D = min(6, T - turn);
    const int MAX_C = 8;       // beam width for depth 1+
    const int MAX_C0 = 20;     // beam width for depth 0

    // Time management
    double t0 = elapsed_sec();
    double t_rem = 1.93 - t0;
    int t_left = T - turn;
    int effective_left = max(1, t_left - 3);
    double t_per = t_rem / max(1, effective_left);
    double phase = (double)turn / T;
    if (phase < 0.5) t_per *= 1.2;  // Spend more time early
    double deadline = t0 + max(t_per, 0.002);

    if (P_MAX_ITERS > 0) deadline = 1e18; // Ignore time if using iter count

    compute_scores();

    // Initialize beams
    vector<priority_queue<BState>> beams(MAX_D + 1);

    BState root;
    root.cached_vl = 0;
    root.cached_top_vl = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int id = i * 10 + j;
            root.own[id] = (int8_t)g_owner[i][j];
            root.lev[id] = (int8_t)g_lv[i][j];
            if (g_owner[i][j] == 0)
                root.cached_vl += VV[i][j] * g_lv[i][j];
            else if (g_owner[i][j] == g_top_player)
                root.cached_top_vl += VV[i][j] * g_lv[i][j];
        }
    // Init cached_fld
    root.cached_fld = 0;
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
             root.cached_fld += get_fld_part(root, i, j);

    for (int p = 0; p < M; p++) {
        root.px[p] = (int8_t)g_px[p];
        root.py[p] = (int8_t)g_py[p];
    }
    root.fx = -1;
    root.fy = -1;
    root.score = evaluate(root, turn, 0);
    beams[0].push(root);

    pair<int8_t,int8_t> cds[100];
    ScoredCand sc[100];

    // Chokudai search loop
    int iter = 0;
    for (;;) {
        if (P_MAX_ITERS > 0) {
            if (iter >= P_MAX_ITERS) break;
        } else {
            if (!(iter & 3) && elapsed_sec() >= deadline) break;
        }
        iter++;

        bool any = false;
        for (int d = 0; d < MAX_D; d++) {
            if (beams[d].empty()) continue;

            BState bs = beams[d].top();
            beams[d].pop();

            int nc = get_cands(bs, 0, cds);
            if (nc == 0) continue;

            // Score and filter candidates
            int nsc = 0;
            for (int i = 0; i < nc; i++) {
                int cx = cds[i].first, cy = cds[i].second;
                int id = cx * 10 + cy;
                if (bs.own[id] == 0 && bs.lev[id] >= U) continue; // Skip max-level own cells
                sc[nsc++] = {quick_eval(bs, cx, cy, turn + d), (int8_t)cx, (int8_t)cy};
            }

            // Top-k selection
            int lim = min(nsc, (d == 0) ? MAX_C0 : MAX_C);
            if (nsc > lim) {
                nth_element(sc, sc + lim, sc + nsc,
                    [](const ScoredCand& a, const ScoredCand& b) {
                        return a.eval > b.eval;
                    });
            }

            // Expand top candidates
            for (int i = 0; i < lim; i++) {
                int cx = sc[i].cx, cy = sc[i].cy;
                BState ns = bs;

                apply_move(ns, 0, cx, cy);

                // Depth-limited opponent simulation
                if (d <= 1) {
                    sim_opponents_full(ns);   // depth 0-1: all opponents
                } else if (d == 2) {
                    sim_top_only(ns);         // depth 2: top only
                }
                // depth 3+: skip opponent simulation

                ns.fx = (d == 0) ? (int8_t)cx : bs.fx;
                ns.fy = (d == 0) ? (int8_t)cy : bs.fy;
                ns.score = evaluate(ns, turn + d + 1, d + 1);

                beams[d + 1].push(ns);
                any = true;
            }
        }
        if (!any) break;
    }

    // ===== Extract best move with ⑥ collision avoidance =====
    BState best;
    best.fx = -1;
    best.fy = -1;
    best.score = -1e18;

    for (int d = MAX_D; d >= 1; d--) {
        if (beams[d].empty()) continue;

        // Extract top candidates from deepest beam
        vector<BState> top;
        while (!beams[d].empty() && (int)top.size() < 30) {
            top.push_back(beams[d].top());
            beams[d].pop();
        }

        // Predict AI targets for collision check
        pair<int,int> ai_tgt[8];
        for (int p = 1; p < M; p++)
            ai_tgt[p] = predict_ai_now(p);

        // Apply collision penalties
        for (auto& bs : top) {
            for (int p = 1; p < M; p++) {
                int dist = abs(g_px[p] - (int)bs.fx) + abs(g_py[p] - (int)bs.fy);
                if (dist <= 1) bs.score -= P_COL_NEAR;
                else if (dist == 2) bs.score -= P_COL_DIST2;
                if (ai_tgt[p].first == (int)bs.fx && ai_tgt[p].second == (int)bs.fy)
                    bs.score -= P_COL_TARGET;
            }
        }

        // Pick best after collision adjustment
        for (const auto& bs : top) {
            if (bs.score > best.score) best = bs;
        }
        break;
    }

    if (best.fx >= 0) return {best.fx, best.fy};

    // Fallback: greedy on quick_eval
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
    for (int turn = 0; turn < T; turn++) {
        auto [tx, ty] = choose_move(turn);
        cout << tx << " " << ty << endl;
        read_turn();
    }
    return 0;
}
