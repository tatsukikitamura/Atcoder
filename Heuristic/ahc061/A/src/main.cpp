#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt")

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
using namespace std;

// ===================== GLOBALS =====================
int N, M, T, U;
int V[100]; // 1D array for value map
double invT; // Inverted T for faster division

// XorShift Random Number Generator
struct XorShift {
    using result_type = uint32_t;
    uint32_t x = 123456789;
    uint32_t y = 362436069;
    uint32_t z = 521288629;
    uint32_t w = 88675123;
    
    XorShift(uint32_t seed = 0) {
        set_seed(seed);
    }
    
    void set_seed(uint32_t seed) {
        x = 123456789; y = 362436069; z = 521288629; w = 88675123;
        x ^= seed;
        y ^= (x << 13);
        z ^= (y >> 17);
        w ^= (z << 5);
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 0xFFFFFFFF; }
    
    inline result_type operator()() {
        uint32_t t = x ^ (x << 11);
        x = y; y = z; z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }
    
    inline double nextDouble() { // [0, 1]
        return (double)operator()() / 4294967295.0;
    }

    void seed(uint32_t s) { set_seed(s); }
};

XorShift rng;
auto programStart = chrono::steady_clock::now();

double elapsedMs() {
    return chrono::duration_cast<chrono::microseconds>(
        chrono::steady_clock::now() - programStart).count() * 0.001;
}

// ===================== TUNABLE HYPERPARAMETERS =====================
struct HyperParams {
    double phase1 = 0.27159946;
    double phase2 = 0.70847988;

    double wa_early = 0.23526016, wb_early = 0.36359119, wc_early = 0.46131418, wd_early = 0.49415821;
    double wa_mid   = 0.75300129, wb_mid   = 0.72386218, wc_mid   = 0.55583898, wd_mid   = 0.21382323;
    double wa_late  = 0.61319270, wb_late  = 0.78662693, wc_late  = 0.76681554, wd_late  = 0.63228284;

    double leader_mult = 1.30321790;
    double ucb_c = 0.68105000;

    double eval_expand = 0.00000069;
    double eval_level  = 0.00000490;
    double eval_reach  = 0.00000390;
    double eval_attack = 0.00000016;
    double eval_trap   = 0.01;

    int rollout_depth = 6;

    int num_particles = 464;
    double pf_noise_w = 0.01800440;
    double pf_noise_eps = 0.00046352;

    double u_wb_boost = 0.75473319;
    double u_wd_penalty = 0.30301423;

    double m_leader_scale = 0.21663018;
} HP[6]; // U=1 to 5 (index 0 unused)

void adaptParams() {
    double uFactor = max(0.0, (U - 2.0) / 3.0);
    HP[U].wb_mid  += HP[U].u_wb_boost * uFactor;
    HP[U].wb_late += HP[U].u_wb_boost * uFactor;
    HP[U].wd_mid  -= HP[U].u_wd_penalty * uFactor;
    HP[U].wd_late -= HP[U].u_wd_penalty * uFactor;
    HP[U].wd_mid  = max(0.01, HP[U].wd_mid);
    HP[U].wd_late = max(0.01, HP[U].wd_late);

    HP[U].leader_mult += HP[U].m_leader_scale * max(0, M - 2);

    if (M >= 6) {
        HP[U].rollout_depth = min(HP[U].rollout_depth, 10);
        HP[U].num_particles = min(HP[U].num_particles, 120);
    } else if (M <= 3) {
        HP[U].rollout_depth = min(HP[U].rollout_depth + 8, 30);
    }

    if (U == 1) {
        HP[U].wb_early = 0.0; HP[U].wb_mid = 0.0; HP[U].wb_late = 0.0;
        HP[U].phase1 = 0.5; HP[U].phase2 = 0.5;
    }
}

// ===================== BITBOARD UTILS =====================
using Bitboard = unsigned __int128;
Bitboard MASK_ALL = 0;
Bitboard MASK_NOT_COL_0 = 0;
Bitboard MASK_NOT_COL_9 = 0;

void initBitboards() {
    MASK_ALL = ((Bitboard)1 << 100) - 1;
    MASK_NOT_COL_0 = MASK_ALL;
    MASK_NOT_COL_9 = MASK_ALL;
    for (int i = 0; i < 10; i++) {
        MASK_NOT_COL_0 &= ~((Bitboard)1 << (i * 10));
        MASK_NOT_COL_9 &= ~((Bitboard)1 << (i * 10 + 9));
    }
}

// ★改善3: ビット抽出の2段階化ユーティリティ
template<typename F>
inline void forEachBit(Bitboard b, F&& func) {
    uint64_t lo = (uint64_t)b;
    while (lo) {
        int idx = __builtin_ctzll(lo);
        lo &= lo - 1; // 最下位ビットクリア
        func(idx);
    }
    uint64_t hi = (uint64_t)(b >> 64);
    while (hi) {
        int idx = 64 + __builtin_ctzll(hi);
        hi &= hi - 1;
        func(idx);
    }
}

inline int popcount128(Bitboard b) {
    uint64_t lo = (uint64_t)b;
    uint64_t hi = (uint64_t)(b >> 64);
    return __builtin_popcountll(lo) + __builtin_popcountll(hi);
}

// ===================== STATE =====================
struct State {
    Bitboard owner_mask[10];
    int8_t owner_map[100];
    uint8_t level[100];
    uint8_t p_pos[10];
    double totalScore[10]; // ★改善2: スコア差分更新用

    void init() {
        for(int i=0; i<10; i++) { owner_mask[i] = 0; totalScore[i] = 0.0; }
        memset(owner_map, -1, sizeof(owner_map));
        memset(level, 0, sizeof(level));
    }

    // ★改善2: O(1)スコア取得
    double score(int p) const {
        return totalScore[p];
    }



    // totalScoreをowner_mask/level/owner_mapから再構築
    void rebuildTotalScore() {
        for(int p=0; p<10; p++) totalScore[p] = 0.0;
        for(int i=0; i<100; i++) {
            int o = owner_map[i];
            if (o != -1) totalScore[o] += (double)V[i] * level[i];
        }
    }
};

// ===================== BFS / MOVABLE SET =====================
struct MoveList {
    uint8_t idx[200];
    int cnt;
};

inline Bitboard getReachableMask(const State& s, int p) {
    Bitboard current = ((Bitboard)1 << s.p_pos[p]);
    Bitboard visited = current;
    Bitboard my_area = s.owner_mask[p];

    while (true) {
        Bitboard next = current;
        next |= ((current & MASK_NOT_COL_0) >> 1);
        next |= ((current & MASK_NOT_COL_9) << 1);
        next |= (current >> 10);
        next |= (current << 10);

        next &= my_area;
        next &= ~visited;

        if (next == 0) break;

        visited |= next;
        current = next;
    }
    return visited;
}

void getMovable(const State& s, int p, MoveList& ml) {
    Bitboard reach = getReachableMask(s, p);

    Bitboard candidates = reach;
    candidates |= ((reach & MASK_NOT_COL_0) >> 1);
    candidates |= ((reach & MASK_NOT_COL_9) << 1);
    candidates |= (reach >> 10);
    candidates |= (reach << 10);

    Bitboard occupied = 0;
    for(int i=0; i<M; i++) if(i != p) occupied |= ((Bitboard)1 << s.p_pos[i]);

    candidates &= ~occupied;
    candidates &= MASK_ALL;

    // ★改善3: 2段階ビット抽出
    ml.cnt = 0;
    forEachBit(candidates, [&](int idx) {
        ml.idx[ml.cnt++] = idx;
    });
}

// ===================== TURN SIMULATION =====================
// ★改善2 + 改善4: totalScore差分更新 + cellCnt除去
void simulateTurn(State& s, int moves[8]) {
    int prevPos[8];
    for (int p = 0; p < M; p++) {
        prevPos[p] = s.p_pos[p];
        s.p_pos[p] = moves[p];
    }

    bool recalled[8] = {};

    // ★改善4: O(M^2)衝突判定 (cellCnt[100]除去)
    for (int p = 0; p < M; p++) {
        int cell = s.p_pos[p];
        bool hasConflict = false;
        for (int q = 0; q < M; q++) {
            if (q != p && s.p_pos[q] == cell) { hasConflict = true; break; }
        }
        if (!hasConflict) continue;

        int co = s.owner_map[cell];
        bool ownerHere = false;
        for (int q = 0; q < M; q++) {
            if (s.p_pos[q] == cell && q == co) { ownerHere = true; break; }
        }

        if (ownerHere && co >= 0) {
            if (p != co) recalled[p] = true;
        } else {
            recalled[p] = true;
        }
    }

    for (int p = 0; p < M; p++) {
        if (recalled[p]) continue;
        int idx = s.p_pos[p];
        int owner = s.owner_map[idx];

        if (owner == -1) {
            // 空きセルを獲得
            s.owner_mask[p] |= ((Bitboard)1 << idx);
            s.owner_map[idx] = p;
            s.level[idx] = 1;
            s.totalScore[p] += (double)V[idx] * 1; // ★改善2
        } else if (owner == p) {
            // 自分のセルを強化
            if (s.level[idx] < U) {
                s.totalScore[p] += (double)V[idx] * 1; // ★改善2: level差分=1
                s.level[idx]++;
            }
        } else {
            // 敵セルを攻撃: levelを1下げる
            s.totalScore[owner] -= (double)V[idx] * 1; // ★改善2: 敵スコア減少
            s.level[idx]--;
            if (s.level[idx] == 0) {
                // 奪取成功
                s.owner_mask[owner] &= ~((Bitboard)1 << idx);
                s.owner_mask[p] |= ((Bitboard)1 << idx);
                s.owner_map[idx] = p;
                s.level[idx] = 1;
                s.totalScore[p] += (double)V[idx] * 1; // ★改善2: 新所有者スコア加算
            } else {
                // 奪取失敗 → リコール (levelの減少は維持される)
                recalled[p] = true;
            }
        }
    }

    for (int p = 0; p < M; p++) {
        if (recalled[p]) s.p_pos[p] = prevPos[p];
    }
}

// ===================== PARTICLE FILTER =====================
struct Particle { double wa, wb, wc, wd, eps; };

vector<vector<Particle>> particles;
vector<vector<double>> pweights;
vector<Particle> estimated;

void initParticlesFor(int p) {
    uniform_real_distribution<double> dw(0.3, 1.0), de(0.1, 0.5);
    particles[p].resize(HP[U].num_particles);
    pweights[p].assign(HP[U].num_particles, 1.0 / HP[U].num_particles);
    for (int k = 0; k < HP[U].num_particles; k++)
        particles[p][k] = {dw(rng), dw(rng), dw(rng), dw(rng), de(rng)};
    estimated[p] = {0.65, 0.65, 0.65, 0.65, 0.3};
}

void initAllParticles() {
    particles.resize(M);
    pweights.resize(M);
    estimated.resize(M);
    for (int p = 1; p < M; p++) initParticlesFor(p);
}

double aiEvalCell(const State& s, int p, int idx,
                  double wa, double wb, double wc, double wd) {
    int o = s.owner_map[idx];
    if (o == -1) return V[idx] * wa;
    if (o == p) return (s.level[idx] < U) ? V[idx] * wb : 0.0;
    return (s.level[idx] == 1) ? V[idx] * wc : V[idx] * wd;
}

// ★改善1: getMovableを外部から受け取るバージョン
double computeLikelihoodCached(const State& pre, int p, int obsIdx,
                                const Particle& pt, const MoveList& ml) {
    if (ml.cnt == 0) return 1.0;

    bool found = false;
    for (int k = 0; k < ml.cnt; k++)
        if (ml.idx[k] == obsIdx) { found = true; break; }
    if (!found) return 1e-10;

    double maxVal = -1e18;
    int maxCnt = 0;
    bool obsIsMax = false;
    for (int k = 0; k < ml.cnt; k++) {
        double val = aiEvalCell(pre, p, ml.idx[k], pt.wa, pt.wb, pt.wc, pt.wd);
        if (val > maxVal + 1e-12) {
            maxVal = val; maxCnt = 1;
            obsIsMax = (ml.idx[k] == obsIdx);
        } else if (val > maxVal - 1e-12) {
            maxCnt++;
            if (ml.idx[k] == obsIdx) obsIsMax = true;
        }
    }

    double prob = pt.eps / ml.cnt;
    if (obsIsMax) prob += (1.0 - pt.eps) / maxCnt;
    return max(prob, 1e-10);
}

// ★改善1: getMovableをループ外で1回だけ呼ぶ
void updateParticles(const State& pre, int p, int obsX, int obsY) {
    int obsIdx = obsX * 10 + obsY;
    int NP = HP[U].num_particles;

    // ★改善1: 1回だけ計算してキャッシュ
    MoveList ml;
    getMovable(pre, p, ml);

    double sumW = 0;
    for (int k = 0; k < NP; k++) {
        double lik = computeLikelihoodCached(pre, p, obsIdx, particles[p][k], ml);
        pweights[p][k] *= lik;
        sumW += pweights[p][k];
    }

    if (sumW < 1e-30) { initParticlesFor(p); return; }
    double invSumW = 1.0 / sumW;
    for (int k = 0; k < NP; k++) pweights[p][k] *= invSumW;

    // Systematic resampling
    static Particle newP[500];
    static double cumW[500];
    cumW[0] = pweights[p][0];
    for (int k = 1; k < NP; k++) cumW[k] = cumW[k-1] + pweights[p][k];

    double invNP = 1.0 / NP;
    uniform_real_distribution<double> uni(0.0, invNP);
    double u = uni(rng);
    int idx = 0;
    for (int k = 0; k < NP; k++) {
        double target = u + (double)k * invNP;
        while (idx < NP - 1 && cumW[idx] < target) idx++;
        newP[k] = particles[p][idx];
    }

    normal_distribution<double> nw(0, HP[U].pf_noise_w), ne(0, HP[U].pf_noise_eps);
    for (int k = 0; k < NP; k++) {
        newP[k].wa  = clamp(newP[k].wa  + nw(rng), 0.3, 1.0);
        newP[k].wb  = clamp(newP[k].wb  + nw(rng), 0.3, 1.0);
        newP[k].wc  = clamp(newP[k].wc  + nw(rng), 0.3, 1.0);
        newP[k].wd  = clamp(newP[k].wd  + nw(rng), 0.3, 1.0);
        newP[k].eps = clamp(newP[k].eps + ne(rng), 0.1, 0.5);
    }

    particles[p].assign(newP, newP + NP);
    pweights[p].assign(NP, invNP);

    estimated[p] = {0, 0, 0, 0, 0};
    for (int k = 0; k < NP; k++) {
        estimated[p].wa += newP[k].wa; estimated[p].wb += newP[k].wb;
        estimated[p].wc += newP[k].wc; estimated[p].wd += newP[k].wd;
        estimated[p].eps += newP[k].eps;
    }
    estimated[p].wa *= invNP; estimated[p].wb *= invNP;
    estimated[p].wc *= invNP; estimated[p].wd *= invNP;
    estimated[p].eps *= invNP;
}

// ===================== AI MOVE GENERATION =====================
int genAIMove(const State& s, int p, const Particle& param, XorShift& lr) {
    MoveList ml;
    getMovable(s, p, ml);
    if (ml.cnt == 0) return s.p_pos[p];

    if (lr.nextDouble() < param.eps) return ml.idx[lr() % ml.cnt];

    double bestVal = -1e18;
    int bestIdx[200], bestCnt = 0;
    for (int k = 0; k < ml.cnt; k++) {
        double val = aiEvalCell(s, p, ml.idx[k], param.wa, param.wb, param.wc, param.wd);
        if (val > bestVal + 1e-12) { bestVal = val; bestCnt = 0; bestIdx[bestCnt++] = k; }
        else if (val > bestVal - 1e-12) { bestIdx[bestCnt++] = k; }
    }
    int c = bestIdx[lr() % bestCnt];
    return ml.idx[c];
}

// ===================== EVALUATION FUNCTION =====================
double evaluate(const State& s, int currentTurn) {
    // ★改善2: score()がO(1)
    double s0 = s.score(0);
    double sa = 0;
    for (int p = 1; p < M; p++) sa = max(sa, s.score(p));
    if (sa < 1e-9) return 20.0;
    double ratio = s0 / sa;

    Bitboard reachMask = getReachableMask(s, 0);
    int myReach = popcount128(reachMask);

    double expandPot = 0, levelPot = 0, attackPot = 0;

    // ★改善3: 2段階ビット抽出
    forEachBit(reachMask, [&](int i) {
        if (s.owner_map[i] == 0 && s.level[i] < U)
            levelPot += V[i] * (U - s.level[i]);

        auto checkNeighbor = [&](int ni) {
            int o = s.owner_map[ni];
            if (o == -1) expandPot += V[ni];
            else if (o > 0 && s.level[ni] == 1) attackPot += V[ni];
        };

        if (i % 10 != 0) checkNeighbor(i - 1);
        if (i % 10 != 9) checkNeighbor(i + 1);
        if (i >= 10)     checkNeighbor(i - 10);
        if (i < 90)      checkNeighbor(i + 10);
    });

    double remain = (double)(T - currentTurn) * invT;
    double baseScore = log2(1.0 + ratio)
           + HP[U].eval_expand * expandPot * remain
           + HP[U].eval_level * levelPot * remain
           + HP[U].eval_reach * myReach
           + HP[U].eval_attack * attackPot * remain;

    // ★追加: M=2のとき、相手の移動可能領域（拡張余地）を減らすと高評価
    if (M <= 3) {
        // 相手(p=1)の到達可能マスク
        Bitboard oppReach = getReachableMask(s, 1);
        Bitboard oppExpand = 0;
        oppExpand |= ((oppReach & MASK_NOT_COL_0) >> 1);
        oppExpand |= ((oppReach & MASK_NOT_COL_9) << 1);
        oppExpand |= (oppReach >> 10);
        oppExpand |= (oppReach << 10);
        oppExpand &= ~oppReach;
        oppExpand &= MASK_ALL;
        
        // 相手の拡張可能数
        int oppMobility = popcount128(oppExpand);
        // Mobilityを減らす方向へ
        // remainを掛けるかどうかは議論があるが、eval_expandと対称性を持たせるなら掛ける
        return baseScore - HP[U].eval_trap * oppMobility * remain;
    }

    return baseScore;
}

// ===================== PLAYER 0 GREEDY (rollout policy) =====================
int greedyMove0(const State& s, int turn, XorShift& lr) {
    MoveList ml;
    getMovable(s, 0, ml);
    if (ml.cnt == 0) return s.p_pos[0];

    double phase = (double)turn * invT;
    double wa0, wb0, wc0, wd0;
    if (phase < HP[U].phase1) {
        wa0 = HP[U].wa_early; wb0 = HP[U].wb_early; wc0 = HP[U].wc_early; wd0 = HP[U].wd_early;
    } else if (phase < HP[U].phase2) {
        wa0 = HP[U].wa_mid; wb0 = HP[U].wb_mid; wc0 = HP[U].wc_mid; wd0 = HP[U].wd_mid;
    } else {
        wa0 = HP[U].wa_late; wb0 = HP[U].wb_late; wc0 = HP[U].wc_late; wd0 = HP[U].wd_late;
    }

    // ★改善2: score()がO(1)
    int leader = 1; double leaderSc = s.score(1);
    for (int p = 2; p < M; p++) { double sc = s.score(p); if (sc > leaderSc) { leaderSc = sc; leader = p; } }

    double bestVal = -1e18;
    int bestIdx[200], bestCnt = 0;
    for (int k = 0; k < ml.cnt; k++) {
        int idx = ml.idx[k];
        int o = s.owner_map[idx];
        double val;
        if (o == -1) val = V[idx] * wa0;
        else if (o == 0) val = (s.level[idx] < U) ? V[idx] * wb0 : -0.01;
        else {
            double mult = (o == leader) ? HP[U].leader_mult : 1.0;
            val = ((s.level[idx] == 1) ? V[idx] * wc0 : V[idx] * wd0) * mult;
        }
        if (val > bestVal + 1e-12) { bestVal = val; bestCnt = 0; bestIdx[bestCnt++] = k; }
        else if (val > bestVal - 1e-12) { bestIdx[bestCnt++] = k; }
    }
    int c = bestIdx[lr() % bestCnt];
    return ml.idx[c];
}

// ===================== ROLLOUT =====================
double rollout(State s, int startTurn, XorShift& lr) {
    int depth = HP[U].rollout_depth;

    // Adaptive Depth for small M (2-3 players) where depth is large
    if (HP[U].rollout_depth > 10) {
        double progress = (double)startTurn * invT;
        if (progress > 0.6) {
             // Linearly decay depth from 100% to 50%
            double decay = 1.0 - (progress - 0.6); 
            depth = max(6, (int)(HP[U].rollout_depth * decay));
        }
    }

    // ★追加: ロールアウト開始前にサンプリング
    Particle sampled[10];
    for (int p = 1; p < M; p++) {
        // int k = uniform_int_distribution<int>(0, HP.num_particles - 1)(lr);
        int k = lr() % HP[U].num_particles; // 高速な代替
        sampled[p] = particles[p][k];
    }

    int endTurn = min(T, startTurn + depth - 1);
    int moves[8];
    for (int t = startTurn; t <= endTurn; t++) {
        moves[0] = greedyMove0(s, t, lr);
        for (int p = 1; p < M; p++) {
            // ★変更: estimated[p] → sampled[p]
            moves[p] = genAIMove(s, p, sampled[p], lr);
        }
        simulateTurn(s, moves);
    }
    return evaluate(s, endTurn);
}

// ===================== MCTS / SEARCH =====================
pair<int,int> selectMove(const State& rootState, int currentTurn) {
    MoveList ml;
    getMovable(rootState, 0, ml);
    if (ml.cnt == 0) return {rootState.p_pos[0]/10, rootState.p_pos[0]%10};

    double scores[200] = {};
    int visits[200] = {};

    double elapsed = elapsedMs();

    int loops = 0;
    while (true) {
        loops++;
        if ((loops & 127) == 0) {
            double now = elapsedMs();
            if (now > 1995) break;
            double budgetMs = (1995 - now) / (T - currentTurn + 1);
            if (now - elapsed > budgetMs) break;
        }

        int bestIdx = -1;
        double bestUcb = -1e18;
        int total = loops - 1;

        for(int k=0; k<ml.cnt; k++) {
            if (visits[k] == 0) { bestIdx = k; break; }
            double avg = scores[k] / visits[k];
            double ucb = avg + HP[U].ucb_c * sqrt(log(total) / visits[k]);
            if (ucb > bestUcb) { bestUcb = ucb; bestIdx = k; }
        }
        if (bestIdx == -1) bestIdx = rng() % ml.cnt;

        State nextS = rootState;
        int moves[8];
        moves[0] = ml.idx[bestIdx];
        for(int p=1; p<M; p++) moves[p] = genAIMove(rootState, p, estimated[p], rng);
        simulateTurn(nextS, moves);

        double sc = rollout(nextS, currentTurn + 1, rng);

        scores[bestIdx] += sc;
        visits[bestIdx]++;
    }

    int bestK = 0;
    double maxAvg = -1e18;
    for(int k=0; k<ml.cnt; k++) {
        if (visits[k] == 0) continue;
        double a = scores[k] / visits[k];
        if (a > maxAvg) { maxAvg = a; bestK = k; }
    }

    int idx = ml.idx[bestK];
    return {idx/10, idx%10};
}

// ===================== PARAMETER LOADING =====================
void loadParams(const string& filename) {
    FILE* fp = fopen(filename.c_str(), "r");
    if (!fp) {
        cerr << "Failed to open parameter file: " << filename << endl;
        return;
    }
    char key[256];
    double val;
    while (fscanf(fp, "%s %lf", key, &val) == 2) {
        string k(key);
        // Check for "uX_" prefix
        int target_u = -1; // -1 means all
        string name = k;

        if (k.size() > 3 && k[0] == 'u' && isdigit(k[1]) && k[2] == '_') {
            target_u = k[1] - '0';
            name = k.substr(3);
        }

        // Apply to specific U or all U
        int start_u = (target_u != -1) ? target_u : 1;
        int end_u   = (target_u != -1) ? target_u : 5;

        for (int u = start_u; u <= end_u; u++) {
            if (u < 1 || u > 5) continue;

            if (name == "phase1") HP[u].phase1 = val;
            else if (name == "phase2") HP[u].phase2 = val;

            else if (name == "wa_early") HP[u].wa_early = val;
            else if (name == "wb_early") HP[u].wb_early = val;
            else if (name == "wc_early") HP[u].wc_early = val;
            else if (name == "wd_early") HP[u].wd_early = val;
            
            else if (name == "wa_mid") HP[u].wa_mid = val;
            else if (name == "wb_mid") HP[u].wb_mid = val;
            else if (name == "wc_mid") HP[u].wc_mid = val;
            else if (name == "wd_mid") HP[u].wd_mid = val;

            else if (name == "wa_late") HP[u].wa_late = val;
            else if (name == "wb_late") HP[u].wb_late = val;
            else if (name == "wc_late") HP[u].wc_late = val;
            else if (name == "wd_late") HP[u].wd_late = val;

            else if (name == "leader_mult") HP[u].leader_mult = val;
            else if (name == "ucb_c") HP[u].ucb_c = val;

            else if (name == "eval_expand") HP[u].eval_expand = val;
            else if (name == "eval_level") HP[u].eval_level = val;
            else if (name == "eval_reach") HP[u].eval_reach = val;
            else if (name == "eval_attack") HP[u].eval_attack = val;
            else if (name == "eval_trap") HP[u].eval_trap = val;

            else if (name == "rollout_depth") HP[u].rollout_depth = (int)val;
            else if (name == "num_particles") HP[u].num_particles = (int)val;
            else if (name == "pf_noise_w") HP[u].pf_noise_w = val;
            else if (name == "pf_noise_eps") HP[u].pf_noise_eps = val;

            else if (name == "u_wb_boost") HP[u].u_wb_boost = val;
            else if (name == "u_wd_penalty") HP[u].u_wd_penalty = val;
            else if (name == "m_leader_scale") HP[u].m_leader_scale = val;
        }
    }
    fclose(fp);
}

// ===================== MAIN =====================
int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc > 1) {
        loadParams(argv[1]);
    }
    
    // Fixed seed support
    unsigned int seed = 12345; // Default fixed seed
    if (argc > 2) {
        seed = (unsigned int)stoul(argv[2]);
    } else {
        // Fallback to random seed if no 2nd argument provided (optional)
        // seed = chrono::steady_clock::now().time_since_epoch().count(); 
    }
    rng.seed(seed);

    initBitboards();

    cin >> N >> M >> T >> U;
    invT = 1.0 / T;

    adaptParams();

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int val; cin >> val;
            V[i*10 + j] = val;
        }

    State state;
    state.init();
    for (int p = 0; p < M; p++) {
        int x, y; cin >> x >> y;
        int idx = x*10 + y;
        state.p_pos[p] = idx;
        state.owner_mask[p] |= ((Bitboard)1 << idx);
        state.owner_map[idx] = p;
        state.level[idx] = 1;
        state.totalScore[p] += (double)V[idx] * 1; // ★改善5: 初期スコア設定
    }

    initAllParticles();

    for (int t = 1; t <= T; t++) {
        State preTurnState = state;

        auto [moveX, moveY] = selectMove(state, t);
        cout << moveX << " " << moveY << "\n";
        cout.flush();

        int tx[8], ty[8], ex[8], ey[8];
        for (int p = 0; p < M; p++) cin >> tx[p] >> ty[p];
        for (int p = 0; p < M; p++) cin >> ex[p] >> ey[p];

        // 盤面再読み込み: owner
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int owner; cin >> owner;
                int idx = i*10 + j;
                state.owner_map[idx] = owner;
            }

        // Rebuild masks
        for(int p=0; p<M; p++) state.owner_mask[p] = 0;
        for(int i=0; i<100; i++) {
            int o = state.owner_map[i];
            if (o != -1) state.owner_mask[o] |= ((Bitboard)1 << i);
        }

        // 盤面再読み込み: level
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int l; cin >> l;
                state.level[i*10 + j] = l;
            }

        for (int p = 0; p < M; p++) {
            state.p_pos[p] = ex[p]*10 + ey[p];
        }

        // ★改善5: totalScoreをジャッジ出力から再構築
        state.rebuildTotalScore();

        for (int p = 1; p < M; p++)
            updateParticles(preTurnState, p, tx[p], ty[p]);
    }

    return 0;
}