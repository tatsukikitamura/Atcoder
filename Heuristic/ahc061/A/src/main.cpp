#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <map>
#include <fstream>
#include <string>
using namespace std;

// ===================== GLOBALS =====================
int N, M, T, U;
int V[100]; // 1D array for value map
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
auto programStart = chrono::steady_clock::now();

double elapsedMs() {
    return chrono::duration_cast<chrono::microseconds>(
        chrono::steady_clock::now() - programStart).count() / 1000.0;
}

// ===================== TUNABLE HYPERPARAMETERS =====================
struct HyperParams {
    double phase1 = 0.20818362;
    double phase2 = 0.68789216;

    double wa_early = 0.85139762, wb_early = 0.21523535, wc_early = 0.93231604, wd_early = 0.29557141;
    double wa_mid   = 0.30730916, wb_mid   = 0.90398914, wc_mid   = 0.51672106, wd_mid   = 0.33848466;
    double wa_late  = 0.10187044, wb_late  = 1.14537644, wc_late  = 0.95442664, wd_late  = 0.43595538;

    double leader_mult = 1.21876323;
    double ucb_c = 0.75114139;

    double eval_expand = 0.00000329;
    double eval_level  = 0.00000339;
    double eval_reach  = 0.00001442;
    double eval_attack = 0.00000371;

    int rollout_depth = 4;

    int num_particles = 373;
    double pf_noise_w = 0.01706855;
    double pf_noise_eps = 0.00119343;

    double u_wb_boost = 0.75843313;
    double u_wd_penalty = 0.48563285;

    double m_leader_scale = 0.24240965;
} HP;

void loadParams(const char* filename) {
    ifstream fin(filename);
    if (!fin.is_open()) return;
    string key; double val;
    map<string, double> params;
    while (fin >> key >> val) params[key] = val;
    fin.close();

    auto get = [&](const string& k, double def) -> double {
        auto it = params.find(k);
        return (it != params.end()) ? it->second : def;
    };

    HP.phase1       = get("phase1", HP.phase1);
    HP.phase2       = get("phase2", HP.phase2);
    HP.wa_early     = get("wa_early", HP.wa_early);
    HP.wb_early     = get("wb_early", HP.wb_early);
    HP.wc_early     = get("wc_early", HP.wc_early);
    HP.wd_early     = get("wd_early", HP.wd_early);
    HP.wa_mid       = get("wa_mid", HP.wa_mid);
    HP.wb_mid       = get("wb_mid", HP.wb_mid);
    HP.wc_mid       = get("wc_mid", HP.wc_mid);
    HP.wd_mid       = get("wd_mid", HP.wd_mid);
    HP.wa_late      = get("wa_late", HP.wa_late);
    HP.wb_late      = get("wb_late", HP.wb_late);
    HP.wc_late      = get("wc_late", HP.wc_late);
    HP.wd_late      = get("wd_late", HP.wd_late);
    HP.leader_mult  = get("leader_mult", HP.leader_mult);
    HP.ucb_c        = get("ucb_c", HP.ucb_c);
    HP.eval_expand  = get("eval_expand", HP.eval_expand);
    HP.eval_level   = get("eval_level", HP.eval_level);
    HP.eval_reach   = get("eval_reach", HP.eval_reach);
    HP.eval_attack  = get("eval_attack", HP.eval_attack);
    HP.rollout_depth = (int)get("rollout_depth", HP.rollout_depth);
    HP.num_particles = (int)get("num_particles", HP.num_particles);
    HP.pf_noise_w   = get("pf_noise_w", HP.pf_noise_w);
    HP.pf_noise_eps = get("pf_noise_eps", HP.pf_noise_eps);
    HP.u_wb_boost   = get("u_wb_boost", HP.u_wb_boost);
    HP.u_wd_penalty = get("u_wd_penalty", HP.u_wd_penalty);
    HP.m_leader_scale = get("m_leader_scale", HP.m_leader_scale);
}

void adaptParams() {
    double uFactor = max(0.0, (U - 2.0) / 3.0);
    HP.wb_mid  += HP.u_wb_boost * uFactor;
    HP.wb_late += HP.u_wb_boost * uFactor;
    HP.wd_mid  -= HP.u_wd_penalty * uFactor;
    HP.wd_late -= HP.u_wd_penalty * uFactor;
    HP.wd_mid  = max(0.01, HP.wd_mid);
    HP.wd_late = max(0.01, HP.wd_late);

    HP.leader_mult += HP.m_leader_scale * max(0, M - 2);

    if (M >= 6) {
        HP.rollout_depth = min(HP.rollout_depth, 10);
        HP.num_particles = min(HP.num_particles, 120);
    } else if (M <= 3) {
        HP.rollout_depth = min(HP.rollout_depth + 8, 30);
    }

    if (U == 1) {
        HP.wb_early = 0.0; HP.wb_mid = 0.0; HP.wb_late = 0.0;
        HP.phase1 = 0.5; HP.phase2 = 0.5;
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

    // デバッグ用: totalScoreの整合性検証
    double scoreNaive(int p) const {
        double s = 0;
        forEachBit(owner_mask[p], [&](int idx) {
            s += (double)V[idx] * level[idx];
        });
        return s;
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
    particles[p].resize(HP.num_particles);
    pweights[p].assign(HP.num_particles, 1.0 / HP.num_particles);
    for (int k = 0; k < HP.num_particles; k++)
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
    int NP = HP.num_particles;

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
    for (int k = 0; k < NP; k++) pweights[p][k] /= sumW;

    // Systematic resampling
    vector<Particle> newP(NP);
    vector<double> cumW(NP);
    cumW[0] = pweights[p][0];
    for (int k = 1; k < NP; k++) cumW[k] = cumW[k-1] + pweights[p][k];

    uniform_real_distribution<double> uni(0.0, 1.0 / NP);
    double u = uni(rng);
    int idx = 0;
    for (int k = 0; k < NP; k++) {
        double target = u + (double)k / NP;
        while (idx < NP - 1 && cumW[idx] < target) idx++;
        newP[k] = particles[p][idx];
    }

    normal_distribution<double> nw(0, HP.pf_noise_w), ne(0, HP.pf_noise_eps);
    for (int k = 0; k < NP; k++) {
        newP[k].wa  = clamp(newP[k].wa  + nw(rng), 0.3, 1.0);
        newP[k].wb  = clamp(newP[k].wb  + nw(rng), 0.3, 1.0);
        newP[k].wc  = clamp(newP[k].wc  + nw(rng), 0.3, 1.0);
        newP[k].wd  = clamp(newP[k].wd  + nw(rng), 0.3, 1.0);
        newP[k].eps = clamp(newP[k].eps + ne(rng), 0.1, 0.5);
    }

    particles[p] = newP;
    pweights[p].assign(NP, 1.0 / NP);

    estimated[p] = {0, 0, 0, 0, 0};
    for (int k = 0; k < NP; k++) {
        estimated[p].wa += newP[k].wa; estimated[p].wb += newP[k].wb;
        estimated[p].wc += newP[k].wc; estimated[p].wd += newP[k].wd;
        estimated[p].eps += newP[k].eps;
    }
    estimated[p].wa /= NP; estimated[p].wb /= NP;
    estimated[p].wc /= NP; estimated[p].wd /= NP;
    estimated[p].eps /= NP;
}

// ===================== AI MOVE GENERATION =====================
int genAIMove(const State& s, int p, const Particle& param, mt19937& lr) {
    MoveList ml;
    getMovable(s, p, ml);
    if (ml.cnt == 0) return s.p_pos[p];

    uniform_real_distribution<double> uni(0.0, 1.0);
    if (uni(lr) < param.eps) return ml.idx[lr() % ml.cnt];

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

    double remain = (double)(T - currentTurn) / T;
    return log2(1.0 + ratio)
           + HP.eval_expand * expandPot * remain
           + HP.eval_level * levelPot * remain
           + HP.eval_reach * myReach
           + HP.eval_attack * attackPot * remain;
}

// ===================== PLAYER 0 GREEDY (rollout policy) =====================
int greedyMove0(const State& s, int turn, mt19937& lr) {
    MoveList ml;
    getMovable(s, 0, ml);
    if (ml.cnt == 0) return s.p_pos[0];

    double phase = (double)turn / T;
    double wa0, wb0, wc0, wd0;
    if (phase < HP.phase1) {
        wa0 = HP.wa_early; wb0 = HP.wb_early; wc0 = HP.wc_early; wd0 = HP.wd_early;
    } else if (phase < HP.phase2) {
        wa0 = HP.wa_mid; wb0 = HP.wb_mid; wc0 = HP.wc_mid; wd0 = HP.wd_mid;
    } else {
        wa0 = HP.wa_late; wb0 = HP.wb_late; wc0 = HP.wc_late; wd0 = HP.wd_late;
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
            double mult = (o == leader) ? HP.leader_mult : 1.0;
            val = ((s.level[idx] == 1) ? V[idx] * wc0 : V[idx] * wd0) * mult;
        }
        if (val > bestVal + 1e-12) { bestVal = val; bestCnt = 0; bestIdx[bestCnt++] = k; }
        else if (val > bestVal - 1e-12) { bestIdx[bestCnt++] = k; }
    }
    int c = bestIdx[lr() % bestCnt];
    return ml.idx[c];
}

// ===================== ROLLOUT =====================
double rollout(State s, int startTurn, mt19937& lr) {
    int endTurn = min(T, startTurn + HP.rollout_depth - 1);
    int moves[8];
    for (int t = startTurn; t <= endTurn; t++) {
        moves[0] = greedyMove0(s, t, lr);
        for (int p = 1; p < M; p++) {
            moves[p] = genAIMove(s, p, estimated[p], lr);
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

    vector<double> scores(ml.cnt, 0.0);
    vector<int> visits(ml.cnt, 0);

    double elapsed = elapsedMs();

    int loops = 0;
    while (true) {
        loops++;
        if ((loops & 127) == 0) {
            double now = elapsedMs();
            if (now - elapsed > 30) break;
            if (now > 1950) break;
            double budgetMs = (1950 - now) / (T - currentTurn + 1);
            if (loops * 0.005 > budgetMs) break;
        }

        int bestIdx = -1;
        double bestUcb = -1e18;
        int total = loops - 1;

        for(int k=0; k<ml.cnt; k++) {
            if (visits[k] == 0) { bestIdx = k; break; }
            double avg = scores[k] / visits[k];
            double ucb = avg + HP.ucb_c * sqrt(log(total) / visits[k]);
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

// ===================== MAIN =====================
int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    initBitboards();

    if (argc >= 2) loadParams(argv[1]);

    cin >> N >> M >> T >> U;

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