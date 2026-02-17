#include <bits/stdc++.h>
using namespace std;

// ===================== GLOBALS =====================
int N, M, T, U;
int V[100]; // 1D array for value map
const int dx[] = {-1, 1, 0, 0}; // for legacy logic if needed
const int dy[] = {0, 0, -1, 1};

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
auto programStart = chrono::steady_clock::now();

double elapsedMs() {
    return chrono::duration_cast<chrono::microseconds>(
        chrono::steady_clock::now() - programStart).count() / 1000.0;
}

// ===================== TUNABLE HYPERPARAMETERS =====================
struct HyperParams {
    // Phase boundaries [0,1]
    double phase1 = 0.20818362;
    double phase2 = 0.68789216;

    // Greedy weights per phase (early / mid / late)
    double wa_early = 0.85139762, wb_early = 0.21523535, wc_early = 0.93231604, wd_early = 0.29557141;
    double wa_mid   = 0.30730916, wb_mid   = 0.90398914, wc_mid   = 0.51672106, wd_mid   = 0.33848466;
    double wa_late  = 0.10187044, wb_late  = 1.14537644, wc_late  = 0.95442664, wd_late  = 0.43595538;

    // Leader attack multiplier
    double leader_mult = 1.74430704;

    // UCB exploration constant
    double ucb_c = 1.10539331;

    // Evaluation function coefficients
    double eval_expand = 0.00000194;
    double eval_level  = 0.00000155;
    double eval_reach  = 0.00007388;

    // Rollout depth
    int rollout_depth = 5;

    // Particle filter
    int num_particles = 373;
    double pf_noise_w = 0.01706855;
    double pf_noise_eps = 0.00119343;

    // U-dependent adjustments
    double u_wb_boost = 0.92623109;    // added to wb in mid/late when U is high
    double u_wd_penalty = 0.42945212;  // subtracted from wd when U is high

    // M-dependent adjustments
    double m_leader_scale = 0.24240965; // leader_mult += m_leader_scale * (M-2)
} HP;

void loadParams(const char* filename) {
    ifstream fin(filename);
    if (!fin.is_open()) return; // use defaults if no config
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
    HP.rollout_depth = (int)get("rollout_depth", HP.rollout_depth);
    HP.num_particles = (int)get("num_particles", HP.num_particles);
    HP.pf_noise_w   = get("pf_noise_w", HP.pf_noise_w);
    HP.pf_noise_eps = get("pf_noise_eps", HP.pf_noise_eps);
    HP.u_wb_boost   = get("u_wb_boost", HP.u_wb_boost);
    HP.u_wd_penalty = get("u_wd_penalty", HP.u_wd_penalty);
    HP.m_leader_scale = get("m_leader_scale", HP.m_leader_scale);
}

// Adapt params based on actual M, U after reading input
void adaptParams() {
    // U-dependent: boost strengthen weight for high U
    double uFactor = max(0.0, (U - 2.0) / 3.0); // 0 for U<=2, 1 for U=5
    HP.wb_mid  += HP.u_wb_boost * uFactor;
    HP.wb_late += HP.u_wb_boost * uFactor;
    HP.wd_mid  -= HP.u_wd_penalty * uFactor;
    HP.wd_late -= HP.u_wd_penalty * uFactor;
    HP.wd_mid  = max(0.01, HP.wd_mid);
    HP.wd_late = max(0.01, HP.wd_late);

    // M-dependent: adjust leader mult and rollout depth
    HP.leader_mult += HP.m_leader_scale * max(0, M - 2);

    // Adjust rollout depth and particles for computation budget
    if (M >= 6) {
        HP.rollout_depth = min(HP.rollout_depth, 10);
        HP.num_particles = min(HP.num_particles, 120);
    } else if (M <= 3) {
        HP.rollout_depth = min(HP.rollout_depth + 8, 30);
    }

    // U=1: no strengthening possible
    if (U == 1) {
        HP.wb_early = 0.0; HP.wb_mid = 0.0; HP.wb_late = 0.0;
        HP.phase1 = 0.5; HP.phase2 = 0.5; // skip mid phase
    }
}

// ===================== BITBOARD UTILS =====================
using Bitboard = unsigned __int128;
Bitboard MASK_ALL = 0;
Bitboard MASK_COL_NOT_0 = 0; // Mask excluding column 0 (for left shift checks where <<1 touches next row if wrapped) - Wait, flattened row-major: i*10+j.
// Left neighbor of (i,j) is (i, j-1) -> idx-1. Valid if j>0.
// Right neighbor of (i,j) is (i, j+1) -> idx+1. Valid if j<9.
// Top neighbor of (i,j) is (i-1, j) -> idx-10. Valid if i>0.
// Bottom neighbor of (i,j) is (i+1, j) -> idx+10. Valid if i<9.

// Precomputed boundary masks to prevent wrapping
Bitboard MASK_NOT_COL_0 = 0;
Bitboard MASK_NOT_COL_9 = 0;

void initBitboards() {
    MASK_ALL = ((Bitboard)1 << 100) - 1;
    MASK_NOT_COL_0 = MASK_ALL;
    MASK_NOT_COL_9 = MASK_ALL;
    for (int i = 0; i < 10; i++) {
        // Col 0 indices: 0, 10, 20...
        MASK_NOT_COL_0 &= ~((Bitboard)1 << (i * 10));
        // Col 9 indices: 9, 19, 29...
        MASK_NOT_COL_9 &= ~((Bitboard)1 << (i * 10 + 9));
    }
}

inline int popcount128(Bitboard b) {
    uint64_t lo = (uint64_t)b;
    uint64_t hi = (uint64_t)(b >> 64);
    return __builtin_popcountll(lo) + __builtin_popcountll(hi);
}

// ===================== STATE =====================
struct State {
    Bitboard owner_mask[10]; // Bitboard for each player
    int8_t owner_map[100];   // Fast lookup for owner of cell i (-1 if none)
    uint8_t level[100];      // Level of each cell
    uint8_t p_pos[10];       // Current position of each player (0..99)

    void init() {
        for(int i=0; i<10; i++) owner_mask[i] = 0;
        memset(owner_map, -1, sizeof(owner_map));
        memset(level, 0, sizeof(level));
    }

    double score(int p) const {
        double s = 0;
        // Iterate only over owned cells
        // Using owner_mask[p] allows skipping non-owned cells
        Bitboard b = owner_mask[p];
        while (b) {
            // Find index of standard trailing zeros
            // __builtin_ctzll handles 64bit. Need 128 bit handling.
            uint64_t lo = (uint64_t)b;
            int idx;
            if (lo) {
                idx = __builtin_ctzll(lo);
                // Remove bit
                b &= ~((Bitboard)1 << idx);
            } else {
                uint64_t hi = (uint64_t)(b >> 64);
                idx = 64 + __builtin_ctzll(hi);
                // Remove bit: hi &= ~(1<<(idx-64)) -> b &= ...
                b &= ~((Bitboard)1 << idx);
            }
            s += (double)V[idx] * level[idx];
        }
        return s;
    }
};

// ===================== BFS / MOVABLE SET =====================
struct MoveList {
    uint8_t idx[200];
    int cnt;
};

// Returns a bitboard of reachable cells for player p (including own territory)
inline Bitboard getReachableMask(const State& s, int p) {
    Bitboard current = ((Bitboard)1 << s.p_pos[p]);
    Bitboard visited = current;
    Bitboard my_area = s.owner_mask[p];
    
    // Iterative bitwise dilation constrained to my_area
    while (true) {
        // Dilation: left, right, up, down
        Bitboard next = current;
        next |= ((current & MASK_NOT_COL_0) >> 1); // Left
        next |= ((current & MASK_NOT_COL_9) << 1); // Right
        next |= (current >> 10);                   // Up
        next |= (current << 10);                   // Down
        
        next &= my_area; // Constrain to owned area
        next &= ~visited; // Only new cells
        
        if (next == 0) break;
        
        visited |= next;
        current = next;
    }
    return visited;
}

void getMovable(const State& s, int p, MoveList& ml) {
    Bitboard reach = getReachableMask(s, p);
    
    // Dilation from reachable area to find adjacent moves
    Bitboard candidates = reach; // Can stay
    candidates |= ((reach & MASK_NOT_COL_0) >> 1);
    candidates |= ((reach & MASK_NOT_COL_9) << 1);
    candidates |= (reach >> 10);
    candidates |= (reach << 10);
    
    // Mask out all current player positions (cannot move to occupied cell)
    Bitboard occupied = 0;
    for(int i=0; i<M; i++) if(i != p) occupied |= ((Bitboard)1 << s.p_pos[i]);
    
    candidates &= ~occupied;
    candidates &= MASK_ALL; // Ensure within bounds (implicit but safe)

    // Extract indices
    ml.cnt = 0;
    while (candidates) {
        uint64_t lo = (uint64_t)candidates;
        int idx;
        if (lo) {
            idx = __builtin_ctzll(lo);
            candidates &= ~((Bitboard)1 << idx);
        } else {
            uint64_t hi = (uint64_t)(candidates >> 64);
            idx = 64 + __builtin_ctzll(hi);
            candidates &= ~((Bitboard)1 << idx);
        }
        ml.idx[ml.cnt++] = idx;
    }
}

// ===================== UNDO / DELTA UPDATE =====================
struct UndoData {
    int p_pos[10];
    struct CellChange {
        uint8_t idx;
        int8_t old_owner;
        uint8_t old_level;
    };
    CellChange changes[20]; 
    int changeCnt = 0;
    
    void addChange(int idx, int8_t owner, uint8_t level) {
        if (changeCnt < 20) {
            changes[changeCnt++] = { (uint8_t)idx, owner, level };
        }
    }
};

void undoTurn(State& s, const UndoData& u) {
    // Restore positions
    for(int p=0; p<M; p++) s.p_pos[p] = u.p_pos[p];
    
    // Reverse cell changes
    for(int i=u.changeCnt-1; i>=0; i--) {
        int idx = u.changes[i].idx;
        int8_t old_o = u.changes[i].old_owner;
        uint8_t old_l = u.changes[i].old_level;
        
        int8_t cur_o = s.owner_map[idx];
        
        // Update masks
        if (cur_o != -1) s.owner_mask[cur_o] &= ~((Bitboard)1 << idx);
        if (old_o != -1) s.owner_mask[old_o] |= ((Bitboard)1 << idx);
        
        s.owner_map[idx] = old_o;
        s.level[idx] = old_l;
    }
}

// ===================== TURN SIMULATION =====================
void simulateTurn(State& s, int moves[8], UndoData* undo = nullptr) { // moves is array of indices (0..99)
    int prevPos[8];
    if (undo) {
        undo->changeCnt = 0;
        for (int p = 0; p < M; p++) undo->p_pos[p] = s.p_pos[p];
    }
    for (int p = 0; p < M; p++) {
        prevPos[p] = s.p_pos[p];
        s.p_pos[p] = moves[p];
    }

    bool recalled[8] = {};
    int cellCnt[100] = {}; // Only needs to track up to 8 players, so safe
    
    for (int p = 0; p < M; p++) cellCnt[s.p_pos[p]]++;

    // Conflict resolution
    for (int i = 0; i < 100; i++) {
        if (cellCnt[i] < 2) continue;
        int co = s.owner_map[i];
        bool ownerHere = false;
        // Check if owner is here
        for (int p = 0; p < M; p++) {
            if (s.p_pos[p] == i && p == co) { ownerHere = true; break; }
        }
        
        for (int p = 0; p < M; p++) {
            if (s.p_pos[p] != i) continue;
            // If owner is here, everyone else is recalled.
            // If owner is NOT here, everyone is recalled.
            if (ownerHere && co >= 0) {
                if (p != co) recalled[p] = true;
            } else {
                recalled[p] = true;
            }
        }
    }

    for (int p = 0; p < M; p++) {
        if (recalled[p]) continue;
        int idx = s.p_pos[p];
        int owner = s.owner_map[idx];
        
        if (undo) undo->addChange(idx, owner, s.level[idx]); // Record BEFORE change
        
        if (owner == -1) {
            s.owner_mask[p] |= ((Bitboard)1 << idx);
            s.owner_map[idx] = p;
            s.level[idx] = 1;
        } else if (owner == p) {
            if (s.level[idx] < U) s.level[idx]++;
        } else {
            s.level[idx]--;
            if (s.level[idx] == 0) {
                // Remove from old owner
                s.owner_mask[owner] &= ~((Bitboard)1 << idx);
                // Add to new owner
                s.owner_mask[p] |= ((Bitboard)1 << idx);
                s.owner_map[idx] = p;
                s.level[idx] = 1;
            } else {
                // Failed attack
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

vector<vector<Particle>> particles; // [player][particle_idx]
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

// aiEvalCell updated for flat index
double aiEvalCell(const State& s, int p, int idx,
                  double wa, double wb, double wc, double wd) {
    int o = s.owner_map[idx];
    if (o == -1) return V[idx] * wa;
    if (o == p) return (s.level[idx] < U) ? V[idx] * wb : 0.0;
    return (s.level[idx] == 1) ? V[idx] * wc : V[idx] * wd;
}

double computeLikelihood(const State& pre, int p, int obsIdx,
                         const Particle& pt) {
    MoveList ml;
    getMovable(pre, p, ml);
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

void updateParticles(const State& pre, int p, int obsX, int obsY) {
    int obsIdx = obsX * 10 + obsY;
    int NP = HP.num_particles;
    double sumW = 0;
    for (int k = 0; k < NP; k++) {
        double lik = computeLikelihood(pre, p, obsIdx, particles[p][k]);
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
    double s0 = s.score(0);
    double sa = 0;
    for (int p = 1; p < M; p++) sa = max(sa, s.score(p));
    if (sa < 1e-9) return 20.0;
    double ratio = s0 / sa;

    Bitboard reachMask = getReachableMask(s, 0);
    int myReach = popcount128(reachMask);
    
    double expandPot = 0, levelPot = 0;
    // Iterate reachable cells
    Bitboard b = reachMask;
    while(b) {
        // Safe extraction
        uint64_t lo = (uint64_t)b;
        int i;
        if(lo) { i = __builtin_ctzll(lo); b &= ~((Bitboard)1<<i); }
        else   { uint64_t hi = (uint64_t)(b>>64); i = 64 + __builtin_ctzll(hi); b &= ~((Bitboard)1<<i); }
        
        // i is index
        if (s.owner_map[i] == 0 && s.level[i] < U)
            levelPot += V[i] * (U - s.level[i]);
            
        // Check neighbors for expansion (if neighbor is -1)
        // Neighbors: left, right, up, down
        // Left (i-1) if i%10 != 0
        if (i % 10 != 0) {
            int ni = i - 1;
            if (s.owner_map[ni] == -1) expandPot += V[ni];
        }
        // Right (i+1) if i%10 != 9
        if (i % 10 != 9) {
            int ni = i + 1;
            if (s.owner_map[ni] == -1) expandPot += V[ni];
        }
        // Up (i-10) if i>=10
        if (i >= 10) {
            int ni = i - 10;
            if (s.owner_map[ni] == -1) expandPot += V[ni];
        }
        // Down (i+10) if i<90
        if (i < 90) {
            int ni = i + 10;
            if (s.owner_map[ni] == -1) expandPot += V[ni];
        }
    }

    double remain = (double)(T - currentTurn) / T;
    return log2(1.0 + ratio)
           + HP.eval_expand * expandPot * remain
           + HP.eval_level * levelPot * remain
           + HP.eval_reach * myReach;
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
struct Node {
    State state;
    int turn;
    vector<int> legalMoves;
    vector<pair<int, double>> children; // {move_idx, total_score}
    vector<int> visits;
    int totalVisits = 0;

    Node(const State& s, int t) : state(s), turn(t) {
        MoveList ml;
        getMovable(s, 0, ml);
        legalMoves.resize(ml.cnt);
        for(int k=0; k<ml.cnt; k++) legalMoves[k] = ml.idx[k];
        
        children.resize(ml.cnt);
        visits.resize(ml.cnt, 0);
    }

    int select(mt19937& lr) {
        int best = -1;
        double bestScore = -1e18;
        for (size_t i = 0; i < legalMoves.size(); i++) {
            if (visits[i] == 0) return i; // Expansion
            double avg = children[i].second / visits[i];
            double uct = avg + HP.ucb_c * sqrt(log(totalVisits) / visits[i]);
            if (uct > bestScore) { bestScore = uct; best = i; }
        }
        return best;
    }
};

pair<int,int> selectMove(State& rootState, int currentTurn) {
    UndoData undo;
    MoveList ml;
    getMovable(rootState, 0, ml);
    if (ml.cnt == 0) return {rootState.p_pos[0]/10, rootState.p_pos[0]%10};

    // Dynamic time management
    double remTime = 1900.0 - elapsedMs();
    int remTurns = T - currentTurn + 1;
    double timeLimit = max(10.0, remTime / remTurns); 
    double startT = elapsedMs();

    // Accumulators
    vector<double> sumScore(ml.cnt, 0.0);
    vector<int> counts(ml.cnt, 0);

    int iter = 0;
    while (true) {
        // Check time every 10 iterations
        if ((iter & 15) == 0) {
            if (elapsedMs() - startT > timeLimit) break;
        }

        int k = iter % ml.cnt; 
        
        int moves[8];
        moves[0] = ml.idx[k];
        for(int p=1; p<M; p++) moves[p] = genAIMove(rootState, p, estimated[p], rng);
        
        simulateTurn(rootState, moves, &undo);
        double sc = evaluate(rootState, currentTurn + 1);
        undoTurn(rootState, undo);
        
        sumScore[k] += sc;
        counts[k]++;
        iter++;
    }
    
    // Find best average
    int bestIdx = -1;
    double bestScore = -1e18;
    for (int k = 0; k < ml.cnt; k++) {
        if (counts[k] == 0) continue;
        double avg = sumScore[k] / counts[k];
        if (avg > bestScore) {
            bestScore = avg;
            bestIdx = ml.idx[k];
        }
    }
    
    if (bestIdx == -1) bestIdx = ml.idx[0];
    return {bestIdx/10, bestIdx%10};
}

// ===================== MAIN =====================
int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    initBitboards();

    // Load hyperparameters from config file (optional)
    if (argc >= 2) loadParams(argv[1]);

    cin >> N >> M >> T >> U;

    // Adapt hyperparams based on actual M, U
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
        
        // Update state from input to be perfectly in sync with judge
        // Actually, we should trust our simulation OR judge.
        // AHC usually provides full state or delta.
        // AHC061 provides full board state after moves.
        
        // State input format:
        // N lines of owner
        // N lines of level
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int owner; cin >> owner;
                int idx = i*10 + j;
                state.owner_map[idx] = owner;
                // Rebuild masks is slow? No, 100 cells.
                // We should clear masks and rebuild to be safe
            }
        
        // Rebuild masks
        for(int p=0; p<M; p++) state.owner_mask[p] = 0;
        for(int i=0; i<100; i++) {
            int o = state.owner_map[i];
            if (o != -1) state.owner_mask[o] |= ((Bitboard)1 << i);
        }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int l; cin >> l;
                state.level[i*10 + j] = l;
            }
            
        for (int p = 0; p < M; p++) { 
            state.p_pos[p] = ex[p]*10 + ey[p]; 
        }

        for (int p = 1; p < M; p++)
            updateParticles(preTurnState, p, tx[p], ty[p]);
    }

    return 0;
}
