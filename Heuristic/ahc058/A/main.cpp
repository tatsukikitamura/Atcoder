#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>
#include <memory>
#include <optional>
#include <unordered_set>
#include <climits>
#include <cstring>

using namespace std;

// Bitset optimization version of baseline (1.cpp)

namespace Config {
    const double TIME_LIMIT = 1.97;
        const double T_START = 150.0;
    const double T_END = 0.01;
    const int INITIAL_SEED_COUNT = 100;
    const int BEAM_WIDTH = 50;
    const bool USE_BEAM_SEARCH = true;
    const double PROB_SWAP = 0.6;
    const double PROB_INSERT_FRONT = 0.25;
    const int SORT_EVAL_WIDTH = 20;
}

int N;
vector<int> H;
vector<int> C;
vector<vector<int>> A;
vector<vector<pair<int, int>>> sorted_weapons;

struct Timer {
    chrono::high_resolution_clock::time_point start;
    Timer() { start = chrono::high_resolution_clock::now(); }
    double elapsed() {
        return chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    }
};

// --- Bitset Implementation ---
struct Bitset256 {
    uint64_t w[4];
    
    Bitset256() { memset(w, 0, sizeof(w)); }
    
    void set(int i) {
        w[i >> 6] |= (1ULL << (i & 63));
    }
    
    void reset(int i) {
        w[i >> 6] &= ~(1ULL << (i & 63));
    }
    
    bool test(int i) const {
        return (w[i >> 6] >> (i & 63)) & 1;
    }
    
    // Returns index of first set bit, or -1 if none
    int first_set() const {
        if (w[0]) return __builtin_ctzll(w[0]);
        if (w[1]) return 64 + __builtin_ctzll(w[1]);
        if (w[2]) return 128 + __builtin_ctzll(w[2]);
        if (w[3]) return 192 + __builtin_ctzll(w[3]);
        return -1;
    }
    
    // Returns bitwise AND with another bitset
    Bitset256 operator&(const Bitset256& other) const {
        Bitset256 res;
        res.w[0] = w[0] & other.w[0];
        res.w[1] = w[1] & other.w[1];
        res.w[2] = w[2] & other.w[2];
        res.w[3] = w[3] & other.w[3];
        return res;
    }

    bool none() const {
        return (w[0] | w[1] | w[2] | w[3]) == 0;
    }
};

struct DamageGroup {
    int dmg;
    Bitset256 mask;
};

vector<vector<DamageGroup>> weapon_groups;

void precompute_weapons() {
    sorted_weapons.resize(N);
    weapon_groups.resize(N);
    
    for (int j = 0; j < N; ++j) {
        struct Entry { int dmg; int w; };
        vector<Entry> entries;
        for (int i = 0; i < N; ++i) {
            if (A[i][j] > 1) {
                entries.push_back({A[i][j], i});
            }
        }
        // Original sorted list
        sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return a.dmg > b.dmg;
        });
        for (const auto& e : entries) {
            sorted_weapons[j].emplace_back(e.dmg, e.w);
        }
        
        // Group by damage
        if (entries.empty()) continue;
        
        int current_dmg = entries[0].dmg;
        Bitset256 current_mask;
        current_mask.set(entries[0].w);
        
        for (size_t k = 1; k < entries.size(); ++k) {
            if (entries[k].dmg != current_dmg) {
                weapon_groups[j].push_back({current_dmg, current_mask});
                current_dmg = entries[k].dmg;
                current_mask = Bitset256(); // clear
            }
            current_mask.set(entries[k].w);
        }
        weapon_groups[j].push_back({current_dmg, current_mask});
    }
}

// Optimized simulate_full
long long simulate_full(const vector<int>& order, const vector<int>& strategies, vector<int>& dur_out) {
    long long total = 0;
    vector<int> dur = C;
    
    Bitset256 unlocked_mask;
    Bitset256 dur_mask;
    
    // Init dur_mask
    for (int i = 0; i < N; ++i) {
        if (dur[i] > 0) dur_mask.set(i);
    }
    
    for (int idx = 0; idx < N; ++idx) {
        int box = order[idx];
        long long rem = H[box];
        int strat = (idx < (int)strategies.size()) ? strategies[idx] : 0;
        
        // Find efficient weapons using bitsets
        // Iterate groups descending by damage
        bool skipped_first = false;
        for (const auto& group : weapon_groups[box]) {
            if (rem <= 0) break;
            
            Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
            
            // Strategy 1: Skip the highest damage group that has available weapons
            if (strat == 1 && !skipped_first && !candidates.none()) {
                skipped_first = true;
                continue;
            }

            // While there are candidates in this damage group
            while (!candidates.none() && rem > 0) {
                int w = candidates.first_set();
                
                int use = min((long long)dur[w], (rem + group.dmg - 1) / group.dmg);
                total += use;
                rem -= (long long)use * group.dmg;
                dur[w] -= use;
                
                if (dur[w] == 0) {
                    dur_mask.reset(w);
                    candidates.reset(w); // Removing from local candidates too
                } else {
                    // Used partially but still has durability
                    break; 
                }
            }
        }
        
        if (rem > 0) total += rem;
        unlocked_mask.set(box);
    }
    
    dur_out = dur;
    return total;
}

// Optimized simulate_from
long long simulate_from(int start_idx, const vector<int>& order, const vector<int>& input_dur) {
    vector<int> dur = input_dur;
    
    Bitset256 unlocked_mask;
    Bitset256 dur_mask;
    
    for (int i = 0; i < start_idx; ++i) unlocked_mask.set(order[i]);
    for (int i = 0; i < N; ++i) if (dur[i] > 0) dur_mask.set(i);
    
    long long total = 0;
    
    for (int idx = start_idx; idx < N; ++idx) {
        int box = order[idx];
        long long rem = H[box];
        
        for (const auto& group : weapon_groups[box]) {
            if (rem <= 0) break;
            
            Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
            
            while (!candidates.none() && rem > 0) {
                int w = candidates.first_set();
                
                int use = min((long long)dur[w], (rem + group.dmg - 1) / group.dmg);
                total += use;
                rem -= (long long)use * group.dmg;
                dur[w] -= use;
                
                if (dur[w] == 0) {
                    dur_mask.reset(w);
                    candidates.reset(w);
                } else {
                    break;
                }
            }
        }
        
        if (rem > 0) total += rem;
        unlocked_mask.set(box);
    }
    
    return total;
}


// --- Beam Search ---

struct BeamNode;

struct Candidate {
    int next_box;
    int strategy;
    shared_ptr<BeamNode> parent;
    long long score;
};

struct BeamNode {
    optional<pair<int, shared_ptr<BeamNode>>> parent;
    int strategy_used; // Strategy used to reach THIS node (i.e. for parent->first box)
    vector<pair<int, weak_ptr<BeamNode>>> children;
    long long score;
    
    BeamNode(optional<pair<int, shared_ptr<BeamNode>>> p, int strat, long long s)
        : parent(p), strategy_used(strat), score(s) {}
};

struct BeamState {
    vector<int> dur;
    Bitset256 used_mask; // Replaces 'used' vector
    Bitset256 dur_mask;
    long long score;
    shared_ptr<BeamNode> node;
    int depth;
    
    BeamState() : dur(C), score(0), depth(0) {
        for(int i=0; i<N; ++i) if(dur[i]>0) dur_mask.set(i);
    }
    
    long long try_next(int next_box, int strategy) const {
        long long new_score = score;
        long long rem = H[next_box];
        
        vector<int> tmp_dur = dur; 
        
        Bitset256 local_dur_mask = dur_mask; 
        
        bool skipped_first = false;
        
        for (const auto& group : weapon_groups[next_box]) {
            if (rem <= 0) break;
            Bitset256 candidates = group.mask & used_mask & local_dur_mask;
            
            if (strategy == 1 && !skipped_first && !candidates.none()) {
                skipped_first = true;
                continue;
            }
            
            while (!candidates.none() && rem > 0) {
                int w = candidates.first_set();
                int use = min((long long)tmp_dur[w], (rem + group.dmg - 1) / group.dmg);
                new_score += use;
                rem -= (long long)use * group.dmg;
                tmp_dur[w] -= use;
                if (tmp_dur[w] == 0) {
                    local_dur_mask.reset(w);
                    candidates.reset(w);
                } else {
                    break;
                }
            }
        }
        
        if (rem > 0) new_score += rem;
        return new_score;
    }

    struct Backup {
        vector<pair<int, int>> dur_changes;
        long long old_score;
        int strategy;
    };
    
    Backup apply(int next_box, int strategy) {
        Backup backup;
        backup.old_score = score;
        backup.strategy = strategy;
        
        long long rem = H[next_box];
        bool skipped_first = false;
        
        for (const auto& group : weapon_groups[next_box]) {
            if (rem <= 0) break;
            Bitset256 candidates = group.mask & used_mask & dur_mask;
            
            if (strategy == 1 && !skipped_first && !candidates.none()) {
                skipped_first = true;
                continue;
            }
            
            while (!candidates.none() && rem > 0) {
                int w = candidates.first_set();
                int old_d = dur[w];
                int use = min((long long)dur[w], (rem + group.dmg - 1) / group.dmg);
                score += use;
                rem -= (long long)use * group.dmg;
                dur[w] -= use;
                backup.dur_changes.emplace_back(w, old_d);
                
                if (dur[w] == 0) {
                    dur_mask.reset(w);
                    candidates.reset(w);
                } else {
                    break;
                }
            }
        }
        
        if (rem > 0) score += rem;
        used_mask.set(next_box);
        ++depth;
        return backup;
    }
    
    void restore(int next_box, const Backup& backup) {
        for (const auto& [idx, old_val] : backup.dur_changes) {
            if (dur[idx] == 0 && old_val > 0) dur_mask.set(idx);
            dur[idx] = old_val;
        }
        score = backup.old_score;
        used_mask.reset(next_box);
        --depth;
    }
};

void collect_candidates(
    BeamState& state,
    vector<Candidate>& candidates,
    int target_depth
) {
    if (state.depth == target_depth) {
        for (int next_box = 0; next_box < N; ++next_box) {
            if (state.used_mask.test(next_box)) continue;
            
            // Try strategies
            // 0: Greedy
            long long s0 = state.try_next(next_box, 0);
            candidates.push_back({next_box, 0, state.node, s0});
            
            // 1: Save Best (only if potentially useful?)
            // For now, try unconditionally or maybe restrict to cases where there IS a best weapon to skip
            long long s1 = state.try_next(next_box, 1);
            if (s1 != s0) { // Only add if different outcome (or maybe always?)
                 candidates.push_back({next_box, 1, state.node, s1});
            }
        }
        return;
    }
    
    auto& children = state.node->children;
    children.erase(
        remove_if(children.begin(), children.end(),
            [](const pair<int, weak_ptr<BeamNode>>& p) {
                return p.second.expired();
            }),
    children.end());
    
    auto current_node = state.node;
    
    for (const auto& [box, weak_child] : children) {
        auto child = weak_child.lock();
        if (!child) continue;
        
        state.node = child;
        auto backup = state.apply(box, child->strategy_used);
        
        collect_candidates(state, candidates, target_depth);
        
        state.restore(box, backup);
        state.node = current_node;
    }
}

pair<vector<int>, vector<int>> beam_search() {
    auto root = make_shared<BeamNode>(nullopt, 0, 0);
    
    BeamState state;
    state.node = root;
    
    vector<shared_ptr<BeamNode>> current_leaves{root};
    
    for (int step = 0; step < N; ++step) {
        vector<Candidate> candidates;
        // Branching factor increases (N-step) * 2 roughly
        candidates.reserve(current_leaves.size() * (N - step) * 2);
        collect_candidates(state, candidates, step);
        
        if (candidates.empty()) break;
        
        int keep = min(Config::BEAM_WIDTH, (int)candidates.size());
        
        if ((int)candidates.size() > keep) {
            nth_element(candidates.begin(), candidates.begin() + keep, candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                    return a.score < b.score;
                });
            candidates.resize(keep);
        }
        
        current_leaves.clear();
        
        for (const auto& [next_box, strategy, parent, score] : candidates) {
            auto child = make_shared<BeamNode>(
                make_pair(next_box, parent), strategy, score
            );
            parent->children.emplace_back(next_box, weak_ptr<BeamNode>(child));
            current_leaves.push_back(child);
        }
        
        if (current_leaves.empty()) break;
    }
    
    shared_ptr<BeamNode> best = nullptr;
    long long best_score = LLONG_MAX;
    
    for (const auto& leaf : current_leaves) {
        if (leaf->score < best_score) {
            best_score = leaf->score;
            best = leaf;
        }
    }
    
    vector<int> order;
    vector<int> strategies;
    if (best) {
        auto curr = best;
        while (curr && curr->parent.has_value()) {
            order.push_back(curr->parent->first);
            strategies.push_back(curr->strategy_used);
            curr = curr->parent->second;
        }
        reverse(order.begin(), order.end());
        reverse(strategies.begin(), strategies.end());
    }
    
    if ((int)order.size() < N) {
        vector<bool> used(N, false);
        for (int box : order) {
            used[box] = true;
        }
        for (int i = 0; i < N; ++i) {
            if (!used[i]) {
                order.push_back(i);
                strategies.push_back(0); // Default greedy
            }
        }
    }
    
    return {order, strategies};
}

void output_result(const vector<int>& order, const vector<int>& strategies) {
    // Just reprint from 1.cpp logic but using bitsets for consistency
    // Re-use logic for exact match
    vector<int> dur = C;
    Bitset256 unlocked_mask;
    Bitset256 dur_mask;
    for(int i=0; i<N; ++i) if(dur[i]>0) dur_mask.set(i);
    
    for (int idx = 0; idx < N; ++idx) {
        int box = order[idx];
        long long rem = H[box];
        int strat = (idx < (int)strategies.size()) ? strategies[idx] : 0;
        
        bool skipped_first = false;
        
        while (rem > 0) {
            int best_w = -1;
            int best_d = 1;
            
            // Search bitsets
            for (const auto& group : weapon_groups[box]) {
                Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
                
                if (strat == 1 && !skipped_first && !candidates.none()) {
                    skipped_first = true;
                    // Skip THIS group but continue to next groups
                    continue; 
                }
                
                if (!candidates.none()) {
                    best_w = candidates.first_set();
                    best_d = group.dmg;
                    break;
                }
            }
            
            if (best_w == -1) {
            }
            
            cout << best_w << " " << box << "\n";
            rem -= best_d;
            if (best_w >= 0) {
                dur[best_w]--;
                if (dur[best_w] == 0) dur_mask.reset(best_w);
            }
        }
        unlocked_mask.set(box);
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    Timer timer;
    
    cin >> N;
    
    H.resize(N);
    for (int i = 0; i < N; ++i) cin >> H[i];
    
    C.resize(N);
    for (int i = 0; i < N; ++i) cin >> C[i];
    
    A.assign(N, vector<int>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            cin >> A[i][j];
    
    precompute_weapons();
    
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    
    vector<double> val(N);
    for (int i = 0; i < N; ++i) {
        vector<int> top_dmg;
        for(const auto& grp : weapon_groups[i]) top_dmg.push_back(grp.dmg); // Approx? No, sorted_weapons has all
        // Re-use original A logic for sorting
        vector<int> raw_dmg = A[i];
        sort(raw_dmg.rbegin(), raw_dmg.rend());
        double s = 0;
        int limit = min(N, Config::SORT_EVAL_WIDTH);
        for (int j = 0; j < limit; ++j) s += raw_dmg[j];
        val[i] = s / limit * C[i];
    }
    sort(order.begin(), order.end(), [&](int a, int b) {
        return val[a] / H[a] > val[b] / H[b];
    });
    
    vector<int> dummy_dur;
    vector<int> dummy_strategies(N, 0); // Default all greedy
    long long score = simulate_full(order, dummy_strategies, dummy_dur);
    
    vector<int> best_order = order;
    vector<int> best_strategies = dummy_strategies;
    long long best_score = score;
    
    if (Config::USE_BEAM_SEARCH) {
        auto [beam_order, beam_strategies] = beam_search();
        long long beam_score = simulate_full(beam_order, beam_strategies, dummy_dur);
        if (beam_score < best_score) {
            best_score = beam_score;
            best_order = beam_order;
            best_strategies = beam_strategies;
            order = beam_order;
            score = beam_score;
        }
    }
    
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    for (int seed = 0; seed < Config::INITIAL_SEED_COUNT; ++seed) {
        vector<int> test_order(N);
        iota(test_order.begin(), test_order.end(), 0);
        shuffle(test_order.begin(), test_order.end(), rng);
        long long test_score = simulate_full(test_order, dummy_strategies, dummy_dur);
        if (test_score < best_score) {
            best_score = test_score;
            best_order = test_order;
            best_strategies = dummy_strategies;
            order = test_order;
            score = test_score;
        }
    }
    
    vector<vector<int>> hist_dur(N + 1);
    vector<long long> hist_score(N + 1, 0);
    
    auto update_history = [&]() {
        vector<int> dur = C;
        Bitset256 unlocked_mask;
        Bitset256 dur_mask;
        for(int i=0; i<N; ++i) if(dur[i]>0) dur_mask.set(i);
        long long total = 0;
        
        hist_dur[0] = C;
        hist_score[0] = 0;
        
        for (int idx = 0; idx < N; ++idx) {
            int box = order[idx];
            long long rem = H[box];
            
            for (const auto& group : weapon_groups[box]) {
                if (rem <= 0) break;
                Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
                while (!candidates.none() && rem > 0) {
                    int w = candidates.first_set();
                    int use = min((long long)dur[w], (rem + group.dmg - 1) / group.dmg);
                    total += use;
                    rem -= (long long)use * group.dmg;
                    dur[w] -= use;
                    if (dur[w] == 0) {
                        dur_mask.reset(w);
                        candidates.reset(w);
                    } else {
                        break;
                    }
                }
            }
            
            if (rem > 0) total += rem;
            unlocked_mask.set(box);
            
            hist_dur[idx + 1] = dur;
            hist_score[idx + 1] = total;
        }
    };
    
    update_history();
    
    uniform_real_distribution<double> dist01(0.0, 1.0);
    uniform_int_distribution<int> dist_idx(0, N - 1);
    
    // int iteration = 0; // Removing profiler iteration counter
    
    while (true) {
        double elapsed = timer.elapsed();
        if (elapsed > Config::TIME_LIMIT) break;
        
        double progress = elapsed / Config::TIME_LIMIT;
        double T = Config::T_START * pow(Config::T_END / Config::T_START, progress);
        
        double r = dist01(rng);
        
        if (r < Config::PROB_SWAP) {
            int i = dist_idx(rng);
            int j = dist_idx(rng);
            if (i == j) continue;
            if (i > j) swap(i, j);
            
            swap(order[i], order[j]);
            long long new_score = hist_score[i] + simulate_from(i, order, hist_dur[i]);
            long long delta = new_score - score;
            
            if (delta <= 0 || dist01(rng) < exp(-delta / T)) {
                score = new_score;
                
                {
                    // Full update from i
                    vector<int> dur = hist_dur[i];
                    Bitset256 unlocked_mask;
                    // Need unlocked_mask for [0...i-1]
                    for(int k=0; k<i; ++k) unlocked_mask.set(order[k]);
                    
                    Bitset256 dur_mask;
                    for(int k=0; k<N; ++k) if(dur[k]>0) dur_mask.set(k);
                    
                    long long total = hist_score[i];
                    
                    for (int idx = i; idx < N; ++idx) {
                        int box = order[idx];
                        long long rem = H[box];
                        
                        for (const auto& group : weapon_groups[box]) {
                            if (rem <= 0) break;
                            Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
                            while (!candidates.none() && rem > 0) {
                                int w = candidates.first_set();
                                int use = min((long long)dur[w], (rem + group.dmg - 1) / group.dmg);
                                total += use;
                                rem -= (long long)use * group.dmg;
                                dur[w] -= use;
                                if (dur[w] == 0) {
                                    dur_mask.reset(w);
                                    candidates.reset(w);
                                } else {
                                    break;
                                }
                            }
                        }
                        if (rem > 0) total += rem;
                        unlocked_mask.set(box);
                        hist_dur[idx + 1] = dur;
                        hist_score[idx + 1] = total;
                    }
                }
                
                if (score < best_score) {
                    best_score = score;
                    best_order = order;
                    best_strategies = vector<int>(N, 0); // SA assumes greedy
                }
            } else {
                swap(order[i], order[j]);
            }
        } 
        // Implement INSERT strategies similarly? Or just skip for brevity as SWAP is main?
        // Let's implement INSERT_FRONT
        else if (r < Config::PROB_SWAP + Config::PROB_INSERT_FRONT) {
            int i = dist_idx(rng);
            if (i == 0) continue;
            uniform_int_distribution<int> dist_j(0, i - 1);
            int j = dist_j(rng);
            
            int elem = order[i];
            for (int k = i; k > j; --k) order[k] = order[k-1];
            order[j] = elem;
            
            long long new_score = hist_score[j] + simulate_from(j, order, hist_dur[j]);
            long long delta = new_score - score;
            
             if (delta <= 0 || dist01(rng) < exp(-delta / T)) {
                score = new_score;
                // Reconstruct history from j
                {
                    vector<int> dur = hist_dur[j];
                    Bitset256 unlocked_mask;
                    for(int k=0; k<j; ++k) unlocked_mask.set(order[k]);
                    Bitset256 dur_mask;
                    for(int k=0; k<N; ++k) if(dur[k]>0) dur_mask.set(k);
                    long long total = hist_score[j];
                    
                    for (int idx = j; idx < N; ++idx) {
                        int box = order[idx];
                        long long rem = H[box];
                        for (const auto& group : weapon_groups[box]) {
                            if (rem <= 0) break;
                            Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
                            while (!candidates.none() && rem > 0) {
                                int w = candidates.first_set();
                                int use = min((long long)dur[w], (rem + group.dmg - 1) / group.dmg);
                                total += use;
                                rem -= (long long)use * group.dmg;
                                dur[w] -= use;
                                if (dur[w] == 0) {
                                    dur_mask.reset(w);
                                    candidates.reset(w);
                                } else {
                                    break;
                                }
                            }
                        }
                        if (rem > 0) total += rem;
                        unlocked_mask.set(box);
                        hist_dur[idx + 1] = dur;
                        hist_score[idx + 1] = total;
                    }
                }
                
                if (score < best_score) {
                    best_score = score;
                    best_order = order;
                    best_strategies = vector<int>(N, 0);
                }
            } else {
                for (int k = j; k < i; ++k) order[k] = order[k+1];
                order[i] = elem;
            }
        }
        else {
             // Backward insert
            int i = dist_idx(rng);
            if (i == N - 1) continue;
            uniform_int_distribution<int> dist_j(i + 1, N - 1);
            int j = dist_j(rng);
            
            int elem = order[i];
            for (int k = i; k < j; ++k) order[k] = order[k+1];
            order[j] = elem;
            
            long long new_score = hist_score[i] + simulate_from(i, order, hist_dur[i]);
            long long delta = new_score - score;
            
            if (delta <= 0 || dist01(rng) < exp(-delta / T)) {
                score = new_score;
                // Reconstruct from i
                {
                    vector<int> dur = hist_dur[i];
                    Bitset256 unlocked_mask;
                    for(int k=0; k<i; ++k) unlocked_mask.set(order[k]);
                    Bitset256 dur_mask;
                    for(int k=0; k<N; ++k) if(dur[k]>0) dur_mask.set(k);
                    long long total = hist_score[i];
                    
                     for (int idx = i; idx < N; ++idx) {
                        int box = order[idx];
                        long long rem = H[box];
                        for (const auto& group : weapon_groups[box]) {
                            if (rem <= 0) break;
                            Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
                            while (!candidates.none() && rem > 0) {
                                int w = candidates.first_set();
                                int use = min((long long)dur[w], (rem + group.dmg - 1) / group.dmg);
                                total += use;
                                rem -= (long long)use * group.dmg;
                                dur[w] -= use;
                                if (dur[w] == 0) {
                                    dur_mask.reset(w);
                                    candidates.reset(w);
                                } else {
                                    break;
                                }
                            }
                        }
                        if (rem > 0) total += rem;
                        unlocked_mask.set(box);
                        hist_dur[idx + 1] = dur;
                        hist_score[idx + 1] = total;
                    }
                }
                
                if (score < best_score) {
                    best_score = score;
                    best_order = order;
                    best_strategies = vector<int>(N, 0);
                }
            } else {
                for (int k = j; k < i; ++k) order[k] = order[k+1];
                order[i] = elem;
            }
        }
        
    }
    
    cerr << "Best: " << best_score << endl;
    output_result(best_order, best_strategies);
    
    return 0;
}
