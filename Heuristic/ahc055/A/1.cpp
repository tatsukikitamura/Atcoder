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
#include <queue>
#include <functional>

using namespace std;

// Bitset optimization version of baseline (1.cpp)

namespace Config {
    const double TIME_LIMIT = 1.97;
    const double T_START = 150.0;
    const double T_END = 0.01;
    const int INITIAL_SEED_COUNT = 100;
    const int BEAM_WIDTH = 50;
    const bool USE_BEAM_SEARCH = true;
    double PROB_SWAP = 0.057;
    double PROB_INSERT_FRONT = 0.379;
    const int SORT_EVAL_WIDTH = 20;
    
    // Chokudai Search Config
    const double CHOKUDAI_TIME_LIMIT = 0.8; // Allocate significant time
    const int CHOKUDAI_MAX_WIDTH = 5000; // Max states per depth in PQ
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
long long simulate_full(const vector<int>& order, vector<int>& dur_out) {
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
        
        // Find efficient weapons using bitsets
        // Iterate groups descending by damage
        // Efficient Overkill Prevention Strategy
        while (rem > 0) {
            int chosend_w = -1;
            int chosen_d = -1;
            int strongest_w = -1;
            int strongest_d = -1;
            int ssd_w = -1;
            int ssd_d = -1;
            
            // Iterate groups descending by damage
            for (const auto& group : weapon_groups[box]) {
                Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
                if (candidates.none()) continue;
                
                int w = candidates.first_set();
                
                if (strongest_w == -1) {
                    strongest_w = w;
                    strongest_d = group.dmg;
                }
                
                if (group.dmg >= rem) {
                    ssd_w = w;
                    ssd_d = group.dmg;
                } else {
                    break;
                }
            }
            
            if (ssd_w != -1) { chosend_w = ssd_w; chosen_d = ssd_d; }
            else { chosend_w = strongest_w; chosen_d = strongest_d; }
            
            if (chosend_w == -1) break;
            
            int use = min((long long)dur[chosend_w], (rem + chosen_d - 1) / chosen_d);
            total += use;
            rem -= (long long)use * chosen_d;
            dur[chosend_w] -= use;
            
            if (dur[chosend_w] == 0) dur_mask.reset(chosend_w);
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
    shared_ptr<BeamNode> parent;
    long long score;
};

struct BeamNode {
    optional<pair<int, shared_ptr<BeamNode>>> parent;
    vector<pair<int, weak_ptr<BeamNode>>> children;
    long long score;
    
    BeamNode(optional<pair<int, shared_ptr<BeamNode>>> p, long long s)
        : parent(p), score(s) {}
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
    
    long long try_next(int next_box) const {
        long long new_score = score;
        long long rem = H[next_box];
        
        vector<int> tmp_dur = dur; // Copy needed? Only for decrements
        // Actually try_next shouldn't modify state. But calculation needs tracking.
        // For speed, BeamSearch 'try_next' can be approximate or just use bitsets but we need durability tracking.
        
        // Let's implement full logic using bitsets
        Bitset256 local_dur_mask = dur_mask; 
        
        // Note: used_mask acts as unlocked_mask here
        
        for (const auto& group : weapon_groups[next_box]) {
            if (rem <= 0) break;
            Bitset256 candidates = group.mask & used_mask & local_dur_mask;
            
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
    };
    
    Backup apply(int next_box) {
        Backup backup;
        backup.old_score = score;
        
        long long rem = H[next_box];
        
        for (const auto& group : weapon_groups[next_box]) {
            if (rem <= 0) break;
            Bitset256 candidates = group.mask & used_mask & dur_mask;
            
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


// --- Chokudai Search ---

// Using existing BeamNode and BeamState structs

void chokudai_search(vector<int>& best_order_out) {
    auto root = make_shared<BeamNode>(nullopt, 0);
    BeamState initial_state;
    initial_state.node = root;
    
    // Priority queues for each depth
    // We want to pop the state with minimum score.
    // However, BeamState is large. We should store shared_ptr<BeamState> or just BeamState?
    // BeamState is ~270 bytes. It's fine to store by value.
    
    struct StateWrapper {
        BeamState state;
        bool operator>(const StateWrapper& other) const {
            return state.score > other.state.score;
        }
    };
    
    vector<priority_queue<StateWrapper, vector<StateWrapper>, greater<StateWrapper>>> queues(N + 1);
    
    queues[0].push({initial_state});
    
    Timer timer;
    long long best_score_found = LLONG_MAX;
    
    while (timer.elapsed() < Config::CHOKUDAI_TIME_LIMIT) {
        bool updated = false;
        
        for (int d = 0; d < N; ++d) {
            if (queues[d].empty()) continue;
            
            // Pop the best state
            // In pure Chokudai Search, we might want to expand just one.
            // But if we have many "beams" (width), we can expand up to width?
            // "Chokudai Search" typically iterates d=0..N-1, expands 1 best from each depth that hasn't been expanded (or just best in PQ).
            
            // Limit checks
            // If we have too many states at d+1, we might skip?
            // But standard behavior is just PQ.
            
            StateWrapper wrapper = queues[d].top();
            queues[d].pop();
            updated = true;
            
            BeamState& state = wrapper.state;
            
            // Expand
            // Find all valid next boxes
            // Similar to collect_candidates logic but inline
            
            for (int next_box = 0; next_box < N; ++next_box) {
                if (state.used_mask.test(next_box)) continue;
                
                // Calculate next score
                // We need to apply the move to get new state
                // BeamState::apply modifies state. So we need a copy.
                
                BeamState next_state = state;
                auto backup = next_state.apply(next_box); 
                // Wait, apply() modifies 'score' and 'dur'. 
                // And adds to used_mask.
                
                // Link new node
                 auto child = make_shared<BeamNode>(
                    make_pair(next_box, state.node), next_state.score
                );
                state.node->children.emplace_back(next_box, weak_ptr<BeamNode>(child));
                next_state.node = child;
                
                if (next_state.depth == N) {
                   if (next_state.score < best_score_found) {
                       best_score_found = next_state.score;
                       // Reconstruct order
                       vector<int> order;
                       auto curr = child;
                        while (curr && curr->parent.has_value()) {
                            order.push_back(curr->parent->first);
                            curr = curr->parent->second;
                        }
                        reverse(order.begin(), order.end());
                        best_order_out = order;
                   }
                } else {
                    if (queues[d+1].size() < Config::CHOKUDAI_MAX_WIDTH) {
                        queues[d+1].push({next_state});
                    }
                }
            }
        }
        
        if (!updated) break; // All queues empty? Should not happen if time allows and we have paths.
    }
}

void output_result(const vector<int>& order) {
    // Just reprint from 1.cpp logic but using bitsets for consistency
    // Re-use logic for exact match
    vector<int> dur = C;
    Bitset256 unlocked_mask;
    Bitset256 dur_mask;
    for(int i=0; i<N; ++i) if(dur[i]>0) dur_mask.set(i);
    
    for (int idx = 0; idx < N; ++idx) {
        int box = order[idx];
        long long rem = H[box];
        
        while (rem > 0) {
            int best_w = -1;
            int best_d = 1;
            
            // Search bitsets
            for (const auto& group : weapon_groups[box]) {
                Bitset256 candidates = group.mask & unlocked_mask & dur_mask;
                if (!candidates.none()) {
                    best_w = candidates.first_set();
                    best_d = group.dmg;
                    break;
                }
            }
            
            // Fallback if no weapon found (damage 1)
            // But wait, if rem > 0 we always hit "best_w = -1" if candidates empty.
            if (best_w == -1) {
                // Should not happen if sorting is correct and we allow dmg=1
                // But sorted_weapons only has dmg > 1.
                // So best_w = -1 is correct (bare hands).
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

int main(int argc, char* argv[]) {
    if (argc >= 3) {
        Config::PROB_SWAP = stod(argv[1]);
        Config::PROB_INSERT_FRONT = stod(argv[2]);
    }
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
    long long score = simulate_full(order, dummy_dur);
    
    vector<int> best_order = order;
    long long best_score = score;
    
    if (Config::USE_BEAM_SEARCH) {
        vector<int> beam_order;
        chokudai_search(beam_order);
        if (!beam_order.empty()) {
             long long beam_score = simulate_full(beam_order, dummy_dur);
             if (beam_score < best_score) {
                 best_score = beam_score;
                 best_order = beam_order;
                 order = beam_order;
                 score = beam_score;
             }
        }
    }
    
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    for (int seed = 0; seed < Config::INITIAL_SEED_COUNT; ++seed) {
        vector<int> test_order(N);
        iota(test_order.begin(), test_order.end(), 0);
        shuffle(test_order.begin(), test_order.end(), rng);
        long long test_score = simulate_full(test_order, dummy_dur);
        if (test_score < best_score) {
            best_score = test_score;
            best_order = test_order;
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
                }
            } else {
                for (int k = j; k > i; --k) order[k] = order[k-1];
                order[i] = elem;
            }
        }
        
    }
    
    cerr << "Best: " << best_score << endl;
    output_result(best_order);
    
    return 0;
}
