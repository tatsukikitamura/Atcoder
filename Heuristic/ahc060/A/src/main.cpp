#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>
using namespace std;

constexpr int MAX_DEPTH = 20;
constexpr int MAX_STATES = 50000;
constexpr uint64_t HASH_BASE = 131ULL;
constexpr uint64_t HASH_EMPTY = 0ULL;

int N, M, K, T;
vector<vector<int>> adj;

struct BFSState {
    int v, pv;
    int parent_idx;
    int depth;
    uint64_t hash;
};

// --- シミュレーションバッファ ---
vector<char>                      sim_flavor;
vector<unordered_set<uint64_t>>   sim_inventory;
vector<int>                       sim_actions;
vector<BFSState>                  sim_states;

void init_buffers() {
    sim_flavor.resize(N);
    sim_inventory.resize(K);
    sim_actions.reserve(T);
    sim_states.reserve(MAX_STATES + 1000);
}

// flip_order: 各木ノードの変換優先順位（小さいほど早く変換）
// flip_quota: 変換可能な上限数
int simulate(const vector<int>& flip_order, int flip_quota, vector<int>& out_actions)
{
    fill(sim_flavor.begin(), sim_flavor.end(), 'W');
    for (int i = 0; i < K; i++) sim_inventory[i].clear();
    out_actions.clear();

    // flip_orderに基づいて、quotaまでの木ノードを特定
    vector<pair<int, int>> order_with_node;  // (order, node)
    for (int i = K; i < N; i++) {
        order_with_node.emplace_back(flip_order[i], i);
    }
    sort(order_with_node.begin(), order_with_node.end());
    
    unordered_set<int> should_flip;
    for (int i = 0; i < min(flip_quota, (int)order_with_node.size()); i++) {
        should_flip.insert(order_with_node[i].second);
    }

    int cur = 0;
    int prev_from = -1;
    uint64_t cone_hash = HASH_EMPTY;
    int steps = 0;
    int flipped_count = 0;

    auto get_path = [&](int idx, int shop) {
        vector<int> path;
        path.push_back(shop);
        for (int p = idx; p > 0; p = sim_states[p].parent_idx)
            path.push_back(sim_states[p].v);
        reverse(path.begin(), path.end());
        return path;
    };

    while (steps < T) {
        sim_states.clear();
        sim_states.push_back({cur, prev_from, -1, 0, cone_hash});

        vector<int> chosen_path;
        int fb_state = -1, fb_shop = -1;
        bool found = false;

        for (int idx = 0;
             idx < (int)sim_states.size() && !found
                 && (int)sim_states.size() < MAX_STATES;
             idx++)
        {
            const auto& st = sim_states[idx];
            if (st.depth >= MAX_DEPTH) continue;

            for (int u : adj[st.v]) {
                if (u == st.pv) continue;
                if (u < K) {
                    if (fb_state == -1) { fb_state = idx; fb_shop = u; }
                    if (sim_inventory[u].count(st.hash) == 0) {
                        chosen_path = get_path(idx, u);
                        found = true;
                        break;
                    }
                } else if ((int)sim_states.size() < MAX_STATES) {
                    uint64_t ch = (sim_flavor[u] == 'W') ? 1ULL : 2ULL;
                    sim_states.push_back(
                        {u, st.v, idx, st.depth + 1, st.hash * HASH_BASE + ch});
                }
            }
        }

        if (chosen_path.empty()) {
            if (fb_state == -1) break;
            chosen_path = get_path(fb_state, fb_shop);
        }

        for (int node : chosen_path) {
            if (steps >= T) break;
            out_actions.push_back(node);
            steps++;
            prev_from = cur;
            cur = node;

            if (node < K) {
                sim_inventory[node].insert(cone_hash);
                cone_hash = HASH_EMPTY;
            } else {
                uint64_t ch = (sim_flavor[node] == 'W') ? 1ULL : 2ULL;
                cone_hash = cone_hash * HASH_BASE + ch;
                
                // flip_orderに基づいてflipするか判定
                if (sim_flavor[node] == 'W' && steps < T
                        && should_flip.count(node) && flipped_count < flip_quota) {
                    out_actions.push_back(-1);
                    steps++;
                    sim_flavor[node] = 'R';
                    flipped_count++;
                }
            }
        }
    }

    int score = 0;
    for (int i = 0; i < K; i++) score += (int)sim_inventory[i].size();
    return score;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N >> M >> K >> T;
    adj.resize(N);
    for (int i = 0; i < M; i++) {
        int a, b; cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    for (int i = 0; i < N; i++) { int x, y; cin >> x >> y; }

    init_buffers();
    mt19937 rng(42);
    uniform_real_distribution<double> urd(0.0, 1.0);

    int max_flips = N - K;

    // flip_order: 各木ノードの変換優先順位（0 ~ max_flips-1をシャッフル）
    vector<int> flip_order(N, 0);
    vector<int> perm;
    for (int i = K; i < N; i++) perm.push_back(i);
    shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < (int)perm.size(); i++) {
        flip_order[perm[i]] = i;
    }

    // 初期解
    vector<int> cur_actions;
    int cur_score = simulate(flip_order, max_flips, cur_actions);

    int best_score = cur_score;
    vector<int> best_actions = cur_actions;
    vector<int> best_flip_order = flip_order;

    // 焼きなまし法
    auto t_start = chrono::steady_clock::now();
    constexpr double TIME_LIMIT = 1.95;

    const double TEMP_START = 8.0;
    const double TEMP_END   = 0.05;

    auto elapsed = [&]() -> double {
        return chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
    };
    auto progress01 = [&]() -> double {
        double p = elapsed() / TIME_LIMIT;
        return min(1.0, max(0.0, p));
    };
    auto temperature = [&]() -> double {
        double p = progress01();
        return TEMP_START * pow(TEMP_END / TEMP_START, p);
    };

    int iter = 0;

    while (elapsed() < TIME_LIMIT) {
        // 近傍: 2つの木ノードのflip_orderをswap
        int idx1 = K + rng() % (N - K);
        int idx2 = K + rng() % (N - K);
        if (idx1 == idx2) continue;

        swap(flip_order[idx1], flip_order[idx2]);

        vector<int> new_actions;
        int new_score = simulate(flip_order, max_flips, new_actions);

        int delta = new_score - cur_score;
        bool accept = false;

        if (delta >= 0) {
            accept = true;
        } else {
            double temp = temperature();
            if (temp > 1e-12) {
                double prob = exp((double)delta / temp);
                if (urd(rng) < prob) accept = true;
            }
        }

        if (accept) {
            cur_score = new_score;
            cur_actions = new_actions;

            if (cur_score > best_score) {
                best_score = cur_score;
                best_actions = cur_actions;
                best_flip_order = flip_order;
            }
        } else {
            swap(flip_order[idx1], flip_order[idx2]);
        }

        iter++;
    }

    for (int a : best_actions)
        cout << a << '\n';

    cerr << "score=" << best_score << " iter=" << iter << endl;
    return 0;
}
