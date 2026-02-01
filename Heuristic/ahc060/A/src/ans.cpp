#include <bits/stdc++.h>
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

// --- 永続シミュレーションバッファ ---
vector<char>                      sim_flavor;
vector<unordered_set<uint64_t>>   sim_inventory;
vector<int>                       sim_actions;
vector<BFSState>                  sim_states;
vector<int>                       sim_visit_count;   // ノード別訪問回数
vector<int>                       sim_path_starts;   // 各BFSパス先頭のaction index

void init_buffers() {
    sim_flavor.resize(N);
    sim_inventory.resize(K);
    sim_actions.reserve(T);
    sim_states.reserve(MAX_STATES + 1000);
    sim_visit_count.resize(N);
    sim_path_starts.reserve(5000);
}

// replay_count 個の旧アクションを再生後、BFS シミュレーションを実行
// 結果は sim_actions / sim_visit_count / sim_path_starts に格納
int simulate(const vector<int>& convert_at,
             const vector<int>* replay_actions = nullptr,
             int replay_count = 0,
             const vector<int>* old_path_starts = nullptr)
{
    fill(sim_flavor.begin(), sim_flavor.end(), 'W');
    for (int i = 0; i < K; i++) sim_inventory[i].clear();
    sim_actions.clear();
    fill(sim_visit_count.begin(), sim_visit_count.end(), 0);
    sim_path_starts.clear();

    int cur = 0;
    int prev_from = -1;
    uint64_t cone_hash = HASH_EMPTY;
    int steps = 0;

    // ---- Phase 1: リプレイ ----
    if (replay_actions && replay_count > 0) {
        if (old_path_starts) {
            for (int ps : *old_path_starts) {
                if (ps < replay_count) sim_path_starts.push_back(ps);
                else break;
            }
        }
        int rc = min(replay_count, (int)replay_actions->size());
        for (int i = 0; i < rc; i++) {
            int action = (*replay_actions)[i];
            sim_actions.push_back(action);
            steps++;
            if (action == -1) {
                sim_flavor[cur] = 'R';
            } else {
                prev_from = cur;
                cur = action;
                sim_visit_count[cur]++;
                if (cur < K) {
                    sim_inventory[cur].insert(cone_hash);
                    cone_hash = HASH_EMPTY;
                } else {
                    uint64_t ch = (sim_flavor[cur] == 'W') ? 1ULL : 2ULL;
                    cone_hash = cone_hash * HASH_BASE + ch;
                }
            }
        }
    }

    // ---- Phase 2: BFS シミュレーション ----
    auto get_path = [&](int idx, int shop) {
        vector<int> path;
        path.push_back(shop);
        for (int p = idx; p > 0; p = sim_states[p].parent_idx)
            path.push_back(sim_states[p].v);
        reverse(path.begin(), path.end());
        return path;
    };

    while (steps < T) {
        sim_path_starts.push_back((int)sim_actions.size());
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
            sim_actions.push_back(node);
            steps++;
            prev_from = cur;
            cur = node;
            sim_visit_count[cur]++;

            if (node < K) {
                sim_inventory[node].insert(cone_hash);
                cone_hash = HASH_EMPTY;
            } else {
                uint64_t ch = (sim_flavor[node] == 'W') ? 1ULL : 2ULL;
                cone_hash = cone_hash * HASH_BASE + ch;
                if (sim_flavor[node] == 'W' && steps < T
                        && steps >= convert_at[node]) {
                    sim_actions.push_back(-1);
                    steps++;
                    sim_flavor[node] = 'R';
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

    // ---- 初期解 ----
    vector<int> convert_at(N, T + 1);
    for (int i = K; i < N; i++)
        convert_at[i] = (int)(rng() % (T-1000))+500;

    // 現在解 (current) を保持
    int cur_score = simulate(convert_at);
    vector<int> cur_actions(sim_actions);
    vector<int> cur_visit_count(sim_visit_count);
    vector<int> cur_path_starts(sim_path_starts);

    // 最良解 (best) を保持
    int best_score = cur_score;
    vector<int> best_actions = cur_actions;

    // 訪問10回以上の木ノード一覧（current から作る）
    vector<int> frequent_nodes;
    auto rebuild_frequent = [&]() {
        frequent_nodes.clear();
        for (int i = K; i < N; i++)
            if (cur_visit_count[i] >= 10)
                frequent_nodes.push_back(i);
    };
    rebuild_frequent();

    // ---- 焼きなまし法 (Simulated Annealing) ----
    auto t_start = chrono::steady_clock::now();
    constexpr double TIME_LIMIT = 1.95;

    // 温度スケジュール（時間ベース）
    const double TEMP_START = 10.0;
    const double TEMP_END   = 0.05;

    auto progress01 = [&]() -> double {
        double t = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
        double p = t / TIME_LIMIT;
        if (p < 0) p = 0;
        if (p > 1) p = 1;
        return p;
    };
    auto temperature = [&]() -> double {
        // 幾何補間: T = T0 * (T1/T0)^p
        double p = progress01();
        return TEMP_START * pow(TEMP_END / TEMP_START, p);
    };

    int iter = 0;

    while (!frequent_nodes.empty()) {
        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
        if (elapsed >= TIME_LIMIT) break;

        // 近傍: 訪問10回以上のノードから1つ選び convert_at を変更
        int node = frequent_nodes[rng() % frequent_nodes.size()];
        int old_val = convert_at[node];
        int new_val = (int)(rng() % (T + 1));
        if (new_val == old_val) continue;

        int t = min(old_val, new_val);

        // 分岐点: ステップ >= t で初めて node を通るアクション（current 側を見る）
        int diverge_idx = -1;
        for (int j = max(0, t - 1), sz = (int)cur_actions.size(); j < sz; j++) {
            if (cur_actions[j] == node) { diverge_idx = j; break; }
        }
        if (diverge_idx == -1) continue;

        // 分岐点を含むパスの先頭までをリプレイ範囲とする（current 側を見る）
        auto it = upper_bound(cur_path_starts.begin(), cur_path_starts.end(), diverge_idx);
        int replay_count = (it != cur_path_starts.begin()) ? *--it : 0;

        // 評価
        convert_at[node] = new_val;
        int score = simulate(convert_at, &cur_actions, replay_count, &cur_path_starts);

        int delta = score - cur_score;
        bool accept = false;

        if (delta >= 0) {
            accept = true;
        } else {
            double temp = temperature();
            // temp が極小でも安全に
            if (temp > 1e-12) {
                double prob = exp((double)delta / temp);
                if (urd(rng) < prob) accept = true;
            }
        }

        if (accept) {
            // current を更新（simulate の結果が sim_* に入っているので swap）
            cur_score = score;
            cur_actions.swap(sim_actions);
            cur_visit_count.swap(sim_visit_count);
            cur_path_starts.swap(sim_path_starts);

            rebuild_frequent();

            if (cur_score > best_score) {
                best_score = cur_score;
                best_actions = cur_actions;
            }
        } else {
            // 近傍を破棄
            convert_at[node] = old_val;
        }

        iter++;
    }

    for (int a : best_actions)
        cout << a << '\n';

    cerr << "score=" << best_score << " iter=" << iter << endl;
    return 0;
}
