#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <deque>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <chrono> // 時間管理
#include <random> // ランダムな "キック" のため

using namespace std;
using namespace std::chrono;

// --- 1. グローバル変数と設定 ---
int N, K, T;
vector<string> V_WALLS;
vector<string> H_WALLS;
vector<pair<int, int>> TARGETS;

double TIME_LIMIT_SEC = 1.8; // 実行時間制限 (秒)
auto start_time = high_resolution_clock::now();

// ビームサーチ設定
int BEAM_WIDTH = 4;
int N_PATHS_PER_STATE = 3;

// ILS (フェーズ2) 設定
int N_KICK = 3; // 1回の反復でキック(最短経路に戻す)する経路の数

// 型定義
using Pos = pair<int, int>; 
using Edge = pair<Pos, Pos>; 
using DijkstraResult = tuple<int, int, vector<Pos>>;

// --- 2. ヘルパー関数 (省略なし) ---
bool can_move(int i, int j, char d) {
    if (d == 'U') {
        if (i == 0) return false;
        return H_WALLS[i - 1][j] == '0';
    }
    if (d == 'D') {
        if (i == N - 1) return false;
        return H_WALLS[i][j] == '0';
    }
    if (d == 'L') {
        if (j == 0) return false;
        return V_WALLS[i][j - 1] == '0';
    }
    if (d == 'R') {
        if (j == N - 1) return false;
        return V_WALLS[i][j] == '0';
    }
    return false;
}

Pos get_next_pos(int i, int j, char d) {
    if (d == 'U') return {i - 1, j};
    if (d == 'D') return {i + 1, j};
    if (d == 'L') return {i, j - 1};
    if (d == 'R') return {i, j + 1};
    return {i, j}; // 'S' or other
}

char get_direction(Pos pos1, Pos pos2) {
    if (pos2.first < pos1.first) return 'U';
    if (pos2.first > pos1.first) return 'D';
    if (pos2.second < pos1.second) return 'L';
    if (pos2.second > pos1.second) return 'R';
    return 'S';
}

// --- 3. BFS (省略なし) ---
pair<int, vector<Pos>> bfs_path_and_length(Pos start_pos, Pos goal_pos) {
    deque<pair<Pos, vector<Pos>>> q;
    q.push_back({start_pos, {start_pos}});
    set<Pos> visited = {start_pos};
    while (!q.empty()) {
        auto [pos, path] = q.front(); q.pop_front();
        if (pos == goal_pos) return {(int)path.size() - 1, path};
        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                if (visited.find(next_pos) == visited.end()) {
                    visited.insert(next_pos);
                    vector<Pos> new_path = path; new_path.push_back(next_pos);
                    q.push_back({next_pos, new_path});
                }
            }
        }
    }
    return {1e9, {}};
}

int bfs_second_path_length(Pos start_pos, Pos goal_pos, set<Edge>& forbidden_edges) {
    deque<pair<Pos, int>> q;
    q.push_back({start_pos, 0});
    set<Pos> visited = {start_pos};
    while (!q.empty()) {
        auto [pos, steps] = q.front(); q.pop_front();
        if (pos == goal_pos) return steps;
        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                Edge edge = {pos, next_pos};
                if (forbidden_edges.count(edge)) continue;
                if (visited.find(next_pos) == visited.end()) {
                    visited.insert(next_pos);
                    q.push_back({next_pos, steps + 1});
                }
            }
        }
    }
    return 1e9;
}

// --- 3.5. 経路復元ヘルパー (省略なし) ---
vector<Pos> reconstruct_path(map<Pos, map<int, pair<Pos, int>>>& prev, Pos start_pos, Pos goal_pos, int steps) {
    vector<Pos> path;
    Pos curr_pos = goal_pos;
    int curr_steps = steps;
    while (true) {
        path.push_back(curr_pos);
        if (curr_pos == start_pos && curr_steps == 0) break;
        if (prev.find(curr_pos) == prev.end() || prev[curr_pos].find(curr_steps) == prev[curr_pos].end()) {
             return {}; // 経路復元失敗
        }
        auto p = prev[curr_pos][curr_steps];
        curr_pos = p.first;
        curr_steps = p.second;
    }
    reverse(path.begin(), path.end());
    return path;
}

// --- 3.6. パレート最適ダイクストラ (ローカルサーチ用 - 単一解) ---
// (省略なし)
vector<Pos> find_path_dijkstra_single(Pos start_pos, Pos goal_pos, int step_limit, const set<Pos>& total_path_cells) {
    using State = tuple<int, int, Pos>;
    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({0, 0, start_pos});
    map<Pos, map<int, int>> dist;
    dist[start_pos][0] = 0;
    map<Pos, map<int, pair<Pos, int>>> prev;
    vector<pair<int, int>> found_goals; 

    while (!pq.empty()) {
        auto [cost, steps, pos] = pq.top(); pq.pop();
        if (dist.count(pos) && dist[pos].count(steps) && dist[pos][steps] < cost) continue;
        if (pos == goal_pos) {
            found_goals.push_back({cost, steps});
            continue;
        }
        if (steps + 1 > step_limit) continue;

        auto current_time = high_resolution_clock::now();
        if (duration_cast<duration<double>>(current_time - start_time).count() > TIME_LIMIT_SEC) {
            break; 
        }

        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                int next_steps = steps + 1;
                int new_cost = cost;
                if (total_path_cells.find(next_pos) == total_path_cells.end()) {
                    new_cost += 1;
                }

                bool is_dominated = false;
                if (dist.count(next_pos)) {
                    for (auto const& [existing_steps, existing_cost] : dist[next_pos]) {
                        if (existing_steps <= next_steps && existing_cost <= new_cost) {
                            is_dominated = true; break;
                        }
                    }
                }
                if (is_dominated) continue;

                if (dist.count(next_pos)) {
                    vector<int> dominated_steps;
                    for (auto const& [existing_steps, existing_cost] : dist[next_pos]) {
                        if (existing_steps >= next_steps && existing_cost >= new_cost) {
                            dominated_steps.push_back(existing_steps);
                        }
                    }
                    for (int s : dominated_steps) {
                        dist[next_pos].erase(s);
                        prev[next_pos].erase(s);
                    }
                }
                dist[next_pos][next_steps] = new_cost;
                prev[next_pos][next_steps] = {pos, steps};
                pq.push({new_cost, next_steps, next_pos});
            }
        }
    }
    if (found_goals.empty()) return {}; // None
    sort(found_goals.begin(), found_goals.end());
    return reconstruct_path(prev, start_pos, goal_pos, found_goals[0].second);
}


// --- 3.7. パレート最適ダイクストラ (ビームサーチ用 - 複数解) ---
// (省略なし)
vector<DijkstraResult> find_path_dijkstra_beam(Pos start_pos, Pos goal_pos, int step_limit, const set<Pos>& total_path_cells) {
    using State = tuple<int, int, Pos>;
    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({0, 0, start_pos});
    map<Pos, map<int, int>> dist;
    dist[start_pos][0] = 0;
    map<Pos, map<int, pair<Pos, int>>> prev;
    vector<pair<int, int>> found_goals;

    while (!pq.empty()) {
        auto [cost, steps, pos] = pq.top(); pq.pop();
        if (dist.count(pos) && dist[pos].count(steps) && dist[pos][steps] < cost) continue;
        if (pos == goal_pos) {
            found_goals.push_back({cost, steps});
            continue;
        }
        if (steps + 1 > step_limit) continue;

        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                int next_steps = steps + 1;
                int new_cost = cost;
                if (total_path_cells.find(next_pos) == total_path_cells.end()) {
                    new_cost += 1;
                }

                bool is_dominated = false;
                if (dist.count(next_pos)) {
                    for (auto const& [existing_steps, existing_cost] : dist[next_pos]) {
                        if (existing_steps <= next_steps && existing_cost <= new_cost) {
                            is_dominated = true; break;
                        }
                    }
                }
                if (is_dominated) continue;

                if (dist.count(next_pos)) {
                    vector<int> dominated_steps;
                    for (auto const& [existing_steps, existing_cost] : dist[next_pos]) {
                        if (existing_steps >= next_steps && existing_cost >= new_cost) {
                            dominated_steps.push_back(existing_steps);
                        }
                    }
                    for (int s : dominated_steps) {
                        dist[next_pos].erase(s);
                        prev[next_pos].erase(s);
                    }
                }
                dist[next_pos][next_steps] = new_cost;
                prev[next_pos][next_steps] = {pos, steps};
                pq.push({new_cost, next_steps, next_pos});
            }
        }
    }
    if (found_goals.empty()) return {};
    sort(found_goals.begin(), found_goals.end());
    if (found_goals.size() > (size_t)N_PATHS_PER_STATE) {
        found_goals.resize(N_PATHS_PER_STATE);
    }
    vector<DijkstraResult> results;
    for (auto const& [cost, steps] : found_goals) {
        vector<Pos> path = reconstruct_path(prev, start_pos, goal_pos, steps);
        if (!path.empty()) { // 経路復元成功
            results.emplace_back(cost, steps, path);
        }
    }
    return results;
}


// --- 4. メインロジック (ハイブリッド) ---

// 4.0. ビームの状態 (解)
struct BeamState {
    vector<vector<Pos>> paths_list;
    set<Pos> path_cells;
    int total_steps;
    int total_C;
    bool operator<(const BeamState& other) const {
        if (total_C != other.total_C) return total_C < other.total_C;
        return total_steps < other.total_steps;
    }
};

// 4.1. 自由度計算用
struct Segment {
    int k; int X_k; vector<Pos> path_k; double freedom;
    bool operator<(const Segment& other) const {
        return freedom < other.freedom;
    }
};

// ヘルパー: BeamState の C と steps を再計算
void recalculate_state(BeamState& state) {
    state.path_cells.clear();
    state.path_cells.insert(TARGETS[0]);
    state.total_steps = 0;
    for (const auto& path : state.paths_list) {
        if (!path.empty()) {
            state.path_cells.insert(path.begin(), path.end());
            state.total_steps += (path.size() - 1);
        }
    }
    state.total_C = state.path_cells.size() + 1; // +1 for dummy color 0
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> N >> K >> T;
    V_WALLS.resize(N); H_WALLS.resize(N - 1); TARGETS.resize(K);
    for (int i = 0; i < N; ++i) cin >> V_WALLS[i];
    for (int i = 0; i < N - 1; ++i) cin >> H_WALLS[i];
    for (int i = 0; i < K; ++i) cin >> TARGETS[i].first >> TARGETS[i].second;

    // 4.1. 全区間の自由度を計算
    vector<Segment> segments_info;
    double total_shortest_steps_X = 0;
    vector<int> shortest_lengths(K - 1);
    vector<vector<Pos>> all_shortest_paths_fallback(K - 1);

    for (int k = 0; k < K - 1; ++k) {
        Pos start_node = TARGETS[k], goal_node = TARGETS[k + 1];
        auto [X_k, path_k] = bfs_path_and_length(start_node, goal_node);
        all_shortest_paths_fallback[k] = path_k;
        shortest_lengths[k] = X_k;
        double freedom_score = 1e18;
        if (X_k < 1e9 && !path_k.empty()) {
            total_shortest_steps_X += X_k;
            set<Edge> forbidden_edges;
            for (size_t i = 0; i < path_k.size() - 1; ++i) forbidden_edges.insert({path_k[i], path_k[i+1]});
            int X_prime_k = bfs_second_path_length(start_node, goal_node, forbidden_edges);
            freedom_score = (double)X_prime_k - X_k;
        }
        segments_info.push_back({k, X_k, path_k, freedom_score});
    }

    // 4.2. 自由度が「低い」順にソート
    sort(segments_info.begin(), segments_info.end());

    // --- 4.3. フェーズ1: ビームサーチ ---
    vector<BeamState> current_beam;
    set<Pos> initial_cells = {TARGETS[0]};
    int initial_C = initial_cells.size() + 1; 
    current_beam.push_back({vector<vector<Pos>>(K - 1), initial_cells, 0, initial_C});
    double X_future = total_shortest_steps_X;

    for (const auto& segment : segments_info) {
        int k = segment.k; int X_k = segment.X_k;
        if (X_k >= 1e9) continue;
        Pos start_node = TARGETS[k], goal_node = TARGETS[k + 1];
        vector<BeamState> next_beam;

        for (const auto& state : current_beam) {
            double steps_allowed_total = T - state.total_steps;
            double margin_for_future = X_future - X_k;
            int step_limit = (int)(steps_allowed_total - margin_for_future);
            if (step_limit < X_k) step_limit = X_k;
                
            vector<DijkstraResult> candidate_paths_info = find_path_dijkstra_beam(start_node, goal_node, step_limit, state.path_cells);
            
            if (candidate_paths_info.empty()) {
                int cost = 0;
                for (const auto& cell : segment.path_k) {
                    if (state.path_cells.find(cell) == state.path_cells.end()) cost += 1;
                }
                candidate_paths_info.push_back({cost, X_k, segment.path_k});
            }
                
            for (const auto& [cost, steps, path] : candidate_paths_info) {
                if (path.empty()) continue; 
                vector<vector<Pos>> new_paths_list = state.paths_list;
                new_paths_list[k] = path;
                set<Pos> new_path_cells = state.path_cells;
                new_path_cells.insert(path.begin(), path.end());
                int new_total_steps = state.total_steps + steps;
                int new_total_C = new_path_cells.size() + 1; 
                next_beam.push_back({new_paths_list, new_path_cells, new_total_steps, new_total_C});
            }
        }
        sort(next_beam.begin(), next_beam.end());
        if (next_beam.size() > (size_t)BEAM_WIDTH) {
            next_beam.resize(BEAM_WIDTH);
        }
        current_beam = next_beam;
        X_future -= X_k;
    }

    // --- 4.4. フェーズ1完了 ---
    BeamState global_best_solution = current_beam[0];
    
    // --- 4.5. フェーズ2: 反復ローカルサーチ (ILS) ---
    mt19937 rng(0); 
    uniform_int_distribution<int> k_dist(0, K - 2);

    auto check_time = [&]() {
        auto current_time = high_resolution_clock::now();
        return duration_cast<duration<double>>(current_time - start_time).count() < TIME_LIMIT_SEC;
    };

    while (check_time()) { // アウターループ
        BeamState current_solution = global_best_solution;

        // [摂動 (Perturb / "キック")]
        set<int> kicked_k;
        for (int i = 0; i < N_KICK && k_dist(rng) < K - 1; ++i) {
             int k_to_kick = k_dist(rng);
             if(shortest_lengths[k_to_kick] < 1e9 && !all_shortest_paths_fallback[k_to_kick].empty()) {
                current_solution.paths_list[k_to_kick] = all_shortest_paths_fallback[k_to_kick];
                kicked_k.insert(k_to_kick);
             }
        }
        if (kicked_k.empty()) { // 1つもキックできなかったら、適当な1つをキック
            int k_to_kick = k_dist(rng);
            if(shortest_lengths[k_to_kick] < 1e9 && !all_shortest_paths_fallback[k_to_kick].empty()) {
                 current_solution.paths_list[k_to_kick] = all_shortest_paths_fallback[k_to_kick];
            }
        }
        recalculate_state(current_solution); // C と steps を再計算

        // [局所探索 (Local Search / "山登り")]
        bool improvement_made = true;
        while (improvement_made && check_time()) {
            improvement_made = false;
            
            for (int k = 0; k < K - 1; ++k) {
                if (shortest_lengths[k] >= 1e9) continue;

                Pos start_node = TARGETS[k], goal_node = TARGETS[k + 1];

                set<Pos> temp_path_cells = {TARGETS[0]};
                int temp_total_steps = 0;
                for (int i = 0; i < K - 1; ++i) {
                    if (i == k || current_solution.paths_list[i].empty()) continue;
                    temp_path_cells.insert(current_solution.paths_list[i].begin(), current_solution.paths_list[i].end());
                    temp_total_steps += (current_solution.paths_list[i].size() - 1);
                }

                int step_limit = T - temp_total_steps;
                if (step_limit < shortest_lengths[k]) continue;

                vector<Pos> new_path_k = find_path_dijkstra_single(start_node, goal_node, step_limit, temp_path_cells);
                if (new_path_k.empty()) continue;
                
                // 評価
                vector<vector<Pos>> new_paths_list = current_solution.paths_list;
                new_paths_list[k] = new_path_k;
                
                set<Pos> new_path_cells = {TARGETS[0]};
                int new_total_steps = 0;
                for (int i = 0; i < K - 1; ++i) {
                    if (new_paths_list[i].empty()) continue;
                    new_path_cells.insert(new_paths_list[i].begin(), new_paths_list[i].end());
                    new_total_steps += (new_paths_list[i].size() - 1);
                }
                int new_C = new_path_cells.size() + 1;

                if (new_C < current_solution.total_C) {
                    current_solution = {new_paths_list, new_path_cells, new_total_steps, new_C};
                    improvement_made = true;
                }
            } 
        } 

        if (current_solution.total_C < global_best_solution.total_C) {
            global_best_solution = current_solution;
        }
    } 


    // --- 5. 最終解の選択 ---
    int C_val = global_best_solution.total_C;
    int Q_val = K;

    // --- 6. 色の割り当てと遷移規則の生成 ---
    map<Pos, int> color_map;
    int current_color = 1; 
    for (const auto& cell : global_best_solution.path_cells) {
        if (color_map.find(cell) == color_map.end()) {
            color_map[cell] = current_color++;
        }
    }

    vector<vector<int>> initial_board(N, vector<int>(N, 0));
    for (auto const& [pos, color] : color_map) {
        initial_board[pos.first][pos.second] = color;
    }

    set<tuple<int, int, int, int, char>> rules;
    for (int k = 0; k < K - 1; ++k) {
        vector<Pos>& path = global_best_solution.paths_list[k];
        if (path.empty()) continue;
        int current_q = k;
        for (size_t p = 0; p < path.size() - 1; ++p) {
            Pos pos = path[p]; Pos next_pos = path[p+1];
            int c = color_map.count(pos) ? color_map[pos] : 0;
            char d = get_direction(pos, next_pos);
            int S = (next_pos == TARGETS[k + 1]) ? k + 1 : k;
            rules.insert({c, current_q, c, S, d});
        }
    }
    int M_val = rules.size();

    // --- 7. 出力 ---
    cout << C_val << " " << Q_val << " " << M_val << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << initial_board[i][j] << (j == N - 1 ? "" : " ");
        }
        cout << "\n";
    }
    for (const auto& rule : rules) {
        cout << get<0>(rule) << " " << get<1>(rule) << " " 
             << get<2>(rule) << " " << get<3>(rule) << " " 
             << get<4>(rule) << "\n";
    }

    return 0;
}