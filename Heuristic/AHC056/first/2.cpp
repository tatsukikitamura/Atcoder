#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <deque>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <chrono> // ### NEW ###: 時間管理

using namespace std;
using namespace std::chrono;

// --- 1. グローバル変数と設定 ---
int N, K, T;
vector<string> V_WALLS;
vector<string> H_WALLS;
vector<pair<int, int>> TARGETS;

// 実行時間制限 (秒)
double TIME_LIMIT_SEC = 1.8;
auto start_time = high_resolution_clock::now();

// ビームサーチ設定
int BEAM_WIDTH = 4;
int N_PATHS_PER_STATE = 3;

// 型定義
using Pos = pair<int, int>; 
using Edge = pair<Pos, Pos>; 
using DijkstraResult = tuple<int, int, vector<Pos>>; // (cost, steps, path)

// --- 2. ヘルパー関数 (変更なし) ---
bool can_move(int i, int j, char d) {
    if (d == 'U') { if (i == 0) return false; return H_WALLS[i - 1][j] == '0'; }
    if (d == 'D') { if (i == N - 1) return false; return H_WALLS[i][j] == '0'; }
    if (d == 'L') { if (j == 0) return false; return V_WALLS[i][j - 1] == '0'; }
    if (d == 'R') { if (j == N - 1) return false; return V_WALLS[i][j] == '0'; }
    return false;
}
Pos get_next_pos(int i, int j, char d) {
    if (d == 'U') return {i - 1, j};
    if (d == 'D') return {i + 1, j};
    if (d == 'L') return {i, j - 1};
    if (d == 'R') return {i, j + 1};
    return {i, j};
}
char get_direction(Pos pos1, Pos pos2) {
    if (pos2.first < pos1.first) return 'U';
    if (pos2.first > pos1.first) return 'D';
    if (pos2.second < pos1.second) return 'L';
    if (pos2.second > pos1.second) return 'R';
    return 'S';
}

// --- 3. BFS (変更なし) ---
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

// --- 3.5. 経路復元ヘルパー ---
vector<Pos> reconstruct_path(map<Pos, map<int, pair<Pos, int>>>& prev, Pos start_pos, Pos goal_pos, int steps) {
    vector<Pos> path;
    Pos curr_pos = goal_pos;
    int curr_steps = steps;
    while (true) {
        path.push_back(curr_pos);
        if (curr_pos == start_pos && curr_steps == 0) break;
        if (prev.find(curr_pos) == prev.end() || prev[curr_pos].find(curr_steps) == prev[curr_pos].end()) {
             // 経路復元失敗 (ロジックエラー or 孤立)
             return {}; // 空のパス
        }
        auto p = prev[curr_pos][curr_steps];
        curr_pos = p.first;
        curr_steps = p.second;
    }
    reverse(path.begin(), path.end());
    return path;
}

// --- 3.6. パレート最適ダイクストラ (ローカルサーチ用 - 単一解) ---
// ### NEW (復活) ###
vector<Pos> find_path_dijkstra_single(Pos start_pos, Pos goal_pos, int step_limit, const set<Pos>& total_path_cells) {
    using State = tuple<int, int, Pos>;
    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({0, 0, start_pos});
    map<Pos, map<int, int>> dist;
    dist[start_pos][0] = 0;
    map<Pos, map<int, pair<Pos, int>>> prev;
    vector<pair<int, int>> found_goals; // (cost, steps)

    while (!pq.empty()) {
        auto [cost, steps, pos] = pq.top(); pq.pop();
        if (dist.count(pos) && dist[pos].count(steps) && dist[pos][steps] < cost) continue;
        if (pos == goal_pos) {
            found_goals.push_back({cost, steps});
            continue;
        }
        if (steps + 1 > step_limit) continue;

        // TLE 対策
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
// (const 修正済み)
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
    if (found_goals.size() > N_PATHS_PER_STATE) {
        found_goals.resize(N_PATHS_PER_STATE);
    }
    vector<DijkstraResult> results;
    for (auto const& [cost, steps] : found_goals) {
        results.emplace_back(cost, steps, reconstruct_path(prev, start_pos, goal_pos, steps));
    }
    return results;
}


// --- 4. メインロジック (ハイブリッド) ---

// 4.0. ビームの状態
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
        if (X_k < 1e9) {
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
    int initial_C = initial_cells.size() + 1; // バグ修正済み
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
                if (path.empty()) continue; // 経路復元失敗
                vector<vector<Pos>> new_paths_list = state.paths_list;
                new_paths_list[k] = path;
                set<Pos> new_path_cells = state.path_cells;
                new_path_cells.insert(path.begin(), path.end());
                int new_total_steps = state.total_steps + steps;
                int new_total_C = new_path_cells.size() + 1; // バグ修正済み
                next_beam.push_back({new_paths_list, new_path_cells, new_total_steps, new_total_C});
            }
        }
        sort(next_beam.begin(), next_beam.end());
        if (next_beam.size() > BEAM_WIDTH) next_beam.resize(BEAM_WIDTH);
        current_beam = next_beam;
        X_future -= X_k;
    }

    // --- 4.4. フェーズ1完了 ---
    BeamState best_solution = current_beam[0];
    
    // --- 4.5. フェーズ2: ローカルサーチ ---
    int loop_count = 0;
    auto check_time = [&]() {
        auto current_time = high_resolution_clock::now();
        return duration_cast<duration<double>>(current_time - start_time).count() < TIME_LIMIT_SEC;
    };

    while (check_time()) {
        int k = loop_count % (K - 1);
        if (shortest_lengths[k] >= 1e9) {
            loop_count++;
            continue;
        }

        Pos start_node = TARGETS[k], goal_node = TARGETS[k + 1];

        // 経路 k を「抜き取り」
        set<Pos> current_path_cells = {TARGETS[0]};
        int current_total_steps = 0;
        for (int i = 0; i < K - 1; ++i) {
            if (i == k || best_solution.paths_list[i].empty()) continue;
            current_path_cells.insert(best_solution.paths_list[i].begin(), best_solution.paths_list[i].end());
            current_total_steps += (best_solution.paths_list[i].size() - 1);
        }

        // `step_limit` を再計算 (kが使える全予算)
        int step_limit = T - current_total_steps;
        if (step_limit < shortest_lengths[k]) {
            loop_count++;
            continue; // 改善の余地なし
        }

        // 「単一解」ダイクストラで再挿入
        vector<Pos> new_path_k = find_path_dijkstra_single(start_node, goal_node, step_limit, current_path_cells);

        if (new_path_k.empty()) {
            loop_count++;
            continue; // 改善失敗
        }
        
        // 評価
        vector<vector<Pos>> new_paths_list = best_solution.paths_list;
        new_paths_list[k] = new_path_k;
        
        set<Pos> new_path_cells = {TARGETS[0]};
        int new_total_steps = 0;
        for (int i = 0; i < K - 1; ++i) {
            if (new_paths_list[i].empty()) continue;
            new_path_cells.insert(new_paths_list[i].begin(), new_paths_list[i].end());
            new_total_steps += (new_paths_list[i].size() - 1);
        }
        int new_C = new_path_cells.size() + 1;

        // 改善していれば更新
        if (new_C < best_solution.total_C) {
            best_solution = {new_paths_list, new_path_cells, new_total_steps, new_C};
        }
        
        loop_count++;
    }


    // --- 5. 最終解の選択 ---
    int C_val = best_solution.total_C;
    int Q_val = K;

    // --- 6. 色の割り当てと遷移規則の生成 ---
    map<Pos, int> color_map;
    int current_color = 1; // 0はダミー
    for (const auto& cell : best_solution.path_cells) {
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
        vector<Pos>& path = best_solution.paths_list[k];
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