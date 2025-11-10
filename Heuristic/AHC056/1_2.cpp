#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <deque>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <chrono>

using namespace std;

// --- 1. グローバル変数と設定 ---
int N, K, T;
vector<string> V_WALLS;
vector<string> H_WALLS;
vector<pair<int, int>> TARGETS;

int BEAM_WIDTH = 5;
int N_PATHS_PER_STATE = 3;

using Pos = pair<int, int>; 
using Edge = pair<Pos, Pos>; 

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
    return {i, j};
}

char get_direction(Pos pos1, Pos pos2) {
    if (pos2.first < pos1.first) return 'U';
    if (pos2.first > pos1.first) return 'D';
    if (pos2.second < pos1.second) return 'L';
    if (pos2.second > pos1.second) return 'R';
    return 'S';
}


// --- 3. BFS (省略なし) ---
// (元のコードのまま)
pair<int, vector<Pos>> bfs_path_and_length(Pos start_pos, Pos goal_pos) {
    deque<pair<Pos, vector<Pos>>> q;
    q.push_back({start_pos, {start_pos}});
    set<Pos> visited = {start_pos};

    while (!q.empty()) {
        Pos pos;
        vector<Pos> path;
        tie(pos, path) = q.front();
        q.pop_front();

        if (pos == goal_pos) {
            return {(int)path.size() - 1, path};
        }

        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                if (visited.find(next_pos) == visited.end()) {
                    visited.insert(next_pos);
                    vector<Pos> new_path = path;
                    new_path.push_back(next_pos);
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
        Pos pos;
        int steps;
        tie(pos, steps) = q.front();
        q.pop_front();

        if (pos == goal_pos) {
            return steps;
        }

        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                Edge edge = {pos, next_pos};
                if (forbidden_edges.count(edge)) {
                    continue;
                }
                if (visited.find(next_pos) == visited.end()) {
                    visited.insert(next_pos);
                    q.push_back({next_pos, steps + 1});
                }
            }
        }
    }
    return 1e9;
}


// --- 3.6. パレート最適ダイクストラ (★変更あり) ---

// コストに long long を使う
using DijkstraResult = tuple<long long, int, vector<Pos>>;

// 経路復元ヘルパー (★変更あり: long long cost)
vector<Pos> reconstruct_path(map<Pos, map<int, pair<Pos, int>>>& prev, Pos start_pos, Pos goal_pos, int steps) {
    vector<Pos> path;
    Pos curr_pos = goal_pos;
    int curr_steps = steps;
    while (true) {
        path.push_back(curr_pos);
        if (curr_pos == start_pos && curr_steps == 0) {
            break;
        }
        auto p = prev[curr_pos][curr_steps];
        curr_pos = p.first;
        curr_steps = p.second;
    }
    reverse(path.begin(), path.end());
    return path;
}

// ★変更: コストスケーリングと潜在価値マップの導入
const long long BASE_COST_SCALE = 1000;
const long long HEURISTIC_WEIGHT = 1;

// ★変更: potential_map を引数に追加
vector<DijkstraResult> find_path_dijkstra_beam(
    Pos start_pos, Pos goal_pos, int step_limit, 
    const set<Pos>& total_path_cells,
    const vector<vector<int>>& potential_map // ★追加
) {
    
    // ★変更: cost を long long に
    using State = tuple<long long, int, Pos>;
    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({0, 0, start_pos});

    // ★変更: cost を long long に
    map<Pos, map<int, long long>> dist;
    dist[start_pos][0] = 0;

    map<Pos, map<int, pair<Pos, int>>> prev;

    // ★変更: cost を long long に
    vector<pair<long long, int>> found_goals;

    while (!pq.empty()) {
        long long cost; // ★変更
        int steps;
        Pos pos;
        tie(cost, steps, pos) = pq.top();
        pq.pop();

        if (dist.count(pos) && dist[pos].count(steps) && dist[pos][steps] < cost) {
            continue;
        }

        if (pos == goal_pos) {
            found_goals.push_back({cost, steps});
            continue;
        }

        if (steps + 1 > step_limit) {
            continue;
        }

        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                int next_steps = steps + 1;
                
                long long new_cost = cost; // ★変更
                
                if (total_path_cells.find(next_pos) == total_path_cells.end()) {
                    // --- ★★★コスト計算の変更★★★ ---
                    // 以前: new_cost += 1;
                    
                    // 変更後:
                    // 基本コスト (1) をスケール
                    new_cost += BASE_COST_SCALE; 
                    // ヒューリスティックコスト (ターゲットに近いほど小さい) を加算
                    new_cost += HEURISTIC_WEIGHT * potential_map[next_pos.first][next_pos.second];
                    // --- ★★★変更ここまで★★★ ---
                }

                bool is_dominated = false;
                if (dist.count(next_pos)) {
                    for (auto const& [existing_steps, existing_cost] : dist[next_pos]) {
                        if (existing_steps <= next_steps && existing_cost <= new_cost) {
                            is_dominated = true;
                            break;
                        }
                    }
                }
                if (is_dominated) {
                    continue;
                }

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

    if (found_goals.empty()) {
        return {}; // None
    }

    sort(found_goals.begin(), found_goals.end());
    if (found_goals.size() > N_PATHS_PER_STATE) {
        found_goals.resize(N_PATHS_PER_STATE);
    }

    vector<DijkstraResult> results;
    for (auto const& [cost, steps] : found_goals) {
        vector<Pos> path = reconstruct_path(prev, start_pos, goal_pos, steps);
        results.emplace_back(cost, steps, path);
    }
    return results;
}


// --- 4. メインロジック (ビームサーチ) ---

// (BeamState, Segment は省略なし)
struct BeamState {
    vector<vector<Pos>> paths_list;
    set<Pos> path_cells;
    int total_steps;
    int total_C;

    // ソート用 (C優先、次にSteps)
    bool operator<(const BeamState& other) const {
        if (total_C != other.total_C) {
            return total_C < other.total_C;
        }
        return total_steps < other.total_steps;
    }
};

struct Segment {
    int k;
    int X_k;
    vector<Pos> path_k;
    double freedom;

    bool operator<(const Segment& other) const {
        return freedom < other.freedom;
    }
};


int main() {
    auto start_time = chrono::high_resolution_clock::now();
    
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> N >> K >> T;
    V_WALLS.resize(N);
    H_WALLS.resize(N - 1);
    TARGETS.resize(K);
    for (int i = 0; i < N; ++i) cin >> V_WALLS[i];
    for (int i = 0; i < N - 1; ++i) cin >> H_WALLS[i];
    for (int i = 0; i < K; ++i) cin >> TARGETS[i].first >> TARGETS[i].second;

    // --- ★★★ 4.0. 潜在価値マップの計算 (マルチソースBFS) ★★★ ---
    vector<vector<int>> potential_map(N, vector<int>(N, 1e9));
    queue<Pos> q_potential;
    for (int k = 0; k < K; ++k) {
        Pos target = TARGETS[k];
        if (potential_map[target.first][target.second] > 0) {
            potential_map[target.first][target.second] = 0;
            q_potential.push(target);
        }
    }
    
    while (!q_potential.empty()) {
        Pos pos = q_potential.front();
        q_potential.pop();
        int current_dist = potential_map[pos.first][pos.second];

        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                if (potential_map[next_pos.first][next_pos.second] == 1e9) {
                    potential_map[next_pos.first][next_pos.second] = current_dist + 1;
                    q_potential.push(next_pos);
                }
            }
        }
    }
    // --- ★★★ ここまで ★★★


    // 4.1. 全区間の自由度を計算 (省略なし)
    vector<Segment> segments_info;
    double total_shortest_steps_X = 0;
    vector<int> shortest_lengths(K - 1);
    vector<vector<Pos>> all_shortest_paths_fallback(K - 1);

    for (int k = 0; k < K - 1; ++k) {
        Pos start_node = TARGETS[k];
        Pos goal_node = TARGETS[k + 1];

        auto [X_k, path_k] = bfs_path_and_length(start_node, goal_node);
        all_shortest_paths_fallback[k] = path_k;
        shortest_lengths[k] = X_k;
        
        double freedom_score = 1e18;
        if (X_k < 1e9) {
            total_shortest_steps_X += X_k;
            set<Edge> forbidden_edges;
            for (size_t i = 0; i < path_k.size() - 1; ++i) {
                forbidden_edges.insert({path_k[i], path_k[i+1]});
            }
            int X_prime_k = bfs_second_path_length(start_node, goal_node, forbidden_edges);
            freedom_score = (double)X_prime_k - X_k;
        }
        segments_info.push_back({k, X_k, path_k, freedom_score});
    }

    // 4.2. 自由度が「低い」順にソート (変更なし)
    sort(segments_info.begin(), segments_info.end());

    // 4.3. ビームサーチの実行 (変更なし)
    vector<BeamState> current_beam;
    set<Pos> initial_cells = {TARGETS[0]};
    int initial_C = initial_cells.size() + 1; 
    current_beam.push_back({vector<vector<Pos>>(K - 1), initial_cells, 0, initial_C});

    double X_future = total_shortest_steps_X;

    for (const auto& segment : segments_info) {
        int k = segment.k;
        int X_k = segment.X_k;
        vector<Pos> path_k_fallback = segment.path_k;

        if (X_k >= 1e9) continue;
        
        Pos start_node = TARGETS[k];
        Pos goal_node = TARGETS[k + 1];
        
        vector<BeamState> next_beam;

        for (const auto& state : current_beam) {
            
            double steps_allowed_total = T - state.total_steps;
            double margin_for_future = X_future - X_k;
            int step_limit = (int)(steps_allowed_total - margin_for_future);
            if (step_limit < X_k) {
                step_limit = X_k;
            }
                
            // ★変更: potential_map を渡す
            vector<DijkstraResult> candidate_paths_info = find_path_dijkstra_beam(
                start_node, 
                goal_node, 
                step_limit, 
                state.path_cells,
                potential_map // ★追加
            );
            
            // ★変更: フォールバックも long long とスケールを合わせる
            if (candidate_paths_info.empty()) {
                long long cost = 0; // ★変更
                for (const auto& cell : path_k_fallback) {
                    if (state.path_cells.find(cell) == state.path_cells.end()) {
                        cost += BASE_COST_SCALE; // ★変更
                        cost += HEURISTIC_WEIGHT * potential_map[cell.first][cell.second]; // ★変更
                    }
                }
                candidate_paths_info.push_back({cost, X_k, path_k_fallback});
            }
                
            // 展開 (変更なし)
            for (const auto& [cost, steps, path] : candidate_paths_info) {
                
                vector<vector<Pos>> new_paths_list = state.paths_list;
                new_paths_list[k] = path;
                
                set<Pos> new_path_cells = state.path_cells;
                new_path_cells.insert(path.begin(), path.end());
                
                int new_total_steps = state.total_steps + steps;
                int new_total_C = new_path_cells.size() + 1;
                
                next_beam.push_back({new_paths_list, new_path_cells, new_total_steps, new_total_C});
            }
        }

        // 4.4. 枝刈り (変更なし)
        sort(next_beam.begin(), next_beam.end());
        if (next_beam.size() > BEAM_WIDTH) {
            next_beam.resize(BEAM_WIDTH);
        }
        current_beam = next_beam;
        
        X_future -= X_k;
    }

    // --- 5. 最終解の選択 (変更なし) ---
    BeamState best_solution = current_beam[0];
    int C_val = best_solution.total_C;
    int Q_val = K;

    // --- 6. 色の割り当てと遷移規則の生成 (変更なし) ---
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
            Pos pos = path[p];
            Pos next_pos = path[p+1];
            
            int c = color_map.count(pos) ? color_map[pos] : 0;
            char d = get_direction(pos, next_pos);
            int A = c; 
            int S = (next_pos == TARGETS[k + 1]) ? k + 1 : k;
            
            rules.insert({c, current_q, A, S, d});
        }
    }
    int M_val = rules.size();


    // --- 7. 出力 (変更なし) ---
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

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "実行時間: " << duration.count() << " ms" << endl;
    cerr << "スコア: " << (C_val + Q_val) << endl;

    return 0;
}