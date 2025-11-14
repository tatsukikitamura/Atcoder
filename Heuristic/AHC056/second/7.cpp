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
#include <cmath>     // exp() のために追加
#include <random>    // 乱数生成のために追加

using namespace std;

// --- 1. グローバル変数と設定 ---
int N, K, T;
vector<string> V_WALLS;
vector<string> H_WALLS;
vector<pair<int, int>> TARGETS;

int BEAM_WIDTH = 20;
int N_PATHS_PER_STATE = 10;

using Pos = pair<int, int>; 
using Edge = pair<Pos, Pos>; 

// --- 2. ヘルパー関数 (変更なし) ---
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


// --- 3. BFS (変更なし) ---
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


// --- 3.6. パレート最適ダイクストラ (★変更あり★) ---

using DijkstraResult = tuple<long long, int, vector<Pos>>;

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

const long long BASE_COST_SCALE = 10000; // C(新規マス)のペナルティ
const long long HEURISTIC_WEIGHT = 1;
const long long LOOP_PENALTY_WEIGHT = 1500; 
const int LOOP_PENALTY_THRESHOLD = 5; 
// const long long REWARD_WEIGHT = 50;     // (削除)
const long long TURN_PENALTY_WEIGHT = 50; // ★ 方向転換ペナルティ (要調整)

vector<DijkstraResult> find_path_dijkstra_beam(
    Pos start_pos, Pos goal_pos, int step_limit, 
    const set<Pos>& total_path_cells, // ★ 変更 (setに戻す)
    const vector<vector<int>>& potential_map, 
    const vector<vector<bool>>& forbidden_cells, 
    const vector<vector<int>>& freedom 
) {
    
    using State = tuple<long long, int, Pos>;
    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({0, 0, start_pos});

    map<Pos, map<int, long long>> dist;
    dist[start_pos][0] = 0;

    map<Pos, map<int, pair<Pos, int>>> prev; // 経路復元と方向特定用

    vector<pair<long long, int>> found_goals;

    while (!pq.empty()) {
        long long cost; 
        int steps;
        Pos pos;
        tie(cost, steps, pos) = pq.top();
        pq.pop();

        // ★ 変更点: ここで「入ってきた方向」を取得
        char incoming_d = 'S'; // 'S' = Start/Stop
        if (steps > 0 && prev.count(pos) && prev[pos].count(steps)) {
            Pos prev_pos = prev[pos][steps].first;
            incoming_d = get_direction(prev_pos, pos);
        }
        // ★ 変更点ここまで

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

        for (char d : {'U', 'D', 'L', 'R'}) { // d は「出ていく方向」
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);

                if (forbidden_cells[next_pos.first][next_pos.second]) {
                    continue;
                }

                int next_steps = steps + 1;
                long long new_cost = cost; 
                
                // ★ 変更点: 方向転換ペナルティを加算
                if (incoming_d != 'S' && d != incoming_d) {
                    new_cost += TURN_PENALTY_WEIGHT;
                }

                // ★ 変更点: コスト関数 (setベースに戻す)
                if (total_path_cells.find(next_pos) == total_path_cells.end()) { 
                    // --- 新規マス ---
                    new_cost += BASE_COST_SCALE; 
                    new_cost += HEURISTIC_WEIGHT * potential_map[next_pos.first][next_pos.second];
                    
                    if (freedom[next_pos.first][next_pos.second] == 2 && 
                        potential_map[next_pos.first][next_pos.second] >= LOOP_PENALTY_THRESHOLD) {
                        new_cost += LOOP_PENALTY_WEIGHT;
                    }
                } 
                // (既存マスの報酬ロジックは削除)
                
                // ... (以降のパレート最適ロジックは変更なし) ...

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
    if (found_goals.size() > static_cast<size_t>(N_PATHS_PER_STATE)) {
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

// ★ 変更点: BeamState 構造体 (setに戻す)
struct BeamState {
    vector<vector<Pos>> paths_list;
    set<Pos> path_cells; // ★ 変更
    int total_steps;
    int total_C; // (ヒューリスティックなC。path_cells.size() + 1)

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
    auto start_time = chrono::high_resolution_clock::now(); // 時間計測開始
    
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> N >> K >> T;
    V_WALLS.resize(N);
    H_WALLS.resize(N - 1);
    TARGETS.resize(K);
    for (int i = 0; i < N; ++i) cin >> V_WALLS[i];
    for (int i = 0; i < N - 1; ++i) cin >> H_WALLS[i];
    for (int i = 0; i < K; ++i) cin >> TARGETS[i].first >> TARGETS[i].second;

    // --- パラメータ調整 (変更なし) ---
    double density = static_cast<double>(K) / (static_cast<double>(N) * static_cast<double>(N));
    
    if (N >= 19) {
        if (density >= 0.75) {
            BEAM_WIDTH =  5;
            N_PATHS_PER_STATE = 3;
        } 
        else if (density >= 0.625) {
            BEAM_WIDTH =  6;
            N_PATHS_PER_STATE = 3;
        } 
        else if (density >= 0.5) {
            BEAM_WIDTH =  7;
            N_PATHS_PER_STATE = 4;
        } 
        else {
            BEAM_WIDTH = 8;
            N_PATHS_PER_STATE = 4;
        }
    } else if (N >= 17) {
        if (density >= 0.875) {
            BEAM_WIDTH = 6;
            N_PATHS_PER_STATE = 3;
        } else if (density >= 0.75) {
            BEAM_WIDTH = 8;
            N_PATHS_PER_STATE = 4;
        } else if (density >= 0.5) {
            BEAM_WIDTH = 9;
            N_PATHS_PER_STATE = 5;
        } else {
            BEAM_WIDTH = 10;
            N_PATHS_PER_STATE = 5;
        }
    } else if (N >= 15) {
        if (density >= 0.875) {
            BEAM_WIDTH = 6;
            N_PATHS_PER_STATE = 3;
        } else if (density >= 0.75) {
            BEAM_WIDTH = 8;
            N_PATHS_PER_STATE = 4;
        } else if (density >= 0.5) { 
            BEAM_WIDTH = 9;
            N_PATHS_PER_STATE = 5;
        } else { 
            BEAM_WIDTH = 10;
            N_PATHS_PER_STATE = 5;
        }
    } else { // N < 15
        if (density >= 0.875) {
            BEAM_WIDTH = 10;
            N_PATHS_PER_STATE = 5;
        } else if (density >= 0.75) {
            BEAM_WIDTH = 12;
            N_PATHS_PER_STATE = 6;
        } else if (density >= 0.5) {
            BEAM_WIDTH = 14;
            N_PATHS_PER_STATE = 7;
        } else {
            BEAM_WIDTH = 16;
            N_PATHS_PER_STATE = 8;
        }
    }
    // --- パラメータ調整ここまで ---


    // --- 4.0. 潜在価値マップの計算 (変更なし) ---
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

    // --- 4.1. 禁止マス(袋小路の奥)の計算 (変更なし) ---
    vector<vector<int>> freedom(N, vector<int>(N, 0));
    vector<vector<bool>> is_target(N, vector<bool>(N, false));
    for(int k=0; k<K; ++k) is_target[TARGETS[k].first][TARGETS[k].second] = true;

    vector<Pos> leafs;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (char d : {'U', 'D', 'L', 'R'}) {
                if (can_move(i, j, d)) freedom[i][j]++;
            }
            if (freedom[i][j] == 1 && !is_target[i][j]) {
                leafs.push_back({i, j});
            }
        }
    }

    vector<vector<bool>> forbidden_cells(N, vector<bool>(N, false));
    vector<vector<bool>> visited_trace(N, vector<bool>(N, false));

    for (const auto& leaf : leafs) {
        if (visited_trace[leaf.first][leaf.second]) continue;

        vector<Pos> path;
        Pos curr = leaf;
        Pos prev = {-1, -1};
        bool path_is_useless = true; 

        while (true) {
            visited_trace[curr.first][curr.second] = true;
            path.push_back(curr);
            
            Pos next_neighbor = {-1, -1};
            for (char d : {'U', 'D', 'L', 'R'}) {
                if (can_move(curr.first, curr.second, d)) {
                    Pos next = get_next_pos(curr.first, curr.second, d);
                    if (next != prev) {
                        next_neighbor = next;
                        break; 
                    }
                }
            }

            if (next_neighbor == Pos{-1, -1}) {
                break;
            }

            if (is_target[next_neighbor.first][next_neighbor.second]) {
                path_is_useless = true; 
                break;
            }
            if (freedom[next_neighbor.first][next_neighbor.second] > 2) {
                path_is_useless = false; 
                break;
            }
            if (visited_trace[next_neighbor.first][next_neighbor.second]) {
                path_is_useless = false; 
                break;
            }
            
            prev = curr;
            curr = next_neighbor;
        }

        if (path_is_useless) {
            for (const auto& p : path) {
                forbidden_cells[p.first][p.second] = true;
            }
        }
    }


    // 4.2. 全区間の自由度を計算 (変更なし)
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

    // 4.3. 自由度が「低い」順にソート (変更なし)
    sort(segments_info.begin(), segments_info.end());

    // 4.4. ビームサーチの実行 (★変更あり★)
    vector<BeamState> current_beam;
    set<Pos> initial_cells = {TARGETS[0]}; // ★ 変更 (setに戻す)
    int initial_C = initial_cells.size() + 1; 
    current_beam.push_back({vector<vector<Pos>>(K - 1), initial_cells, 0, initial_C});

    double X_future = total_shortest_steps_X;

    for (const auto& segment : segments_info) {
        auto current_time_beam = chrono::high_resolution_clock::now();
        auto elapsed_ms_beam = chrono::duration_cast<chrono::milliseconds>(current_time_beam - start_time).count();
        if (elapsed_ms_beam >= 1800) { // 1.8秒でビームサーチを打ち切り
            cerr << "ビームサーチ時間切れ: " << elapsed_ms_beam << " ms" << endl;
            break;
        }
        
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
                    
            vector<DijkstraResult> candidate_paths_info = find_path_dijkstra_beam(
                start_node, 
                goal_node, 
                step_limit, 
                state.path_cells, // ★ 変更 (setに戻す)
                potential_map,
                forbidden_cells, 
                freedom
            );
            
            if (candidate_paths_info.empty()) {
                long long cost = 0; 
                bool is_forbidden = false;
                for (const auto& cell : path_k_fallback) {
                    if (forbidden_cells[cell.first][cell.second]) {
                        is_forbidden = true;
                    }
                    if (state.path_cells.find(cell) == state.path_cells.end()) { // ★ 変更 (setに戻す)
                        cost += BASE_COST_SCALE; 
                        cost += HEURISTIC_WEIGHT * potential_map[cell.first][cell.second];
                        if (freedom[cell.first][cell.second] == 2 && 
                            potential_map[cell.first][cell.second] >= LOOP_PENALTY_THRESHOLD) {
                            cost += LOOP_PENALTY_WEIGHT;
                        }
                    } 
                    // (報酬ロジックは削除)
                }
                if (is_forbidden) {
                    cost += 1e18; 
                }
                candidate_paths_info.push_back({cost, X_k, path_k_fallback});
            }
                    
            for (const auto& [cost, steps, path] : candidate_paths_info) {
                
                vector<vector<Pos>> new_paths_list = state.paths_list;
                new_paths_list[k] = path;
                
                set<Pos> new_path_cells = state.path_cells; // ★ 変更 (setに戻す)
                new_path_cells.insert(path.begin(), path.end()); // ★ 変更 (setに戻す)

                int new_total_steps = state.total_steps + steps;
                int new_total_C = new_path_cells.size() + 1; // ★ 変更 (setに戻す)
                
                next_beam.push_back({new_paths_list, new_path_cells, new_total_steps, new_total_C});
            }
        }

        // 4.5. 枝刈り (変更なし)
        sort(next_beam.begin(), next_beam.end());
        if (next_beam.size() > static_cast<size_t>(BEAM_WIDTH)) {
            next_beam.resize(BEAM_WIDTH);
        }
        current_beam = next_beam;
        
        X_future -= X_k;
    }

    // --- 5. 最終解の選択 と 焼きなまし法による改善 (★変更あり★) ---
    
    // (A) ビームサーチの解（粒子）を取得
    BeamState best_solution; 
    
    if (current_beam.empty()) {
        cerr << "警告: ビームが空です。最短経路でフォールバックします。" << endl;
        vector<vector<Pos>> fallback_paths;
        set<Pos> fallback_cells = {TARGETS[0]}; // ★ 変更 (setに戻す)
        int fallback_total_steps = 0;
        for (int k = 0; k < K - 1; ++k) {
            fallback_paths.push_back(all_shortest_paths_fallback[k]);
            fallback_cells.insert(all_shortest_paths_fallback[k].begin(), all_shortest_paths_fallback[k].end()); // ★ 変更 (setに戻す)
            fallback_total_steps += shortest_lengths[k];
        }
        best_solution = {fallback_paths, fallback_cells, fallback_total_steps, (int)fallback_cells.size() + 1}; // ★ 変更 (setに戻す)
    } else {
        
        // (B) 不完全な解を補完 (全粒子に適用)
        for (auto& state : current_beam) {
            bool is_incomplete = false;
            for (int k = 0; k < K - 1; ++k) {
                if (state.paths_list[k].empty()) {
                    is_incomplete = true;
                    break;
                }
            }

            if (is_incomplete) {
                cerr << "警告: 途中の解が不完全です。最短経路で補完します。" << endl;
                set<Pos> new_cells = state.path_cells; // ★ 変更 (setに戻す)
                vector<vector<Pos>> new_paths = state.paths_list;
                int new_steps = state.total_steps;
                
                for (int k = 0; k < K - 1; ++k) {
                    if (new_paths[k].empty()) {
                        new_paths[k] = all_shortest_paths_fallback[k];
                        new_cells.insert(all_shortest_paths_fallback[k].begin(), all_shortest_paths_fallback[k].end()); // ★ 変更 (setに戻す)
                        new_steps += shortest_lengths[k];
                    }
                }
                state = {new_paths, new_cells, new_steps, (int)new_cells.size() + 1}; // ★ 変更 (setに戻す)
            }
        }
        
        // (C) ★★★ 焼きなまし法 (粒子ごとに実行) ★★★
        
        auto sa_start_time_point = chrono::high_resolution_clock::now();
        long long elapsed_ms_before_sa = chrono::duration_cast<chrono::milliseconds>(sa_start_time_point - start_time).count();
        const long long TIME_LIMIT_MS = 1970;
        long long remaining_ms = TIME_LIMIT_MS - elapsed_ms_before_sa;

        int N_PARTICLES = min((int)current_beam.size(), 3); 
        
        best_solution = current_beam[0]; 
        
        if (remaining_ms > 0 && N_PARTICLES > 0) {
            
            vector<double> time_allocations;
            if (N_PARTICLES == 1) {
                time_allocations = {1.0}; 
            } else if (N_PARTICLES == 2) {
                time_allocations = {0.6, 0.4}; 
            } else { 
                time_allocations = {0.5, 0.3, 0.2}; 
            }
            
            cerr << "SA開始: 残り " << remaining_ms << "ms を " << N_PARTICLES << " 粒子で傾斜配分" << endl;

            for (int i = 0; i < N_PARTICLES; ++i) {
                
                long long time_per_particle = (long long)(remaining_ms * time_allocations[i]);
                if (time_per_particle <= 0) continue; 

                long long sa_particle_start_time_ms = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count();
                long long sa_particle_end_time_ms = sa_particle_start_time_ms + time_per_particle;

                BeamState particle_solution = current_beam[i]; 
                
                const double START_TEMP = 1.5; 
                const double END_TEMP = 0.01; 
                
                random_device rd;
                mt19937 rnd_gen(rd());
                uniform_real_distribution<> sa_prob_dist(0.0, 1.0); 
                set<pair<int, Pos>> tried_improve; 

                while (true) {
                    auto current_time = chrono::high_resolution_clock::now();
                    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(current_time - start_time).count();
                    if (elapsed_ms >= sa_particle_end_time_ms) {
                        break; 
                    }

                    double time_ratio = min(1.0, (double)(elapsed_ms - sa_particle_start_time_ms) / time_per_particle);
                    double current_temp = START_TEMP + (END_TEMP - START_TEMP) * time_ratio;
                    
                    vector<double> k_weights(K - 1);
                    double total_weight = 0;
                    for (int k = 0; k < K - 1; ++k) {
                        if (particle_solution.paths_list[k].empty()) {
                            k_weights[k] = 0;
                        } else {
                            double weight = max(1.0, (double)particle_solution.paths_list[k].size() - 1);
                            k_weights[k] = weight;
                            total_weight += weight;
                        }
                    }
                    if (total_weight == 0) { 
                        fill(k_weights.begin(), k_weights.end(), 1.0);
                    }
                    discrete_distribution<> k_dist_weighted(k_weights.begin(), k_weights.end());

                    // ★ 変更点: SAループ内で usage_count を毎回計算
                    map<Pos, int> usage_count;
                    for (int k = 0; k < K - 1; ++k) {
                         if (particle_solution.paths_list[k].empty()) continue;
                        set<Pos> path_unique_cells(particle_solution.paths_list[k].begin(), particle_solution.paths_list[k].end()); 
                        for (const auto& cell : path_unique_cells) {
                            usage_count[cell]++;
                        }
                    }
                    
                    if (sa_prob_dist(rnd_gen) < 0.8) 
                    {
                        // --- 近傍A: C改善 (確率80%) ---
                        Pos target_cell = {-1, -1};
                        int target_k = -1;
                        
                        for (int k_idx = segments_info.size() - 1; k_idx >= 0; --k_idx) {
                            int k = segments_info[k_idx].k; 
                            if (particle_solution.paths_list[k].empty()) continue;
                            for (const auto& cell : particle_solution.paths_list[k]) {
                                if (usage_count.count(cell) && usage_count[cell] == 1 && // ★ 変更 (usage_count)
                                    cell != TARGETS[k] && cell != TARGETS[k+1] &&
                                    tried_improve.find({k, cell}) == tried_improve.end()) { 
                                    target_cell = cell;
                                    target_k = k;
                                    break;
                                }
                            }
                            if (target_k != -1) break;
                        }
                        if (target_k == -1) { tried_improve.clear(); continue; }
                        tried_improve.insert({target_k, target_cell});

                        vector<vector<bool>> local_forbidden_cells = forbidden_cells; 
                        local_forbidden_cells[target_cell.first][target_cell.second] = true; 
                        
                        set<Pos> other_cells; // ★ 変更 (setに戻す)
                        for (int k = 0; k < K - 1; ++k) {
                            if (k == target_k) continue; 
                            if (particle_solution.paths_list[k].empty()) continue;
                            other_cells.insert(particle_solution.paths_list[k].begin(), particle_solution.paths_list[k].end()); // ★ 変更
                        }

                        int current_steps_k = particle_solution.paths_list[target_k].size() - 1;
                        int other_steps = particle_solution.total_steps - current_steps_k;
                        int step_limit = T - other_steps; 

                        vector<DijkstraResult> candidates = find_path_dijkstra_beam(
                            TARGETS[target_k], TARGETS[target_k + 1], step_limit, 
                            other_cells, potential_map, local_forbidden_cells, freedom // ★ 変更
                        );
                        if (candidates.empty()) continue; 

                        auto [best_new_cost, best_new_steps, best_new_path] = candidates[0];
                        
                        set<Pos> new_total_cells = other_cells; // ★ 変更 (setに戻す)
                        new_total_cells.insert(best_new_path.begin(), best_new_path.end()); // ★ 変更
                        int new_C = new_total_cells.size() + 1; // ★ 変更
                        int new_total_steps = other_steps + best_new_steps; 
                        int delta_score = (new_C - particle_solution.total_C);
                        if (new_total_steps > T) delta_score += 1e9; 

                        if (delta_score < 0) {
                            particle_solution.paths_list[target_k] = best_new_path;
                            particle_solution.path_cells = new_total_cells; // ★ 更新
                            particle_solution.total_C = new_C; // ★ 更新
                            particle_solution.total_steps = new_total_steps; // ★ 更新
                            tried_improve.clear();
                        } else {
                            double probability = exp(-(double)delta_score / current_temp);
                            if (sa_prob_dist(rnd_gen) < probability) {
                                particle_solution.paths_list[target_k] = best_new_path;
                                particle_solution.path_cells = new_total_cells; // ★ 更新
                                particle_solution.total_C = new_C; // ★ 更新
                                particle_solution.total_steps = new_total_steps; // ★ 更新
                                tried_improve.clear();
                            }
                        }
                    } 
                    else 
                    {
                        // --- 近傍B: T改善 (確率20%) ---
                        int target_k = k_dist_weighted(rnd_gen); 

                        if (particle_solution.paths_list[target_k].empty()) continue;
                        Pos target_cell = {-1, -1};
                        for (const auto& cell : particle_solution.paths_list[target_k]) {
                            if (usage_count.count(cell) && usage_count[cell] >= 2 && // ★ 変更 (usage_count)
                                cell != TARGETS[target_k] && cell != TARGETS[target_k + 1]) {
                                target_cell = cell;
                                break; 
                            }
                        }
                        if (target_cell.first == -1) continue; 

                        vector<vector<bool>> local_forbidden_cells = forbidden_cells;
                        local_forbidden_cells[target_cell.first][target_cell.second] = true;
                        
                        set<Pos> other_cells; // ★ 変更 (setに戻す)
                        for (int k = 0; k < K - 1; ++k) {
                            if (k == target_k) continue; 
                            if (particle_solution.paths_list[k].empty()) continue;
                            other_cells.insert(particle_solution.paths_list[k].begin(), particle_solution.paths_list[k].end()); // ★ 変更
                        }
                        
                        int current_steps_k = particle_solution.paths_list[target_k].size() - 1;
                        int other_steps = particle_solution.total_steps - current_steps_k;
                        int step_limit = T - other_steps; 

                        vector<DijkstraResult> candidates = find_path_dijkstra_beam(
                            TARGETS[target_k], TARGETS[target_k + 1], step_limit, 
                            other_cells, potential_map, local_forbidden_cells, freedom // ★ 変更
                        );
                        if (candidates.empty()) continue; 

                        auto [best_new_cost, best_new_steps, best_new_path] = candidates[0];
                        
                        set<Pos> new_total_cells = other_cells; // ★ 変更 (setに戻す)
                        new_total_cells.insert(best_new_path.begin(), best_new_path.end()); // ★ 変更
                        int new_C = new_total_cells.size() + 1; // ★ 変更
                        int new_total_steps = other_steps + best_new_steps;

                        const double W_C_FOR_T_IMPROVE = 10.0; 
                        
                        int delta_T = new_total_steps - particle_solution.total_steps;
                        int delta_C = new_C - particle_solution.total_C;

                        double delta_score = (double)delta_T + W_C_FOR_T_IMPROVE * (double)max(0, delta_C);
                        
                        if (delta_T > 0 && delta_C >= 0) {
                            delta_score = 1e9; 
                        }

                        if (delta_score < 0 || sa_prob_dist(rnd_gen) < exp(-delta_score / current_temp)) 
                        {
                            particle_solution.paths_list[target_k] = best_new_path;
                            particle_solution.path_cells = new_total_cells; // ★ 更新
                            particle_solution.total_C = new_C; // ★ 更新
                            particle_solution.total_steps = new_total_steps; // ★ 更新
                            tried_improve.clear();
                        }
                    }
                } // ★★★ 1粒子のSAここまで ★★★

                if (particle_solution < best_solution) {
                    cerr << "  -> 粒子 " << i << " が総合最良解を更新 (C: " << particle_solution.total_C << ", T: " << particle_solution.total_steps << ")" << endl;
                    best_solution = particle_solution;
                }

            } // --- 全粒子のSAループ終了 ---
        } else {
             cerr << "SA時間なし。ビームサーチの最良解を採用します。" << endl;
        }
    } // if (current_beam.empty()) ... else ... の終了
    
    // --- 6. ★★★ 高度な座標圧縮と遷移規則の生成 (★TLE対策 変更済み★) ★★★
    
    // この時点で best_solution には、最良解が格納されている。
    
    // 6.1. 「どのマスが」「どの状態で」「どう動くか」のマップを作成
    map<Pos, map<int, pair<int, char>>> rules_by_pos;
    
    // 6.2. 実際に使用するルールセット
    set<tuple<int, int, int, int, char>> final_rules_by_color_id;
    
    // 6.3. 「動作」 -> 色ID のマッピング (★ TLE対策: mapキーをvectorキーに変更)
    map<vector<tuple<int, int, char>>, int> behavior_to_color_id_vec; 
    int next_color_id = 1; // 色0 (未使用マス) 以外は 1 からスタート
    
    // 6.4. 最終的な (Pos -> 色ID) のマッピング
    map<Pos, int> final_color_map;

    
    for (int k = 0; k < K - 1; ++k) {
        vector<Pos>& path = best_solution.paths_list[k];
        if (path.empty()) continue;
        
        int current_q = k;
        
        for (size_t p = 0; p < path.size() - 1; ++p) {
            Pos pos = path[p];
            Pos next_pos = path[p+1];
            
            char d = get_direction(pos, next_pos);
            int S = (next_pos == TARGETS[k + 1]) ? k + 1 : k;
            
            rules_by_pos[pos][current_q] = {S, d};
        }
    }

    // 6.5. 動作マップから色IDを決定し、最終的なルールを構築
    int C_val = 1; 
    int Q_val = K;

    // (★ TLE対策: 目的地マスを先に処理する必要はなくなったため、ループを1つに統合)
    for (auto const& [pos, behavior_map] : rules_by_pos) {
        
        // ★ TLE対策: map -> vector 変換
        vector<tuple<int, int, char>> behavior_vec;
        for (auto const& [q, action] : behavior_map) {
            behavior_vec.emplace_back(q, action.first, action.second); // (q, S, D)
        }
        // ★ ソートは不要 (mapから来ているので既にqでソート済み)

        int color_id;
        // この動作(behavior)パターンに初めて遭遇した場合
        if (behavior_to_color_id_vec.find(behavior_vec) == behavior_to_color_id_vec.end()) { // ★ 変更
            color_id = next_color_id++;
            behavior_to_color_id_vec[behavior_vec] = color_id; // ★ 変更
            
            // この新しい色ID (color_id) が行うべき動作を登録
            for (auto const& [q, S, D] : behavior_vec) { // ★ 変更
                int A = color_id; 
                final_rules_by_color_id.insert({color_id, q, A, S, D});
            }
        } else {
            // 既に登録済みの動作パターン
            color_id = behavior_to_color_id_vec[behavior_vec]; // ★ 変更
        }
        
        // このマス(Pos)には、この色IDを割り当てる
        final_color_map[pos] = color_id;
    }
    
    C_val = next_color_id; 
    int M_val = final_rules_by_color_id.size();

    // 6.6. 初期盤面の生成
    vector<vector<int>> initial_board(N, vector<int>(N, 0)); 
    for (auto const& [pos, color] : final_color_map) {
        initial_board[pos.first][pos.second] = color;
    }


    // --- 7. 出力 (変更なし) ---
    cout << C_val << " " << Q_val << " " << M_val << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << initial_board[i][j] << (j == N - 1 ? "" : " ");
        }
        cout << "\n";
    }
    for (const auto& rule : final_rules_by_color_id) { 
        cout << get<0>(rule) << " " << get<1>(rule) << " " 
             << get<2>(rule) << " " << get<3>(rule) << " " 
             << get<4>(rule) << "\n";
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cerr << "実行時間: " << duration.count() << " ms" << endl;
    cerr << "スコア (C+Q): " << (C_val + Q_val) << endl;
    cerr << "圧縮後のC: " << C_val << ", Q: " << Q_val << endl;
    cerr << "BW: " << BEAM_WIDTH << ", NPPS: " << N_PATHS_PER_STATE << endl;

    return 0;
}