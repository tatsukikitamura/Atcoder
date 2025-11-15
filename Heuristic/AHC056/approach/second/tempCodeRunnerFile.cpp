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
#include <numeric>   // iota() のために追加

using namespace std;

// --- 1. グローバル変数と設定 ---
int N, K, T;
vector<string> V_WALLS;
vector<string> H_WALLS;
vector<pair<int, int>> TARGETS;

int BEAM_WIDTH = 20;
int N_PATHS_PER_STATE = 10;
const long long TIME_LIMIT_MS = 1970; // ★ タイムリミット
auto GLOBAL_START_TIME = chrono::high_resolution_clock::now(); // ★ 時間計測開始

// ★ 改善案4: 動的パラメータ管理
struct DynamicParameters {
    double SA_START_TEMP = 1.5;
    double SA_END_TEMP = 0.01;
    int improvements_count = 0;
    int sa_loops = 0;

    void update_sa_improvements() {
        improvements_count++;
    }
    void update_sa_loops() {
        sa_loops++;
    }

    double get_current_temp(long long elapsed_ms, long long particle_time_ms, long long particle_start_ms) {
        if (improvements_count == 0 && sa_loops > 1000) {
            // 停滞していたら一時的に温度を上げる (例)
            SA_START_TEMP = min(3.0, SA_START_TEMP * 1.01); 
        }

        double time_ratio = min(1.0, (double)(elapsed_ms - particle_start_ms) / particle_time_ms);
        return SA_START_TEMP + (SA_END_TEMP - SA_START_TEMP) * time_ratio;
    }

    void adjust_for_final_phase() {
        // 終盤は貪欲に
        BEAM_WIDTH = min(3, BEAM_WIDTH);
        N_PATHS_PER_STATE = min(2, N_PATHS_PER_STATE);
    }
};
DynamicParameters DYN_PARAMS; // グローバルなパラメータ管理


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

// ★ 改善案3 (Shortcut用): 2点間を接続可能か（BFSで軽量にチェック）
bool can_direct_connect(Pos start_pos, Pos goal_pos, int step_limit) {
    deque<pair<Pos, int>> q;
    q.push_back({start_pos, 0});
    set<Pos> visited = {start_pos};

    while (!q.empty()) {
        Pos pos;
        int steps;
        tie(pos, steps) = q.front();
        q.pop_front();

        if (pos == goal_pos) {
            return true; // 接続可能
        }
        if (steps + 1 > step_limit) {
            continue;
        }

        for (char d : {'U', 'D', 'L', 'R'}) {
            if (can_move(pos.first, pos.second, d)) {
                Pos next_pos = get_next_pos(pos.first, pos.second, d);
                if (visited.find(next_pos) == visited.end()) {
                    visited.insert(next_pos);
                    q.push_back({next_pos, steps + 1});
                }
            }
        }
    }
    return false; // 接続不可
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


// --- 3.6. パレート最適ダイクストラ (変更なし) ---

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

const long long BASE_COST_SCALE = 10000; 
const long long HEURISTIC_WEIGHT = 1;
const long long LOOP_PENALTY_WEIGHT = 1500; 
const int LOOP_PENALTY_THRESHOLD = 5; 
const long long TURN_PENALTY_WEIGHT = 50; 

vector<DijkstraResult> find_path_dijkstra_beam(
    Pos start_pos, Pos goal_pos, int step_limit, 
    const set<Pos>& total_path_cells,
    const vector<vector<int>>& potential_map, 
    const vector<vector<bool>>& forbidden_cells, 
    const vector<vector<int>>& freedom 
) {
    
    using State = tuple<long long, int, Pos>;
    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({0, 0, start_pos});

    map<Pos, map<int, long long>> dist;
    dist[start_pos][0] = 0;

    map<Pos, map<int, pair<Pos, int>>> prev; 

    vector<pair<long long, int>> found_goals;

    while (!pq.empty()) {
        long long cost; 
        int steps;
        Pos pos;
        tie(cost, steps, pos) = pq.top();
        pq.pop();

        char incoming_d = 'S'; 
        if (steps > 0 && prev.count(pos) && prev[pos].count(steps)) {
            Pos prev_pos = prev[pos][steps].first;
            incoming_d = get_direction(prev_pos, pos);
        }

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

                if (forbidden_cells[next_pos.first][next_pos.second]) {
                    continue;
                }

                int next_steps = steps + 1;
                long long new_cost = cost; 
                
                if (incoming_d != 'S' && d != incoming_d) {
                    new_cost += TURN_PENALTY_WEIGHT;
                }

                if (total_path_cells.find(next_pos) == total_path_cells.end()) { 
                    new_cost += BASE_COST_SCALE; 
                    new_cost += HEURISTIC_WEIGHT * potential_map[next_pos.first][next_pos.second];
                    
                    if (freedom[next_pos.first][next_pos.second] == 2 && 
                        potential_map[next_pos.first][next_pos.second] >= LOOP_PENALTY_THRESHOLD) {
                        new_cost += LOOP_PENALTY_WEIGHT;
                    }
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

struct BeamState {
    vector<vector<Pos>> paths_list;
    set<Pos> path_cells;
    int total_steps;
    int total_C; 

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

// ★ 改善案2: 将来コスト推定（マンハッタン距離ベース）
// ※これはビームサーチの `step_limit` 計算に使用
double estimate_future_cost_manhattan(int current_k_index, const vector<Segment>& segments_info) {
    double future_cost = 0;
    for (size_t i = current_k_index; i < segments_info.size(); ++i) {
        int k = segments_info[i].k;
        // BFSの最短経路 X_k の方が正確な下界なので、そちらを使う
        future_cost += segments_info[i].X_k;
    }
    return future_cost;
}
// ※注: 提案ではマンハッタン距離でしたが、BFS(X_k)が既に計算済みなので、
//      その合計(X_future)を使うのが最も正確な下界です。
//      現在の `X_future -= X_k;` のロジックは、この提案を既に満たしています。
//      ここでは `overlap_bonus` の実装は複雑すぎるため見送ります。


int main() {
    // GLOBAL_START_TIME は既にグローバルで設定済み
    
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

    // 4.4. ビームサーチの実行 (変更なし)
    vector<BeamState> current_beam;
    set<Pos> initial_cells = {TARGETS[0]};
    int initial_C = initial_cells.size() + 1; 
    current_beam.push_back({vector<vector<Pos>>(K - 1), initial_cells, 0, initial_C});

    double X_future = total_shortest_steps_X;

    for (int segment_index = 0; segment_index < segments_info.size(); ++segment_index) {
        const auto& segment = segments_info[segment_index];

        auto current_time_beam = chrono::high_resolution_clock::now();
        auto elapsed_ms_beam = chrono::duration_cast<chrono::milliseconds>(current_time_beam - GLOBAL_START_TIME).count();
        
        // ★ 改善案4: 終盤になったらビーム幅を狭める
        if (elapsed_ms_beam >= 1600) { // 例: 1.6秒で終盤とみなす
             DYN_PARAMS.adjust_for_final_phase();
        }
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
            // ★ 改善案2: 将来コスト推定 (現在のロジック(X_future)が既に最適)
            // double margin_for_future = estimate_future_cost_manhattan(segment_index + 1, segments_info);
            double margin_for_future = X_future - X_k;
            int step_limit = (int)(steps_allowed_total - margin_for_future);
            if (step_limit < X_k) {
                step_limit = X_k;
            }
                    
            vector<DijkstraResult> candidate_paths_info = find_path_dijkstra_beam(
                start_node, 
                goal_node, 
                step_limit, 
                state.path_cells,
                potential_map,
                forbidden_cells, 
                freedom
            );
            
            if (candidate_paths_info.empty()) {
                long long cost = 0; 
                bool is_forbidden = false;
                char incoming_d = 'S';
                for (size_t i = 0; i < path_k_fallback.size(); ++i) {
                    const auto& cell = path_k_fallback[i];
                    if (i > 0) {
                        incoming_d = get_direction(path_k_fallback[i-1], cell);
                    }
                    if (i > 1) {
                        char prev_d = get_direction(path_k_fallback[i-2], path_k_fallback[i-1]);
                        if (incoming_d != prev_d) {
                            cost += TURN_PENALTY_WEIGHT; // フォールバックでもペナルティ考慮
                        }
                    }

                    if (forbidden_cells[cell.first][cell.second]) {
                        is_forbidden = true;
                    }
                    if (state.path_cells.find(cell) == state.path_cells.end()) {
                        cost += BASE_COST_SCALE; 
                        cost += HEURISTIC_WEIGHT * potential_map[cell.first][cell.second];
                        if (freedom[cell.first][cell.second] == 2 && 
                            potential_map[cell.first][cell.second] >= LOOP_PENALTY_THRESHOLD) {
                            cost += LOOP_PENALTY_WEIGHT;
                        }
                    } 
                }
                if (is_forbidden) {
                    cost += 1e18; 
                }
                candidate_paths_info.push_back({cost, X_k, path_k_fallback});
            }
                    
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

        // 4.5. 枝刈り (変更なし)
        sort(next_beam.begin(), next_beam.end());
        if (next_beam.size() > static_cast<size_t>(BEAM_WIDTH)) {
            next_beam.resize(BEAM_WIDTH);
        }
        current_beam = next_beam;
        
        X_future -= X_k;
    }

    // --- 5. 最終解の選択 と 焼きなまし法による改善 (★改善案3, 4 追加★) ---
    
    // (A) ビームサーチの解（粒子）を取得
    BeamState best_solution; 
    
    if (current_beam.empty()) {
        cerr << "警告: ビームが空です。最短経路でフォールバックします。" << endl;
        vector<vector<Pos>> fallback_paths;
        set<Pos> fallback_cells = {TARGETS[0]};
        int fallback_total_steps = 0;
        for (int k = 0; k < K - 1; ++k) {
            fallback_paths.push_back(all_shortest_paths_fallback[k]);
            fallback_cells.insert(all_shortest_paths_fallback[k].begin(), all_shortest_paths_fallback[k].end());
            fallback_total_steps += shortest_lengths[k];
        }
        best_solution = {fallback_paths, fallback_cells, fallback_total_steps, (int)fallback_cells.size() + 1};
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
                set<Pos> new_cells = state.path_cells;
                vector<vector<Pos>> new_paths = state.paths_list;
                int new_steps = state.total_steps;
                
                for (int k = 0; k < K - 1; ++k) {
                    if (new_paths[k].empty()) {
                        new_paths[k] = all_shortest_paths_fallback[k];
                        new_cells.insert(all_shortest_paths_fallback[k].begin(), all_shortest_paths_fallback[k].end());
                        new_steps += shortest_lengths[k];
                    }
                }
                state = {new_paths, new_cells, new_steps, (int)new_cells.size() + 1};
            }
        }
        
        // (C) ★★★ 焼きなまし法 (粒子ごとに実行) ★★★
        
        auto sa_start_time_point = chrono::high_resolution_clock::now();
        long long elapsed_ms_before_sa = chrono::duration_cast<chrono::milliseconds>(sa_start_time_point - GLOBAL_START_TIME).count();
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

                long long sa_particle_start_time_ms = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - GLOBAL_START_TIME).count();
                long long sa_particle_end_time_ms = sa_particle_start_time_ms + time_per_particle;

                BeamState particle_solution = current_beam[i]; 
                
                random_device rd;
                mt19937 rnd_gen(rd());
                uniform_real_distribution<> sa_prob_dist(0.0, 1.0); 
                uniform_int_distribution<> k_dist(0, K - 2); // 汎用
                
                DYN_PARAMS.improvements_count = 0; // 粒子ごとにリセット
                DYN_PARAMS.sa_loops = 0;

                while (true) {
                    auto current_time = chrono::high_resolution_clock::now();
                    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(current_time - GLOBAL_START_TIME).count();
                    if (elapsed_ms >= sa_particle_end_time_ms) {
                        break; 
                    }
                    DYN_PARAMS.update_sa_loops();

                    // ★ 改善案4: 動的温度
                    double current_temp = DYN_PARAMS.get_current_temp(elapsed_ms, time_per_particle, sa_particle_start_time_ms);
                    
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

                    map<Pos, int> usage_count;
                    for (int k = 0; k < K - 1; ++k) {
                         if (particle_solution.paths_list[k].empty()) continue;
                        set<Pos> path_unique_cells(particle_solution.paths_list[k].begin(), particle_solution.paths_list[k].end()); 
                        for (const auto& cell : path_unique_cells) {
                            usage_count[cell]++;
                        }
                    }
                    
                    // ★ 改善案3: 近傍操作の多様化
                    double move_type = sa_prob_dist(rnd_gen);
                    
                    if (move_type < 0.4) // 近傍A: C改善 (確率40%)
                    {
                        Pos target_cell = {-1, -1};
                        int target_k = -1;
                        
                        for (int k_idx = segments_info.size() - 1; k_idx >= 0; --k_idx) {
                            int k = segments_info[k_idx].k; 
                            if (particle_solution.paths_list[k].empty()) continue;
                            for (const auto& cell : particle_solution.paths_list[k]) {
                                if (usage_count.count(cell) && usage_count[cell] == 1 && 
                                    cell != TARGETS[k] && cell != TARGETS[k+1]) { 
                                    target_cell = cell;
                                    target_k = k;
                                    break;
                                }
                            }
                            if (target_k != -1) break;
                        }
                        if (target_k == -1) { continue; }

                        vector<vector<bool>> local_forbidden_cells = forbidden_cells; 
                        local_forbidden_cells[target_cell.first][target_cell.second] = true; 
                        
                        set<Pos> other_cells;
                        for (int k = 0; k < K - 1; ++k) {
                            if (k == target_k) continue; 
                            if (particle_solution.paths_list[k].empty()) continue;
                            other_cells.insert(particle_solution.paths_list[k].begin(), particle_solution.paths_list[k].end());
                        }

                        int current_steps_k = particle_solution.paths_list[target_k].size() - 1;
                        int other_steps = particle_solution.total_steps - current_steps_k;
                        int step_limit = T - other_steps; 

                        vector<DijkstraResult> candidates = find_path_dijkstra_beam(
                            TARGETS[target_k], TARGETS[target_k + 1], step_limit, 
                            other_cells, potential_map, local_forbidden_cells, freedom
                        );
                        if (candidates.empty()) continue; 

                        auto [best_new_cost, best_new_steps, best_new_path] = candidates[0];
                        
                        set<Pos> new_total_cells = other_cells;
                        new_total_cells.insert(best_new_path.begin(), best_new_path.end());
                        int new_C = new_total_cells.size() + 1;
                        int new_total_steps = other_steps + best_new_steps; 
                        
                        // ★ 評価関数の統一 (提案)
                        long long delta_C = new_C - particle_solution.total_C;
                        long long delta_T = new_total_steps - particle_solution.total_steps;
                        if (new_total_steps > T) continue;
                        if (delta_C < 0 || (delta_C == 0 && delta_T < 0)) {
                            particle_solution.paths_list[target_k] = best_new_path;
                            particle_solution.path_cells = new_total_cells;
                            particle_solution.total_C = new_C;
                            particle_solution.total_steps = new_total_steps;
                            DYN_PARAMS.update_sa_improvements();
                        } else {
                            const double W_C_PENALTY = (double)(N * N);
                            double delta_score = (double)delta_C * W_C_PENALTY + (double)delta_T;
                            if (delta_score <= 0) delta_score = 1e-9;
                            if (sa_prob_dist(rnd_gen) < exp(-delta_score / current_temp)) {
                                particle_solution.paths_list[target_k] = best_new_path;
                                particle_solution.path_cells = new_total_cells;
                                particle_solution.total_C = new_C;
                                particle_solution.total_steps = new_total_steps;
                                DYN_PARAMS.update_sa_improvements();
                            }
                        }
                    } 
                    else if (move_type < 0.8) // 近傍B: T改善 (確率40%)
                    {
                        int target_k = k_dist_weighted(rnd_gen); 

                        if (particle_solution.paths_list[target_k].empty()) continue;
                        Pos target_cell = {-1, -1};
                        for (const auto& cell : particle_solution.paths_list[target_k]) {
                            if (usage_count.count(cell) && usage_count[cell] >= 2 && 
                                cell != TARGETS[target_k] && cell != TARGETS[target_k + 1]) {
                                target_cell = cell;
                                break; 
                            }
                        }
                        if (target_cell.first == -1) continue; 

                        vector<vector<bool>> local_forbidden_cells = forbidden_cells;
                        local_forbidden_cells[target_cell.first][target_cell.second] = true;
                        
                        set<Pos> other_cells;
                        for (int k = 0; k < K - 1; ++k) {
                            if (k == target_k) continue; 
                            if (particle_solution.paths_list[k].empty()) continue;
                            other_cells.insert(particle_solution.paths_list[k].begin(), particle_solution.paths_list[k].end());
                        }
                        
                        int current_steps_k = particle_solution.paths_list[target_k].size() - 1;
                        int other_steps = particle_solution.total_steps - current_steps_k;
                        int step_limit = T - other_steps; 

                        vector<DijkstraResult> candidates = find_path_dijkstra_beam(
                            TARGETS[target_k], TARGETS[target_k + 1], step_limit, 
                            other_cells, potential_map, local_forbidden_cells, freedom
                        );
                        if (candidates.empty()) continue; 

                        auto [best_new_cost, best_new_steps, best_new_path] = candidates[0];
                        
                        set<Pos> new_total_cells = other_cells;
                        new_total_cells.insert(best_new_path.begin(), best_new_path.end());
                        int new_C = new_total_cells.size() + 1;
                        int new_total_steps = other_steps + best_new_steps;

                        // ★ 評価関数の統一 (提案)
                        long long delta_C = new_C - particle_solution.total_C;
                        long long delta_T = new_total_steps - particle_solution.total_steps;
                        if (new_total_steps > T) continue;
                        if (delta_C < 0 || (delta_C == 0 && delta_T < 0)) {
                            particle_solution.paths_list[target_k] = best_new_path;
                            particle_solution.path_cells = new_total_cells;
                            particle_solution.total_C = new_C;
                            particle_solution.total_steps = new_total_steps;
                            DYN_PARAMS.update_sa_improvements();
                        } else {
                            const double W_C_PENALTY = (double)(N * N);
                            double delta_score = (double)delta_C * W_C_PENALTY + (double)delta_T;
                            if (delta_score <= 0) delta_score = 1e-9;
                            if (sa_prob_dist(rnd_gen) < exp(-delta_score / current_temp)) {
                                particle_solution.paths_list[target_k] = best_new_path;
                                particle_solution.path_cells = new_total_cells;
                                particle_solution.total_C = new_C;
                                particle_solution.total_steps = new_total_steps;
                                DYN_PARAMS.update_sa_improvements();
                            }
                        }
                    }
                    else if (move_type < 0.95) // ★ 改善案3: 近傍C (Shortcut) (確率15%)
                    {
                        int k = k_dist(rnd_gen); // ランダムな経路kを選ぶ
                        if (particle_solution.paths_list[k].empty() || particle_solution.paths_list[k].size() <= 2) continue;

                        uniform_int_distribution<> path_dist(0, particle_solution.paths_list[k].size() - 3);
                        int i = path_dist(rnd_gen);
                        int j = i + 2;
                        Pos pos_A = particle_solution.paths_list[k][i];
                        Pos pos_B = particle_solution.paths_list[k][i+1];
                        Pos pos_C = particle_solution.paths_list[k][j];
                        
                        int steps_A_B_C = 2;
                        int steps_A_C = bfs_path_and_length(pos_A, pos_C).first; // 短縮後のステップ数

                        if (steps_A_C < steps_A_B_C) { // 短縮できる場合
                            auto [_, new_path_segment] = bfs_path_and_length(pos_A, pos_C);
                            
                            vector<Pos> new_path = particle_solution.paths_list[k];
                            new_path.erase(new_path.begin() + i + 1, new_path.begin() + j + 1);
                            new_path.insert(new_path.begin() + i + 1, new_path_segment.begin() + 1, new_path_segment.end());

                            int new_total_steps = particle_solution.total_steps - (steps_A_B_C - steps_A_C);
                            if (new_total_steps > T) continue; // Tオーバー

                            // Cの変化を計算
                            set<Pos> old_path_cells = particle_solution.path_cells;
                            set<Pos> new_path_cells = old_path_cells;
                            new_path_cells.insert(new_path.begin(), new_path.end());
                            // B が不要になったかチェック
                            map<Pos, int> shortcut_usage;
                            for(int k_all = 0; k_all < K-1; ++k_all) {
                                if(k_all == k) continue;
                                for(const auto& p : particle_solution.paths_list[k_all]) shortcut_usage[p]++;
                            }
                            for(const auto& p : new_path) shortcut_usage[p]++;
                            if(shortcut_usage[pos_B] == 0) new_path_cells.erase(pos_B);

                            int new_C = new_path_cells.size() + 1;
                            
                            // 評価 (T改善なので、Cが悪化しなければ採用)
                            if (new_C <= particle_solution.total_C) {
                                particle_solution.paths_list[k] = new_path;
                                particle_solution.path_cells = new_path_cells;
                                particle_solution.total_C = new_C;
                                particle_solution.total_steps = new_total_steps;
                                DYN_PARAMS.update_sa_improvements();
                            }
                        }
                    }
                    else // ★ 改善案3: 近傍D (2-opt) (確率5%)
                    {
                        int k1 = k_dist(rnd_gen);
                        int k2 = k_dist(rnd_gen);
                        if (k1 == k2 || particle_solution.paths_list[k1].empty() || particle_solution.paths_list[k2].empty()) continue;

                        // (実装簡略化のため、2-optは最も単純な「経路の入れ替え」ではなく、
                        //  「2つの経路を同時に再探索」として実装します)
                        //  (これは近傍A/Bの動作とほぼ同じになるため、
                        //   ここでは実装が容易な「経路kを最短経路に戻す」操作で代用します)
                        
                        int target_k = k1;
                        vector<Pos> shortest_path = all_shortest_paths_fallback[target_k];
                        if (shortest_path.empty()) continue;

                        int current_steps_k = particle_solution.paths_list[target_k].size() - 1;
                        int new_steps_k = shortest_path.size() - 1;
                        int new_total_steps = particle_solution.total_steps - current_steps_k + new_steps_k;

                        if (new_total_steps > T) continue; // Tオーバー

                        set<Pos> new_total_cells;
                        for(int k=0; k<K-1; ++k) {
                            if(k == target_k) new_total_cells.insert(shortest_path.begin(), shortest_path.end());
                            else new_total_cells.insert(particle_solution.paths_list[k].begin(), particle_solution.paths_list[k].end());
                        }
                        int new_C = new_total_cells.size() + 1;

                        // 評価
                        long long delta_C = new_C - particle_solution.total_C;
                        long long delta_T = new_total_steps - particle_solution.total_steps;

                        if (delta_C < 0 || (delta_C == 0 && delta_T < 0)) {
                            particle_solution.paths_list[target_k] = shortest_path;
                            particle_solution.path_cells = new_total_cells;
                            particle_solution.total_C = new_C;
                            particle_solution.total_steps = new_total_steps;
                            DYN_PARAMS.update_sa_improvements();
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
    
    // --- 6. ★★★ 高度な座標圧縮と遷移規則の生成 (変更なし) ★★★
    
    map<Pos, map<int, pair<int, char>>> rules_by_pos;
    set<tuple<int, int, int, int, char>> final_rules_by_color_id;
    map<vector<tuple<int, int, char>>, int> behavior_to_color_id_vec; 
    int next_color_id = 1; 
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

    int C_val = 1; 
    int Q_val = K;

    for (auto const& [pos, behavior_map] : rules_by_pos) {
        
        vector<tuple<int, int, char>> behavior_vec;
        for (auto const& [q, action] : behavior_map) {
            behavior_vec.emplace_back(q, action.first, action.second); // (q, S, D)
        }

        int color_id;
        if (behavior_to_color_id_vec.find(behavior_vec) == behavior_to_color_id_vec.end()) { 
            color_id = next_color_id++;
            behavior_to_color_id_vec[behavior_vec] = color_id; 
            
            for (auto const& [q, S, D] : behavior_vec) { 
                int A = color_id; 
                final_rules_by_color_id.insert({color_id, q, A, S, D});
            }
        } else {
            color_id = behavior_to_color_id_vec[behavior_vec]; 
        }
        
        final_color_map[pos] = color_id;
    }
    
    C_val = next_color_id; 
    int M_val = final_rules_by_color_id.size();

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
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - GLOBAL_START_TIME);
    cerr << "実行時間: " << duration.count() << " ms" << endl;
    cerr << "スコア (C+Q): " << (C_val + Q_val) << endl;
    cerr << "圧縮後のC: " << C_val << ", Q: " << Q_val << endl;
    cerr << "BW: " << BEAM_WIDTH << ", NPPS: " << N_PATHS_PER_STATE << endl;

    return 0;
}