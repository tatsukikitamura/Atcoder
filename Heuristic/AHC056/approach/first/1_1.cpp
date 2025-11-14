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
#include <cmath>   // ★ exp() のために追加
#include <random>  // ★ 乱数生成のために追加

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

const long long BASE_COST_SCALE = 1000;
const long long HEURISTIC_WEIGHT = 1;
const long long LOOP_PENALTY_WEIGHT = 1500; 
const int LOOP_PENALTY_THRESHOLD = 5; 

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


int main() {
    auto start_time = chrono::high_resolution_clock::now(); // ★ 時間計測開始
    
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> N >> K >> T;
    V_WALLS.resize(N);
    H_WALLS.resize(N - 1);
    TARGETS.resize(K);
    for (int i = 0; i < N; ++i) cin >> V_WALLS[i];
    for (int i = 0; i < N - 1; ++i) cin >> H_WALLS[i];
    for (int i = 0; i < K; ++i) cin >> TARGETS[i].first >> TARGETS[i].second;

    // --- ★ パラメータ調整 (ユーザー定義) ---
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
    // --- ★ パラメータ調整ここまで ---


    // --- ★ 4.0. 潜在価値マップの計算 (変更なし) ---
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

    // --- ★★★ 4.1. 禁止マス(袋小路の奥)の計算 (変更なし) ★★★ ---
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

    // 4.4. ビームサーチの実行
    vector<BeamState> current_beam;
    set<Pos> initial_cells = {TARGETS[0]};
    int initial_C = initial_cells.size() + 1; 
    current_beam.push_back({vector<vector<Pos>>(K - 1), initial_cells, 0, initial_C});

    double X_future = total_shortest_steps_X;

    for (const auto& segment : segments_info) {
        // ★ 焼きなましに時間を残すため、ビームサーチは早めに打ち切る
        auto current_time_beam = chrono::high_resolution_clock::now();
        auto elapsed_ms_beam = chrono::duration_cast<chrono::milliseconds>(current_time_beam - start_time).count();
        if (elapsed_ms_beam >= 1800) { // ★ 1.8秒でビームサーチを打ち切り
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
                state.path_cells,
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

    // --- 5. 最終解の選択 と 焼きなまし法による改善 ★★★
    
    // (A) まずビームサーチの最良解を取得
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
        best_solution = current_beam[0];
    }
    
    // (B) 不完全なら補完
    bool is_incomplete = false;
    for (int k = 0; k < K - 1; ++k) {
        if (best_solution.paths_list[k].empty()) {
            is_incomplete = true;
            break;
        }
    }
    
    if (is_incomplete) {
        cerr << "警告: 途中の解が不完全です。最短経路で補完します。" << endl;
        set<Pos> fallback_cells = best_solution.path_cells;
        vector<vector<Pos>> fallback_paths = best_solution.paths_list;
        int fallback_total_steps = best_solution.total_steps;
        
        for (int k = 0; k < K - 1; ++k) {
            if (fallback_paths[k].empty()) {
                fallback_paths[k] = all_shortest_paths_fallback[k];
                fallback_cells.insert(all_shortest_paths_fallback[k].begin(), all_shortest_paths_fallback[k].end());
                fallback_total_steps += shortest_lengths[k];
            }
        }
        best_solution = {fallback_paths, fallback_cells, fallback_total_steps, (int)fallback_cells.size() + 1};
    }

    // (C) ★★★ 焼きなまし法 (Simulated Annealing) による改善 ★★★
    const long long TIME_LIMIT_MS = 1970; // 全体の制限時間
    // ★ 温度設定 (C_val の変化は 1 なので、1.0 付近から始めるのが標準的)
    const double START_TEMP = 1.5; // ★ 要調整
    const double END_TEMP = 0.01;  // ★ 要調整

    // ★ 乱数生成器
    random_device rd;
    mt19937 rnd_gen(rd());
    uniform_real_distribution<> rnd_dist(0.0, 1.0);

    // 改善不可能なセルを記録（簡易的なタブーリスト）
    set<pair<int, Pos>> tried_improve; 

    // ★ 時間ベースのループに変更
    while (true) {
        auto current_time = chrono::high_resolution_clock::now();
        auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(current_time - start_time).count();
        if (elapsed_ms >= TIME_LIMIT_MS) {
            cerr << "焼きなまし時間切れ: " << elapsed_ms << " ms" << endl;
            break;
        }

        // ★ 温度計算 (線形補間)
        double time_ratio = min(1.0, (double)elapsed_ms / TIME_LIMIT_MS);
        double current_temp = START_TEMP + (END_TEMP - START_TEMP) * time_ratio;
        
        // 1. 全マスの使用回数を集計 (変更なし)
        map<Pos, int> usage_count;
        for (int k = 0; k < K - 1; ++k) {
             if (best_solution.paths_list[k].empty()) continue;
            set<Pos> path_unique_cells(best_solution.paths_list[k].begin(), best_solution.paths_list[k].end()); 
            for (const auto& cell : path_unique_cells) {
                usage_count[cell]++;
            }
        }
        
        // 2. 「1回だけ」使われているマスを探す (変更なし)
        Pos target_cell = {-1, -1};
        int target_k = -1;
        
        for (int k_idx = segments_info.size() - 1; k_idx >= 0; --k_idx) {
            int k = segments_info[k_idx].k; 
            if (best_solution.paths_list[k].empty()) continue;
            
            for (const auto& cell : best_solution.paths_list[k]) {
                if (usage_count[cell] == 1 && 
                    cell != TARGETS[k] && cell != TARGETS[k+1] &&
                    tried_improve.find({k, cell}) == tried_improve.end()) { 
                    
                    target_cell = cell;
                    target_k = k;
                    break;
                }
            }
            if (target_k != -1) break;
        }

        if (target_k == -1) {
            // 改善対象（1回マス）がタブー以外にない
            // タブーリストをリセットして、もう一度最初から探し直す
            tried_improve.clear();
            continue; 
        }
        
        // この改善（k, cell）を試行したと記録
        tried_improve.insert({target_k, target_cell});

        // 3. 代替ルートの探索 (変更なし)
        Pos start_node = TARGETS[target_k];
        Pos goal_node = TARGETS[target_k + 1];
        
        vector<vector<bool>> local_forbidden_cells = forbidden_cells; 
        local_forbidden_cells[target_cell.first][target_cell.second] = true; 

        set<Pos> other_cells;
        for (int k = 0; k < K - 1; ++k) {
            if (k == target_k) continue; 
            if (best_solution.paths_list[k].empty()) continue;
            other_cells.insert(best_solution.paths_list[k].begin(), best_solution.paths_list[k].end());
        }
        
        int current_steps_k = best_solution.paths_list[target_k].size() - 1;
        int other_steps = best_solution.total_steps - current_steps_k;
        int step_limit = T - other_steps; 

        vector<DijkstraResult> candidates = find_path_dijkstra_beam(
            start_node, 
            goal_node, 
            step_limit, 
            other_cells, 
            potential_map,
            local_forbidden_cells,
            freedom 
        );

        if (candidates.empty()) {
            continue; // 代替ルートが見つからなかった
        }

        // 4. 解の評価と遷移 (★ 焼きなまし法のロジック)
        auto [best_new_cost, best_new_steps, best_new_path] = candidates[0];
        
        set<Pos> new_total_cells = other_cells;
        new_total_cells.insert(best_new_path.begin(), best_new_path.end());
        int new_C = new_total_cells.size() + 1;
        int new_total_steps = other_steps + best_new_steps; // ★ 新しいステップ数

        // ★ スコア差分 (Delta) を計算。Cを優先
        int delta_score = (new_C - best_solution.total_C);

        // ★ TLEペナルティ: Tを超える解は受け入れない (delta_scoreを巨大にする)
        if (new_total_steps > T) {
            delta_score += 1e9; // 事実上の拒否
        }

        if (delta_score < 0) {
            // ★★★ 改善成功 (常に受け入れ) ★★★
            cerr << "SA 改善: k=" << target_k << " (cell: " << target_cell.first << "," << target_cell.second << ") を更新。 C: " << best_solution.total_C << " -> " << new_C << endl;
            best_solution.paths_list[target_k] = best_new_path;
            best_solution.path_cells = new_total_cells;
            best_solution.total_C = new_C;
            best_solution.total_steps = new_total_steps;
            tried_improve.clear(); // 状態が変化したのでタブーリセット
        } else {
            // ★★★ 悪化または維持 (確率で受け入れ) ★★★
            double probability = exp(-(double)delta_score / current_temp);
            if (rnd_dist(rnd_gen) < probability) {
                cerr << "SA 悪化遷移: k=" << target_k << " (cell: " << target_cell.first << "," << target_cell.second << ") を更新。 C: " << best_solution.total_C << " -> " << new_C << " (Prob: " << probability << ")" << endl;
                best_solution.paths_list[target_k] = best_new_path;
                best_solution.path_cells = new_total_cells;
                best_solution.total_C = new_C;
                best_solution.total_steps = new_total_steps;
                tried_improve.clear(); // 状態が変化したのでタブーリセット
            }
            // else: 受け入れない (タブーリストには入ったまま)
        }
    }
    // ★★★ 焼きなましここまで ★★★


    int C_val = best_solution.total_C;
    int Q_val = K;

    // --- 6. 色の割り当てと遷移規則の生成 (変更なし) ---
    map<Pos, int> color_map;
    int current_color = 1; 
    for (const auto& cell : best_solution.path_cells) {
        if (color_map.find(cell) == color_map.end()) {
            color_map[cell] = current_color++;
        }
    }

    vector<vector<int>> initial_board(N, vector<int>(N));
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
    cerr << "スコア (C+Q): " << (C_val + Q_val) << endl;
    cerr << "BW: " << BEAM_WIDTH << ", NPPS: " << N_PATHS_PER_STATE << endl;

    return 0;
}