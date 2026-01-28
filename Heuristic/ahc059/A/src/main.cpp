#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
// #pragma GCC target("avx2")  // x86のみ。AtCoder提出時は有効化
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <cmath>
#include <chrono>
#include <random>
#include <set>
#include <bitset>
using namespace std;


// ===== Constants =====
constexpr int N = 20;
constexpr int NN = N * N;           // 400
constexpr int NUM_CARDS = NN / 2;   // 200 (カード数)
constexpr int INF = 1e9;

// ===== Timer =====
struct Timer {
    chrono::high_resolution_clock::time_point start;
    double limit;
    
    Timer(double limit_sec = 1.98) : limit(limit_sec) {
        start = chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start).count();
    }
    
    bool is_over() const {
        return elapsed() >= limit;
    }
};

// ===== Random =====
struct XorShift {
    unsigned int x = 123456789;
    unsigned int y = 362436069;
    unsigned int z = 521288629;
    unsigned int w;
    
    XorShift(unsigned int seed = 42) : w(seed) {}
    
    inline unsigned int next() {
        unsigned int t = x ^ (x << 11);
        x = y; y = z; z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }
    
    inline int nextInt(int n) {
        return next() % n;
    }
    
    inline double nextDouble() {
        return next() * 2.3283064365386963e-10; // 1 / 2^32
    }
    
    inline unsigned int operator()() {
        return next();
    }
};

// ===== Position =====
struct Pos {
    int r, c;
    Pos() : r(0), c(0) {}
    Pos(int r, int c) : r(r), c(c) {}
    bool operator==(const Pos& o) const { return r == o.r && c == o.c; }
    bool operator!=(const Pos& o) const { return !(*this == o); }
};

// ===== Input =====
struct Input {
    int board[N][N];
    pair<Pos, Pos> card_pos[NUM_CARDS];
    
    void read() {
        int n;
        if (!(cin >> n)) return;
        vector<bool> first_seen(NUM_CARDS, true);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cin >> board[i][j];
                int card = board[i][j];
                if (card < 0 || card >= NUM_CARDS) {
                    cerr << "[ERROR] Invalid card value at (" << i << "," << j << "): " << card << endl;
                    exit(1);
                }
                if (first_seen[card]) {
                    card_pos[card].first = Pos(i, j);
                    first_seen[card] = false;
                } else {
                    card_pos[card].second = Pos(i, j);
                }
            }
        }
    }
};



int manhattan_dist(const Pos& a, const Pos& b) {
    return abs(a.r - b.r) + abs(a.c - b.c);
}

void append_path(string& path, int& move_count, const Pos& from, const Pos& to) {
    int dr = to.r - from.r;
    int dc = to.c - from.c;
    char r_char = (dr > 0) ? 'D' : 'U';
    char c_char = (dc > 0) ? 'R' : 'L';
    dr = abs(dr);
    dc = abs(dc);
    path.append(dr, r_char);
    path.append(dc, c_char);
    move_count += dr + dc;
}

// ===== Solution Structure =====
// 解の構造を保存（局所探索で改良するため）
struct Solution {
    // groups[i] = i番目のグループ（カードIDとvisit_firstのペアのリスト）
    vector<vector<pair<int, bool>>> groups;
    // キャッシュ: 各グループの終了位置とコスト
    vector<Pos> end_positions;   // end_positions[i] = groups[i]の後の位置
    vector<int> group_costs;     // group_costs[i] = groups[i]のコスト
    int cost;
    
    Solution() : cost(INF) {}
};

// ===== Solver =====
class Solver {
public:
    Input input;
    Timer timer;
    
    Solution best_solution;
    string best_operations;
    int best_move_count;
    
    vector<bool> used;
    XorShift rng;
    
    struct Candidate {
        int card;
        bool visit_first;
        int cost;
    };
    vector<Candidate> candidates;
    
    Solver() : best_move_count(INF), rng(42) {
        used.resize(NUM_CARDS);
        candidates.reserve(NN);
        // BFS用変数の初期化
        for(int i=0; i<N; i++) for(int j=0; j<N; j++) bfs_visit[i][j] = 0;
        fill(card_seen, card_seen + NUM_CARDS, 0);
    }
    
    void solve() {
        input.read();
        
        // ===== Phase 1: 探索フェーズ =====
        double phase1_time = 1.5;  // 1.0秒（探索を多めに）
        int iteration = 0;
        
        while (true) {
            if (iteration % 100 == 0 && timer.elapsed() >= phase1_time) break;
            bool deterministic = (iteration == 0);
            Solution sol = build_solution(deterministic);
            
            if (sol.cost < best_solution.cost) {
                best_solution = sol;
                cerr << "[Phase1] New Best: " << sol.cost << " (Iter " << iteration + 1 << ")" << endl;
            }
            iteration++;
        }
        cerr << "[Phase1] Done: " << iteration << " iterations, best=" << best_solution.cost << endl;
        
        // ===== Phase 2: 改良フェーズ (SA) =====
        int ls_iterations = 0;
        int improvements = 0;
        int accepts = 0;
        
        // SAパラメータ（チューニング済み）
        double t0 = 1.0;    // 初期温度
        double t1 = 0.01;   // 最終温度
        double phase2_start = timer.elapsed();
        double phase2_end = 1.95;
        
        // 最良解を保存
        Solution current_solution = best_solution;
        
        double elapsed = phase2_start;
        
        while (true) {
            if (ls_iterations % 100 == 0) {
                elapsed = timer.elapsed();
                if (elapsed >= phase2_end) break;
            }
            // 温度計算
            double progress = (elapsed - phase2_start) / (phase2_end - phase2_start);
            progress = min(1.0, max(0.0, progress));
            double temp = t0 * pow(t1 / t0, progress);
            
            bool accepted = sa_step(current_solution, temp);
            if (accepted) {
                accepts++;
                if (current_solution.cost < best_solution.cost) {
                    best_solution = current_solution;
                    improvements++;
                    cerr << "[Phase2/SA] New Best: " << best_solution.cost 
                         << " (Iter " << ls_iterations << ", T=" << temp << ")" << endl;
                }
            }
            ls_iterations++;
        }
        cerr << "[Phase2] Done: " << ls_iterations << " iterations, " 
             << improvements << " improvements, " << accepts << " accepts" << endl;
        
        // 最終解を操作列に変換
        convert_to_operations(best_solution);
        output();
    }
    
    // BFS用変数
    int bfs_visit[N][N];
    int bfs_token = 0;
    int card_seen[NUM_CARDS];
    int card_token = 0;

    Solution build_solution(bool deterministic) {
        Solution sol;
        sol.groups.clear();
        sol.cost = 0;
        
        fill(used.begin(), used.end(), false);
        Pos current_pos(0, 0);
        int remaining = NUM_CARDS;
        
        // BFS用キューを確保（再利用）
        static queue<pair<Pos, int>> q; 

        while (remaining > 0) {
            candidates.clear();
            
            // BFSで近い順に候補を探す (上位4件 + α 見つかるまで)
            while (!q.empty()) q.pop();
            q.push({current_pos, 0});
            
            bfs_token++;
            card_token++;
            
            bfs_visit[current_pos.r][current_pos.c] = bfs_token;
            
            int K = 4;
            int found_count = 0;
            
            // 4方向の移動ベクトル
            const int dr[] = {0, 0, 1, -1};
            const int dc[] = {1, -1, 0, 0};

            while (!q.empty()) {
                auto [curr, dist] = q.front();
                q.pop();
                
                // このマスのカードをチェック
                int card = input.board[curr.r][curr.c];
                if (!used[card] && card_seen[card] != card_token) {
                    card_seen[card] = card_token;
                    
                    // first/second どちらが近いかは、今出会ったのがどっちかで判定
                    bool visit_first = (curr == input.card_pos[card].first);
                    candidates.push_back({card, visit_first, dist});
                    found_count++;
                }

                if (found_count >= K) break; // 十分な数が見つかったら終了

                // 隣接マスへ
                for (int i = 0; i < 4; i++) {
                    int nr = curr.r + dr[i];
                    int nc = curr.c + dc[i];
                    
                    if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                        if (bfs_visit[nr][nc] != bfs_token) {
                            bfs_visit[nr][nc] = bfs_token;
                            q.push({Pos(nr, nc), dist + 1});
                        }
                    }
                }
            }
            
            if (candidates.empty()) break;
            
            // BFSなので既に距離順に並んでいる -> partial_sort不要
            
            Candidate best = candidates[0];
            if (!deterministic) {
                int range = min((int)candidates.size(), K);
                double T = 1.5;
                // スタック上に確保して動的確保を回避
                double weights[4]; 
                double sum_weights = 0.0;
                int min_cost = candidates[0].cost;
                for (int i = 0; i < range; i++) {
                    weights[i] = exp(-(candidates[i].cost - min_cost) / T);
                    sum_weights += weights[i];
                }
                double r = rng.nextDouble() * sum_weights;
                double cumsum = 0.0;
                int pick_idx = 0;
                for (int i = 0; i < range; i++) {
                    cumsum += weights[i];
                    if (r <= cumsum) {
                        pick_idx = i;
                        break;
                    }
                }
                best = candidates[pick_idx];
            }
            
            // グループを構築
            vector<pair<int, bool>> group = find_proximity_group(best.card, best.visit_first, used, 200);
            optimize_group_order(current_pos, group);
            
            // グループのコストを加算
            sol.cost += calc_group_cost(current_pos, group);
            
            // 終了位置を更新
            int last_card = group[0].first;
            bool last_visit_first = group[0].second;
            current_pos = last_visit_first ? input.card_pos[last_card].second : input.card_pos[last_card].first;
            
            // 使用済みマーク
            for (auto& p : group) {
                if (!used[p.first]) {
                    used[p.first] = true;
                    remaining--;
                }
            }
            

            
            sol.groups.push_back(group);
        }
        
        // キャッシュを構築
        rebuild_cache(sol);
        
        return sol;
    }
    
    // キャッシュを再構築
    void rebuild_cache(Solution& sol) {
        int n = sol.groups.size();
        sol.end_positions.resize(n);
        sol.group_costs.resize(n);
        sol.cost = 0;
        
        Pos current_pos(0, 0);
        for (int i = 0; i < n; i++) {
            sol.group_costs[i] = calc_group_cost(current_pos, sol.groups[i]);
            sol.cost += sol.group_costs[i];
            
            // 終了位置を計算
            int last_card = sol.groups[i][0].first;
            bool last_visit_first = sol.groups[i][0].second;
            current_pos = last_visit_first ? input.card_pos[last_card].second : input.card_pos[last_card].first;
            sol.end_positions[i] = current_pos;
        }
    }
    
    // 範囲[from, to)のコストを再計算して差分を返す
    int recalculate_range(Solution& sol, int from, int to) {
        int old_cost = 0;
        int new_cost = 0;
        
        for (int i = from; i < to; i++) {
            old_cost += sol.group_costs[i];
        }
        
        Pos current_pos = (from == 0) ? Pos(0, 0) : sol.end_positions[from - 1];
        
        for (int i = from; i < to; i++) {
            int gc = calc_group_cost(current_pos, sol.groups[i]);
            new_cost += gc;
            
            // 終了位置を更新
            int last_card = sol.groups[i][0].first;
            bool last_visit_first = sol.groups[i][0].second;
            current_pos = last_visit_first ? input.card_pos[last_card].second : input.card_pos[last_card].first;
        }
        
        return new_cost - old_cost;
    }
    
    // キャッシュを範囲[from, to)で更新
    void update_cache_range(Solution& sol, int from, int to) {
        Pos current_pos = (from == 0) ? Pos(0, 0) : sol.end_positions[from - 1];
        
        for (int i = from; i < to; i++) {
            sol.group_costs[i] = calc_group_cost(current_pos, sol.groups[i]);
            
            int last_card = sol.groups[i][0].first;
            bool last_visit_first = sol.groups[i][0].second;
            current_pos = last_visit_first ? input.card_pos[last_card].second : input.card_pos[last_card].first;
            sol.end_positions[i] = current_pos;
        }
    }
    
    // ===== SAの1ステップ（差分計算版） =====
    bool sa_step(Solution& sol, double temp) {
        if (sol.groups.size() < 2) return false;
        
        int n = sol.groups.size();
        
        // ランダムに操作を選択（6種類）
        int op = rng.nextInt(6);
        
        if (op == 0) {
            // 操作1: 隣接グループを入れ替え
            int i = rng.nextInt(n - 1);
            swap(sol.groups[i], sol.groups[i + 1]);
            
            int delta = recalculate_range(sol, i, n);
            // SA判定: delta < 0 なら必ず採用、delta >= 0 でも確率的に採用
            if (delta < 0 || rng.nextDouble() < exp(-delta / temp)) {
                sol.cost += delta;
                update_cache_range(sol, i, n);
                return true;
            } else {
                swap(sol.groups[i], sol.groups[i + 1]);
                return false;
            }
        } else if (op == 1) {
            // 操作2: 2つのグループを入れ替え
            int i = rng.nextInt(n);
            int j = rng.nextInt(n);
            if (i == j) return false;
            if (i > j) swap(i, j);
            
            swap(sol.groups[i], sol.groups[j]);
            
            // SA判定
            int delta = recalculate_range(sol, i, n);
            if (delta < 0 || rng.nextDouble() < exp(-delta / temp)) {
                sol.cost += delta;
                update_cache_range(sol, i, n);
                return true;
            } else {
                swap(sol.groups[i], sol.groups[j]);
                return false;
            }
        } else if (op == 2) {
            // 操作3: グループ内の訪問順序を反転
            int i = rng.nextInt(n);
            if (sol.groups[i].size() < 2) return false;
            
            for (auto& p : sol.groups[i]) {
                p.second = !p.second;
            }
            reverse(sol.groups[i].begin(), sol.groups[i].end());
            
            // SA判定
            int delta = recalculate_range(sol, i, n);
            if (delta < 0 || rng.nextDouble() < exp(-delta / temp)) {
                sol.cost += delta;
                update_cache_range(sol, i, n);
                return true;
            } else {
                reverse(sol.groups[i].begin(), sol.groups[i].end());
                for (auto& p : sol.groups[i]) {
                    p.second = !p.second;
                }
                return false;
            }
        } else if (op == 3) {
            // 操作4: カードを別グループに移動
            int from_g = rng.nextInt(n);
            int to_g = rng.nextInt(n);
            if (from_g == to_g) return false;
            if (sol.groups[from_g].size() <= 1) return false;
            
            int card_idx = rng.nextInt(sol.groups[from_g].size());
            auto card = sol.groups[from_g][card_idx];
            
            sol.groups[from_g].erase(sol.groups[from_g].begin() + card_idx);
            sol.groups[to_g].push_back(card);
            
            // SA判定
            int start = min(from_g, to_g);
            int delta = recalculate_range(sol, start, n);
            if (delta < 0 || rng.nextDouble() < exp(-delta / temp)) {
                sol.cost += delta;
                update_cache_range(sol, start, n);
                return true;
            } else {
                sol.groups[to_g].pop_back();
                sol.groups[from_g].insert(sol.groups[from_g].begin() + card_idx, card);
                return false;
            }
        } else if (op == 4) {
            // 操作5: 単一カードの向きを反転
            int g = rng.nextInt(n);
            int card_idx = rng.nextInt(sol.groups[g].size());
            
            sol.groups[g][card_idx].second = !sol.groups[g][card_idx].second;
            
            // SA判定
            int delta = recalculate_range(sol, g, n);
            if (delta < 0 || rng.nextDouble() < exp(-delta / temp)) {
                sol.cost += delta;
                update_cache_range(sol, g, n);
                return true;
            } else {
                sol.groups[g][card_idx].second = !sol.groups[g][card_idx].second;
                return false;
            }
        } else {
            // 操作6: 2-opt（グループ順序の区間反転）
            int i = rng.nextInt(n);
            int j = rng.nextInt(n);
            if (i == j) return false;
            if (i > j) swap(i, j);
            if (j - i < 2) return false;  // 最低2つ以上の区間
            
            // 区間[i, j]を反転
            reverse(sol.groups.begin() + i, sol.groups.begin() + j + 1);
            
            // SA判定
            int delta = recalculate_range(sol, i, n);
            if (delta < 0 || rng.nextDouble() < exp(-delta / temp)) {
                sol.cost += delta;
                update_cache_range(sol, i, n);
                return true;
            } else {
                // 元に戻す
                reverse(sol.groups.begin() + i, sol.groups.begin() + j + 1);
                return false;
            }
        }
    }
    
    // ===== 解のコストを再計算 =====
    int recalculate_cost(const Solution& sol) {
        int cost = 0;
        Pos current_pos(0, 0);
        
        for (const auto& group : sol.groups) {
            cost += calc_group_cost(current_pos, group);
            
            // 終了位置を更新
            int last_card = group[0].first;
            bool last_visit_first = group[0].second;
            current_pos = last_visit_first ? input.card_pos[last_card].second : input.card_pos[last_card].first;
        }
        
        return cost;
    }
    
    // ===== 解を操作列に変換 =====
    void convert_to_operations(const Solution& sol) {
        best_operations.clear();
        best_move_count = 0;
        Pos current_pos(0, 0);
        
        for (const auto& group : sol.groups) {
            vector<int> cards;
            vector<bool> visit_first;
            for (const auto& p : group) {
                cards.push_back(p.first);
                visit_first.push_back(p.second);
            }
            
            // 往路
            for (size_t i = 0; i < cards.size(); ++i) {
                int card = cards[i];
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                Pos target = visit_first[i] ? p1 : p2;
                
                append_path(best_operations, best_move_count, current_pos, target);
                current_pos = target;
                best_operations += 'Z';
            }
            
            // 復路
            for (int i = (int)cards.size() - 1; i >= 0; --i) {
                int card = cards[i];
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                Pos target = visit_first[i] ? p2 : p1;
                
                append_path(best_operations, best_move_count, current_pos, target);
                current_pos = target;
                best_operations += 'Z';
            }
        }
    }
    
    // ===== ヘルパー関数（既存） =====

    
    vector<pair<int, bool>> find_proximity_group(int main_card, bool main_visit_first, 
                                     const vector<bool>& used, int max_depth = 3) {
        vector<pair<int, bool>> group;
        group.push_back({main_card, main_visit_first});
        
        bitset<NUM_CARDS> in_group;
        in_group.set(main_card);
        
        Pos current_start = main_visit_first ? input.card_pos[main_card].first : input.card_pos[main_card].second;
        Pos current_end = main_visit_first ? input.card_pos[main_card].second : input.card_pos[main_card].first;
        
        for (int depth = 1; depth < max_depth; depth++) {
            int best_inner = -1;
            int min_insertion_cost = INF;
            bool best_inner_visit_first = true;
            
            // ループ内で不変な値を事前に計算
            // calc_insertion_cost = (dist(S, p_in) + dist(p_in, p_out) + dist(p_out, E)) - (dist(S, E) + dist(p_in, p_out))
            //                     = dist(S, p_in) + dist(p_out, E) - dist(S, E)
            int base_dist = manhattan_dist(current_start, current_end);

            for (int card = 0; card < NUM_CARDS; card++) {
                if (used[card]) continue;
                if (in_group[card]) continue;
                
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                
                // cost1: enter p1, exit p2
                int cost1 = manhattan_dist(current_start, p1) + manhattan_dist(p2, current_end) - base_dist;
                // cost2: enter p2, exit p1
                int cost2 = manhattan_dist(current_start, p2) + manhattan_dist(p1, current_end) - base_dist;
                
                if (cost1 < min_insertion_cost) {
                    min_insertion_cost = cost1;
                    best_inner = card;
                    best_inner_visit_first = true;
                }
                if (cost2 < min_insertion_cost) {
                    min_insertion_cost = cost2;
                    best_inner = card;
                    best_inner_visit_first = false;
                }
            }
            
            if (best_inner != -1 && min_insertion_cost <= 1) {
                group.push_back({best_inner, best_inner_visit_first});
                in_group.set(best_inner);
                Pos p1 = input.card_pos[best_inner].first;
                Pos p2 = input.card_pos[best_inner].second;
                if (best_inner_visit_first) {
                    current_start = p1; current_end = p2;
                } else {
                    current_start = p2; current_end = p1;
                }
            } else {
                break;
            }
        }
        return group;
    }
    
    int calc_group_cost(const Pos& start_pos, const vector<pair<int, bool>>& group) {
        if (group.empty()) return 0;
        
        int cost = 0;
        Pos cur = start_pos;
        
        // 往路
        for (size_t i = 0; i < group.size(); i++) {
            int card = group[i].first;
            bool visit_first = group[i].second;
            Pos target = visit_first ? input.card_pos[card].first : input.card_pos[card].second;
            cost += manhattan_dist(cur, target);
            cur = target;
        }
        
        // 復路
        for (int i = (int)group.size() - 1; i >= 0; i--) {
            int card = group[i].first;
            bool visit_first = group[i].second;
            Pos target = visit_first ? input.card_pos[card].second : input.card_pos[card].first;
            cost += manhattan_dist(cur, target);
            cur = target;
        }
        
        return cost;
    }
    
    void optimize_group_order(const Pos& start_pos, vector<pair<int, bool>>& group) {
        if (group.size() <= 1) return;
        
        int n = group.size();
        
        if (n <= 6) {
            vector<pair<int, bool>> inner(group.begin() + 1, group.end());
            sort(inner.begin(), inner.end());
            
            int best_cost = calc_group_cost(start_pos, group);
            vector<pair<int, bool>> best_order = group;
            
            do {
                vector<pair<int, bool>> candidate;
                candidate.push_back(group[0]);
                for (auto& p : inner) candidate.push_back(p);
                
                int cost = calc_group_cost(start_pos, candidate);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_order = candidate;
                }
            } while (next_permutation(inner.begin(), inner.end()));
            
            group = best_order;
        } else {
            int max_iterations = 10;
            for (int iter = 0; iter < max_iterations; iter++) {
                bool improved = false;
                int best_cost = calc_group_cost(start_pos, group);
                
                for (int i = 1; i < n - 1; i++) {
                    for (int j = i + 1; j < n; j++) {
                        vector<pair<int, bool>> candidate = group;
                        reverse(candidate.begin() + i, candidate.begin() + j + 1);
                        
                        int cost = calc_group_cost(start_pos, candidate);
                        if (cost < best_cost) {
                            best_cost = cost;
                            group = candidate;
                            improved = true;
                        }
                    }
                }
                if (!improved) break;
            }
        }
    }
    
    void output() {
        for (char c : best_operations) {
            cout << c << "\n";
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    Solver solver;
    solver.solve();
    
    return 0;
}
