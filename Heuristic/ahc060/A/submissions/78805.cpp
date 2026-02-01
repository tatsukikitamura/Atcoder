/**
 * AHC060 - Ice Cream Collection
 * 山登り法で「どの木をRにするか」を最適化 → 貪欲法で解構築
 */

 #include <iostream>
 #include <vector>
 #include <string>
 #include <algorithm>
 #include <queue>
 #include <cmath>
 #include <chrono>
 #include <random>
 #include <set>
 #include <map>
 #include <climits>
 using namespace std;
 
 // ===== Constants =====
 constexpr int MAX_NO_PROGRESS = 800;
 constexpr double HILL_CLIMB_TIME_LIMIT = 1.5;
 constexpr double TOTAL_TIME_LIMIT = 1.8;
 
 // ===== Timer =====
 class Timer {
 public:
     chrono::high_resolution_clock::time_point start_time;
     double time_limit;
     
     Timer(double limit_sec = TOTAL_TIME_LIMIT) : time_limit(limit_sec) {
         start_time = chrono::high_resolution_clock::now();
     }
     
     double elapsed() const {
         auto now = chrono::high_resolution_clock::now();
         return chrono::duration<double>(now - start_time).count();
     }
     
     bool has_time() const {
         return elapsed() < time_limit;
     }
 };
 
 // ===== Random (xorshift128+) =====
 class Random {
     uint64_t s[2];
 public:
     Random(uint64_t seed = 42) {
         s[0] = seed;
         s[1] = seed * 0x123456789ABCDEF;
     }
     
     uint64_t next() {
         uint64_t x = s[0], y = s[1];
         s[0] = y;
         x ^= x << 23;
         s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
         return s[1] + y;
     }
     
     int randint(int lo, int hi) {
         return lo + next() % (uint64_t)(hi - lo + 1);
     }
     
     double uniform(double lo = 0.0, double hi = 1.0) {
         return lo + (hi - lo) * (next() * (1.0 / UINT64_MAX));
     }
 };
 
 // ===== Solver =====
 class Solver {
 public:
     int N, M, K, T;
     vector<vector<int>> adj;
     vector<int> X, Y;
     vector<bool> is_red;
     vector<set<string>> shop_inventory;
     vector<int> actions;
     
     int current_pos;
     int prev_pos;
     string current_cone;
     int step_count;
     
     vector<vector<int>> shop_dist;
     
     // ----- Input/Output -----
     void read_input() {
         cin >> N >> M >> K >> T;
         adj.resize(N);
         X.resize(N);
         Y.resize(N);
         is_red.resize(N, false);
         shop_inventory.resize(K);
         
         for (int i = 0; i < M; i++) {
             int a, b;
             cin >> a >> b;
             adj[a].push_back(b);
             adj[b].push_back(a);
         }
         
         for (int i = 0; i < N; i++) {
             cin >> X[i] >> Y[i];
         }
         
         current_pos = 0;
         prev_pos = -1;
         current_cone = "";
         step_count = 0;
     }
     
     void output() {
         for (int a : actions) {
             cout << a << "\n";
         }
     }
     
     // ----- Helpers -----
     bool is_shop(int v) const { return v < K; }
     bool is_tree(int v) const { return v >= K; }
     
     bool move_to(int v) {
         if (step_count >= T) return false;
         if (v == prev_pos) return false;
         
         bool found = false;
         for (int u : adj[current_pos]) {
             if (u == v) { found = true; break; }
         }
         if (!found) return false;
         
         actions.push_back(v);
         step_count++;
         prev_pos = current_pos;
         current_pos = v;
         
         if (is_tree(v)) {
             current_cone += (is_red[v] ? 'R' : 'W');
         } else {
             shop_inventory[v].insert(current_cone);
             current_cone = "";
         }
         
         return true;
     }
     
     bool change_to_red() {
         if (step_count >= T) return false;
         if (!is_tree(current_pos)) return false;
         if (is_red[current_pos]) return false;
         
         actions.push_back(-1);
         step_count++;
         is_red[current_pos] = true;
         
         return true;
     }
     
     vector<int> bfs_dist(int start) {
         vector<int> dist(N, -1);
         queue<int> q;
         q.push(start);
         dist[start] = 0;
         while (!q.empty()) {
             int u = q.front(); q.pop();
             for (int v : adj[u]) {
                 if (dist[v] == -1) {
                     dist[v] = dist[u] + 1;
                     q.push(v);
                 }
             }
         }
         return dist;
     }
     
     // ----- Score Calculation (Common Logic) -----
     double calc_move_score(int v, const string& cone,
                           const vector<set<string>>& inventory,
                           const vector<bool>& red_flags,
                           const set<int>& trees_to_red) const {
         if (is_shop(v)) {
             if (inventory[v].count(cone) == 0) {
                 return 1000.0 + cone.length() * 50;
             } else {
                 return -100.0 - cone.length() * 30;
             }
         } else {
             char ice = red_flags[v] ? 'R' : 'W';
             string new_cone = cone + ice;
             
             int new_count = 0;
             int min_dist = INT_MAX;
             for (int s = 0; s < K; s++) {
                 if (inventory[s].count(new_cone) == 0) {
                     new_count++;
                     min_dist = min(min_dist, shop_dist[s][v]);
                 }
             }
             
             double score = (new_count > 0) ? (100.0 * new_count - min_dist) : -50.0;
             
             if (trees_to_red.count(v) && !red_flags[v]) {
                 score += 30.0;
             }
             
             return score;
         }
     }
     
     int select_best_move(const vector<int>& candidates, int prev,
                          const string& cone,
                          const vector<set<string>>& inventory,
                          const vector<bool>& red_flags,
                          const set<int>& trees_to_red,
                          Random& rng) const {
         int best_v = -1;
         double best_score = -1e18;
         
         for (int v : candidates) {
             if (v == prev) continue;
             
             double score = calc_move_score(v, cone, inventory, red_flags, trees_to_red);
             score += rng.uniform(-1, 1);
             
             if (score > best_score) {
                 best_score = score;
                 best_v = v;
             }
         }
         
         if (best_v == -1 && !candidates.empty()) {
             // Fallback: ランダム選択（prevを除く）
             vector<int> valid;
             for (int v : candidates) {
                 if (v != prev) valid.push_back(v);
             }
             if (!valid.empty()) {
                 best_v = valid[rng.randint(0, valid.size() - 1)];
             }
         }
         
         return best_v;
     }
     
     // ----- Simulation -----
     int simulate(const set<int>& trees_to_red, Random& rng, int max_steps = -1) {
         if (max_steps < 0) max_steps = T;
         
         vector<bool> sim_red(N, false);
         vector<set<string>> sim_inventory(K);
         int sim_pos = 0;
         int sim_prev = -1;
         string sim_cone = "";
         int sim_steps = 0;
         
         set<int> remaining_to_red = trees_to_red;
         
         while (sim_steps < max_steps) {
             int best_v = select_best_move(adj[sim_pos], sim_prev, sim_cone,
                                           sim_inventory, sim_red, remaining_to_red, rng);
             if (best_v == -1) break;
             
             sim_steps++;
             sim_prev = sim_pos;
             sim_pos = best_v;
             
             if (is_tree(best_v)) {
                 sim_cone += (sim_red[best_v] ? 'R' : 'W');
                 
                 if (remaining_to_red.count(best_v) && !sim_red[best_v]) {
                     if (sim_steps < max_steps) {
                         sim_steps++;
                         sim_red[best_v] = true;
                         remaining_to_red.erase(best_v);
                     }
                 }
             } else {
                 sim_inventory[best_v].insert(sim_cone);
                 sim_cone = "";
             }
         }
         
         int total = 0;
         for (int s = 0; s < K; s++) {
             total += sim_inventory[s].size();
         }
         return total;
     }
     
     // ----- Hill Climbing -----
     set<int> hill_climbing(Timer& timer, Random& rng) {
         cerr << "=== Hill Climbing ===" << endl;
         
         // 初期解: 優先度の高い木の上位半分をRにする
         vector<pair<double, int>> tree_priority;
         for (int t = K; t < N; t++) {
             double priority = 0;
             for (int s = 0; s < K; s++) {
                 priority += 1.0 / (shop_dist[s][t] + 1);
             }
             int adj_tree_count = 0;
             for (int u : adj[t]) {
                 if (is_tree(u)) adj_tree_count++;
             }
             priority += adj_tree_count * 0.5;
             tree_priority.push_back({priority, t});
         }
         sort(tree_priority.rbegin(), tree_priority.rend());
         
         set<int> best_trees;
         for (int i = 0; i < (int)tree_priority.size() / 2; i++) {
             best_trees.insert(tree_priority[i].second);
         }
         
         int sim_steps = T / 2;
         int best_score = simulate(best_trees, rng, sim_steps);
         cerr << "Initial score (partial): " << best_score << endl;
         
         int iterations = 0;
         int improvements = 0;
         
         while (timer.elapsed() < HILL_CLIMB_TIME_LIMIT) {
             iterations++;
             
             set<int> new_trees = best_trees;
             int num_changes = rng.randint(1, 3);
             
             for (int i = 0; i < num_changes; i++) {
                 int t = rng.randint(K, N - 1);
                 if (new_trees.count(t)) {
                     new_trees.erase(t);
                 } else {
                     new_trees.insert(t);
                 }
             }
             
             int new_score = simulate(new_trees, rng, sim_steps);
             
             if (new_score > best_score) {
                 best_score = new_score;
                 best_trees = new_trees;
                 improvements++;
             }
         }
         
         cerr << "Hill climbing: " << iterations << " iterations, " 
              << improvements << " improvements" << endl;
         cerr << "Best score (partial): " << best_score << endl;
         cerr << "Trees to red: " << best_trees.size() << "/" << (N - K) << endl;
         
         return best_trees;
     }
     
     // ----- Build Final Solution -----
     void build_final_solution(const set<int>& trees_to_red, Random& rng) {
         cerr << "=== Building Final Solution ===" << endl;
         
         fill(is_red.begin(), is_red.end(), false);
         for (int s = 0; s < K; s++) {
             shop_inventory[s].clear();
         }
         actions.clear();
         current_pos = 0;
         prev_pos = -1;
         current_cone = "";
         step_count = 0;
         
         set<int> remaining_to_red = trees_to_red;
         int no_progress_count = 0;
         
         while (step_count < T && no_progress_count < MAX_NO_PROGRESS) {
             int best_v = select_best_move(adj[current_pos], prev_pos, current_cone,
                                           shop_inventory, is_red, remaining_to_red, rng);
             if (best_v == -1) break;
             
             int old_score = 0;
             for (int s = 0; s < K; s++) {
                 old_score += shop_inventory[s].size();
             }
             
             move_to(best_v);
             
             if (is_tree(current_pos) && remaining_to_red.count(current_pos) && !is_red[current_pos]) {
                 if (step_count < T) {
                     change_to_red();
                     remaining_to_red.erase(current_pos);
                 }
             }
             
             int new_score = 0;
             for (int s = 0; s < K; s++) {
                 new_score += shop_inventory[s].size();
             }
             
             if (new_score > old_score) {
                 no_progress_count = 0;
             } else {
                 no_progress_count++;
             }
         }
         
         // 残りステップで追加収集
         while (step_count < T) {
             set<int> empty_set;
             int best_v = select_best_move(adj[current_pos], prev_pos, current_cone,
                                           shop_inventory, is_red, empty_set, rng);
             if (best_v == -1) break;
             move_to(best_v);
         }
     }
     
     // ----- Main Solve -----
     void solve() {
         Timer timer(TOTAL_TIME_LIMIT);
         Random rng(42);
         
         cerr << "N=" << N << " M=" << M << " K=" << K << " T=" << T << endl;
         
         // 全ショップへの最短距離を事前計算
         shop_dist.resize(K);
         for (int s = 0; s < K; s++) {
             shop_dist[s] = bfs_dist(s);
         }
         
         // 山登り法で最適な木の配置を探索
         set<int> best_trees = hill_climbing(timer, rng);
         
         // 最終解を構築
         build_final_solution(best_trees, rng);
         
         // 結果出力
         int total_score = 0;
         for (int s = 0; s < K; s++) {
             total_score += shop_inventory[s].size();
             cerr << "Shop " << s << ": " << shop_inventory[s].size() << " patterns" << endl;
         }
         cerr << "Total Score: " << total_score << endl;
         cerr << "Steps: " << step_count << "/" << T << endl;
         cerr << "Time: " << timer.elapsed() << "s" << endl;
     }
 };
 
 int main() {
     ios::sync_with_stdio(false);
     cin.tie(nullptr);
     
     Solver solver;
     solver.read_input();
     solver.solve();
     solver.output();
     
     return 0;
 }
 