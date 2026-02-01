/**
 * AHC060 - Ice Cream Collection
 * 焼きなまし法で「どの木をRにするか」を最適化 → 貪欲法で解構築
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
 #include <bitset>
 using namespace std;
  
// ===== Constants =====
constexpr int MAX_NO_PROGRESS = 800;
constexpr double SA_TIME_LIMIT = 1.5;
constexpr double TOTAL_TIME_LIMIT = 1.8;

// ===== Simulated Annealing Parameters =====
constexpr double SA_START_TEMP = 50.0;
constexpr double SA_END_TEMP = 0.1;
 
 // ===== Fast Simulation Parameters =====
 constexpr int MAX_PATTERN_LEN = 12;
 constexpr int NUM_PATTERNS = (1 << (MAX_PATTERN_LEN + 1)) - 2;  // 長さ1-12の全パターン数
  
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
         return lo + (hi - lo) * (next() * (1.0 / (double)UINT64_MAX));
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
     
     // パターンをintにエンコード（高速シミュレーション用）
     int encode_pattern(const string& s) const {
         if (s.empty()) return -1;
         if ((int)s.length() > MAX_PATTERN_LEN) return -1;
         int base = (1 << s.length()) - 2;
         int offset = 0;
         for (char c : s) {
             offset = offset * 2 + (c == 'R' ? 1 : 0);
         }
         return base + offset;
     }
     
     // ビットマスクでの赤/白管理
     bool get_red_bit(uint64_t mask1, uint64_t mask2, int tree_idx) const {
         int idx = tree_idx - K;
         if (idx < 64) return (mask1 >> idx) & 1;
         return (mask2 >> (idx - 64)) & 1;
     }
     
     void set_red_bit(uint64_t& mask1, uint64_t& mask2, int tree_idx) const {
         int idx = tree_idx - K;
         if (idx < 64) mask1 |= (1ULL << idx);
         else mask2 |= (1ULL << (idx - 64));
     }
      
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
             if (inventory[v].count(cone) == 0 && !cone.empty()) {
                 // 新しいパターンを納品できる
                 // 長さボーナスを指数的に + 在庫バランスを考慮
                 double length_bonus = pow(1.5, cone.length());
                 int shop_size = inventory[v].size();
                 double balance_bonus = 50.0 / (shop_size + 1);  // 在庫が少ないショップを優先
                 return 1000.0 + length_bonus * 30 + balance_bonus;
             } else if (cone.empty()) {
                 // 空文字列は価値が低い
                 return -200.0;
             } else {
                 // 重複パターン
                 return -100.0 - cone.length() * 20;
             }
         } else {
             char ice = red_flags[v] ? 'R' : 'W';
             string new_cone = cone + ice;
             
             int new_count = 0;
             int min_dist = INT_MAX;
             double balance_score = 0;
             
             for (int s = 0; s < K; s++) {
                 if (inventory[s].count(new_cone) == 0) {
                     new_count++;
                     int dist = shop_dist[s][v];
                     min_dist = min(min_dist, dist);
                     // 在庫が少ないショップへの距離を重視
                     int shop_size = inventory[s].size();
                     balance_score += 10.0 / ((shop_size + 1) * (dist + 1));
                 }
             }
             
             double score;
             if (new_count > 0) {
                 // 長さボーナスを指数的に
                 double length_bonus = pow(1.3, new_cone.length());
                 score = 100.0 * new_count + length_bonus * 10 + balance_score - min_dist * 0.5;
             } else {
                 score = -50.0;
             }
             
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
     
    // ----- Lookahead evaluation (2-step) -----
    double evaluate_lookahead(int pos, int prev, const string& cone,
                              vector<set<string>>& inventory,
                              const vector<bool>& red_flags,
                              const set<int>& trees_to_red,
                              int depth, Random& rng) {
        if (depth == 0) {
            return 0.0;
        }
        
        double best_future = -1e18;
        
        for (int v : adj[pos]) {
            if (v == prev) continue;
            
            double immediate = calc_move_score(v, cone, inventory, red_flags, trees_to_red);
            
            // 次の状態を計算
            string next_cone = cone;
            if (is_tree(v)) {
                next_cone += (red_flags[v] ? 'R' : 'W');
            } else {
                // ショップに納品 - 一時的にinventoryを更新
                bool is_new = (inventory[v].count(cone) == 0 && !cone.empty());
                if (is_new) {
                    inventory[v].insert(cone);
                }
                next_cone = "";
                
                // 再帰的に評価
                double future = evaluate_lookahead(v, pos, next_cone, inventory, red_flags, trees_to_red, depth - 1, rng);
                
                // inventoryを元に戻す
                if (is_new) {
                    inventory[v].erase(cone);
                }
                
                double total = immediate + future * 0.5;  // 将来の価値を減衰
                best_future = max(best_future, total);
                continue;
            }
            
            // 木の場合
            double future = evaluate_lookahead(v, pos, next_cone, inventory, red_flags, trees_to_red, depth - 1, rng);
            double total = immediate + future * 0.5;
            best_future = max(best_future, total);
        }
        
        return best_future;
    }
    
    // ----- Select best move with lookahead -----
    int select_best_move_with_lookahead(const vector<int>& candidates, int prev,
                                        const string& cone,
                                        vector<set<string>>& inventory,
                                        const vector<bool>& red_flags,
                                        const set<int>& trees_to_red,
                                        Random& rng, int lookahead_depth = 2) {
        // まず全候補のスコアを計算
        vector<pair<double, int>> scored_moves;
        for (int v : candidates) {
            if (v == prev) continue;
            double score = calc_move_score(v, cone, inventory, red_flags, trees_to_red);
            scored_moves.push_back({score, v});
        }
        
        if (scored_moves.empty()) return -1;
        
        // スコア順にソート
        sort(scored_moves.rbegin(), scored_moves.rend());
        
        // 上位3つの候補に対して先読みを行う
        int top_k = min(3, (int)scored_moves.size());
        int best_v = scored_moves[0].second;
        double best_total = -1e18;
        
        for (int i = 0; i < top_k; i++) {
            int v = scored_moves[i].second;
            double immediate = scored_moves[i].first;
            
            // 次の状態を計算して先読み
            string next_cone = cone;
            if (is_tree(v)) {
                next_cone += (red_flags[v] ? 'R' : 'W');
            } else {
                // ショップの場合
                bool is_new = (inventory[v].count(cone) == 0 && !cone.empty());
                if (is_new) {
                    inventory[v].insert(cone);
                }
                next_cone = "";
                
                double future = evaluate_lookahead(v, prev, next_cone, inventory, red_flags, trees_to_red, lookahead_depth - 1, rng);
                
                if (is_new) {
                    inventory[v].erase(cone);
                }
                
                double total = immediate + future * 0.5;
                if (total > best_total) {
                    best_total = total;
                    best_v = v;
                }
                continue;
            }
            
            double future = evaluate_lookahead(v, prev, next_cone, inventory, red_flags, trees_to_red, lookahead_depth - 1, rng);
            double total = immediate + future * 0.5;
            total += rng.uniform(-0.5, 0.5);  // ランダム性を追加
            
            if (total > best_total) {
                best_total = total;
                best_v = v;
            }
        }
        
        return best_v;
    }
     
   // ----- Fast Simulation (using bitset) -----
    // 実際の解構築と同じモデル: 最初は全てW、通ったらRに変更
    int simulate_fast(uint64_t red_mask1, uint64_t red_mask2, Random& rng, int max_steps = -1) {
        if (max_steps < 0) max_steps = T;
        
        bitset<NUM_PATTERNS> inventory[10];  // K <= 10
        // 現在の木の状態（最初は全てW）
        uint64_t current_red1 = 0, current_red2 = 0;
        // まだRに変更していない木（red_maskで指定された木）
        uint64_t remaining_red1 = red_mask1, remaining_red2 = red_mask2;
        
        int sim_pos = 0;
        int sim_prev = -1;
        string sim_cone = "";
        int sim_steps = 0;
        int score = 0;
        
        while (sim_steps < max_steps) {
            // 簡易的な貪欲選択（高速版）
            int best_v = -1;
            double best_score = -1e18;
            
            for (int v : adj[sim_pos]) {
                if (v == sim_prev) continue;
                
                double move_score;
                if (is_shop(v)) {
                    int pattern_idx = encode_pattern(sim_cone);
                    if (pattern_idx >= 0 && !inventory[v].test(pattern_idx)) {
                        // 新しいパターン: 長さボーナス + 在庫バランス
                        double length_bonus = pow(1.5, sim_cone.length());
                        int shop_count = inventory[v].count();
                        double balance_bonus = 50.0 / (shop_count + 1);
                        move_score = 1000.0 + length_bonus * 30 + balance_bonus;
                    } else if (sim_cone.empty()) {
                        move_score = -200.0;  // 空文字列
                    } else {
                        move_score = -100.0 - sim_cone.length() * 20;
                    }
                } else {
                    // 現在の木の状態を使う（最初はW、変更後はR）
                    char ice = get_red_bit(current_red1, current_red2, v) ? 'R' : 'W';
                    string new_cone = sim_cone + ice;
                    int pattern_idx = encode_pattern(new_cone);
                    
                    int new_count = 0;
                    int min_dist = INT_MAX;
                    double balance_score = 0;
                    
                    if (pattern_idx >= 0) {
                        for (int s = 0; s < K; s++) {
                            if (!inventory[s].test(pattern_idx)) {
                                new_count++;
                                int dist = shop_dist[s][v];
                                min_dist = min(min_dist, dist);
                                int shop_count = inventory[s].count();
                                balance_score += 10.0 / ((shop_count + 1) * (dist + 1));
                            }
                        }
                    }
                    
                    if (new_count > 0) {
                        double length_bonus = pow(1.3, new_cone.length());
                        move_score = 100.0 * new_count + length_bonus * 10 + balance_score - min_dist * 0.5;
                    } else {
                        move_score = -50.0;
                    }
                    
                    // Rにすべき木で、まだRになっていないならボーナス
                    if (get_red_bit(remaining_red1, remaining_red2, v)) {
                        move_score += 30.0;
                    }
                }
                
                move_score += rng.uniform(-1, 1);
                if (move_score > best_score) {
                    best_score = move_score;
                    best_v = v;
                }
            }
            
            if (best_v == -1) break;
            
            sim_steps++;
            sim_prev = sim_pos;
            sim_pos = best_v;
            
            if (is_tree(best_v)) {
                // 現在の状態でアイスを取得
                sim_cone += (get_red_bit(current_red1, current_red2, best_v) ? 'R' : 'W');
                
                // この木をRに変更すべきで、まだRでないなら変更
                if (get_red_bit(remaining_red1, remaining_red2, best_v) && 
                    !get_red_bit(current_red1, current_red2, best_v)) {
                    if (sim_steps < max_steps) {
                        sim_steps++;  // 変更に1ステップ消費
                        set_red_bit(current_red1, current_red2, best_v);
                        // remaining から削除
                        int idx = best_v - K;
                        if (idx < 64) remaining_red1 &= ~(1ULL << idx);
                        else remaining_red2 &= ~(1ULL << (idx - 64));
                    }
                }
            } else {
                 int pattern_idx = encode_pattern(sim_cone);
                 if (pattern_idx >= 0 && !inventory[best_v].test(pattern_idx)) {
                     inventory[best_v].set(pattern_idx);
                     score++;
                 }
                 sim_cone = "";
             }
         }
         
         return score;
     }
     
     // set<int>からビットマスクに変換
     pair<uint64_t, uint64_t> trees_to_bitmask(const set<int>& trees) const {
         uint64_t mask1 = 0, mask2 = 0;
         for (int t : trees) {
             set_red_bit(mask1, mask2, t);
         }
         return {mask1, mask2};
     }
     
     // ビットマスクからset<int>に変換
     set<int> bitmask_to_trees(uint64_t mask1, uint64_t mask2) const {
         set<int> trees;
         for (int t = K; t < N; t++) {
             if (get_red_bit(mask1, mask2, t)) {
                 trees.insert(t);
             }
         }
         return trees;
     }
     
     // ----- Simulation (original, for final solution) -----
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
      
    // ----- SA Search from a single initial solution (with adaptive reheating) -----
    tuple<uint64_t, uint64_t, int> sa_search_single(
        uint64_t init_mask1, uint64_t init_mask2,
        const vector<pair<double, int>>& tree_priority,
        double time_limit, int sim_steps, Random& rng) {
        
        uint64_t current_mask1 = init_mask1, current_mask2 = init_mask2;
        int current_score = simulate_fast(current_mask1, current_mask2, rng, sim_steps);
        
        uint64_t best_mask1 = current_mask1, best_mask2 = current_mask2;
        int best_score = current_score;
        
        auto start = chrono::high_resolution_clock::now();
        
        // 適応的温度調整用
        constexpr int REHEAT_THRESHOLD = 50;  // 改善なしの回数閾値
        int no_improvement_count = 0;
        double temp_multiplier = 1.0;  // 温度の乗数（再加熱時に上げる）
        
        while (true) {
            auto now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(now - start).count();
            if (elapsed >= time_limit) break;
            
            // 温度スケジュール（適応的）
            double progress = elapsed / time_limit;
            double base_temp = SA_START_TEMP * pow(SA_END_TEMP / SA_START_TEMP, progress);
            double temp = base_temp * temp_multiplier;
            
            // 再加熱: 改善が停滞したら温度を一時的に上げる
            if (no_improvement_count >= REHEAT_THRESHOLD) {
                temp_multiplier = min(temp_multiplier * 1.5, 3.0);  // 最大3倍まで
                no_improvement_count = 0;
            } else {
                // 徐々に元に戻す
                temp_multiplier = max(1.0, temp_multiplier * 0.99);
            }
            
            uint64_t new_mask1 = current_mask1, new_mask2 = current_mask2;
            
            // 近傍操作の選択
            int op = rng.randint(0, 99);
            
            if (op < 40) {
                int t = rng.randint(K, N - 1);
                int idx = t - K;
                if (idx < 64) new_mask1 ^= (1ULL << idx);
                else new_mask2 ^= (1ULL << (idx - 64));
            } else if (op < 75) {
                int flip1 = rng.randint(K, N - 1);
                int flip2 = rng.randint(K, N - 1);
                while (flip2 == flip1) flip2 = rng.randint(K, N - 1);
                
                bool is_red1 = get_red_bit(new_mask1, new_mask2, flip1);
                bool is_red2 = get_red_bit(new_mask1, new_mask2, flip2);
                
                if (is_red1 != is_red2) {
                    int idx1 = flip1 - K, idx2 = flip2 - K;
                    if (idx1 < 64) new_mask1 ^= (1ULL << idx1);
                    else new_mask2 ^= (1ULL << (idx1 - 64));
                    if (idx2 < 64) new_mask1 ^= (1ULL << idx2);
                    else new_mask2 ^= (1ULL << (idx2 - 64));
                }
            } else if (op < 90) {
                int t = rng.randint(K, N - 1);
                int idx = t - K;
                if (idx < 64) new_mask1 ^= (1ULL << idx);
                else new_mask2 ^= (1ULL << (idx - 64));
                
                for (int u : adj[t]) {
                    if (is_tree(u) && rng.uniform() < 0.5) {
                        int uidx = u - K;
                        if (uidx < 64) new_mask1 ^= (1ULL << uidx);
                        else new_mask2 ^= (1ULL << (uidx - 64));
                    }
                }
            } else {
                int selected_idx = rng.randint(0, (int)tree_priority.size() - 1);
                if (rng.uniform() < 0.7) {
                    selected_idx = rng.randint(0, max(1, (int)tree_priority.size() / 3));
                }
                int t = tree_priority[selected_idx].second;
                int idx = t - K;
                if (idx < 64) new_mask1 ^= (1ULL << idx);
                else new_mask2 ^= (1ULL << (idx - 64));
            }
            
            int new_score = simulate_fast(new_mask1, new_mask2, rng, sim_steps);
            
            double diff = new_score - current_score;
            if (diff > 0 || rng.uniform() < exp(diff / temp)) {
                current_score = new_score;
                current_mask1 = new_mask1;
                current_mask2 = new_mask2;
                
                if (new_score > best_score) {
                    best_score = new_score;
                    best_mask1 = new_mask1;
                    best_mask2 = new_mask2;
                    no_improvement_count = 0;  // 改善があったのでリセット
                } else {
                    no_improvement_count++;
                }
            } else {
                no_improvement_count++;
            }
        }
        
        return {best_mask1, best_mask2, best_score};
    }
    
    // ----- Simulated Annealing with Multiple Initial Solutions -----
    set<int> simulated_annealing(Timer& timer, Random& rng) {
        cerr << "=== Simulated Annealing (Multi-Start) ===" << endl;
        
        // 木の優先度を計算
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
        
        int num_trees = N - K;
        int sim_steps = T / 2;
        
        // 複数の初期解を生成
        vector<pair<uint64_t, uint64_t>> initial_solutions;
        
        // 初期解1: 優先度上位60%をR
        {
            uint64_t mask1 = 0, mask2 = 0;
            int count = num_trees * 60 / 100;
            for (int i = 0; i < count; i++) {
                set_red_bit(mask1, mask2, tree_priority[i].second);
            }
            initial_solutions.push_back({mask1, mask2});
        }
        
        // 初期解2: ランダム50%をR
        {
            uint64_t mask1 = 0, mask2 = 0;
            for (int t = K; t < N; t++) {
                if (rng.uniform() < 0.5) {
                    set_red_bit(mask1, mask2, t);
                }
            }
            initial_solutions.push_back({mask1, mask2});
        }
        
        // 初期解3: 優先度上位40%をR（少なめ）
        {
            uint64_t mask1 = 0, mask2 = 0;
            int count = num_trees * 40 / 100;
            for (int i = 0; i < count; i++) {
                set_red_bit(mask1, mask2, tree_priority[i].second);
            }
            initial_solutions.push_back({mask1, mask2});
        }
        
        // 各初期解に時間を配分して探索
        double remaining_time = SA_TIME_LIMIT - timer.elapsed();
        double time_per_solution = remaining_time / initial_solutions.size();
        
        uint64_t global_best_mask1 = 0, global_best_mask2 = 0;
        int global_best_score = -1;
        
        for (size_t i = 0; i < initial_solutions.size(); i++) {
            auto [init_mask1, init_mask2] = initial_solutions[i];
            
            auto [best_mask1, best_mask2, best_score] = sa_search_single(
                init_mask1, init_mask2, tree_priority, time_per_solution, sim_steps, rng);
            
            cerr << "  Initial solution " << i << ": score = " << best_score << endl;
            
            if (best_score > global_best_score) {
                global_best_score = best_score;
                global_best_mask1 = best_mask1;
                global_best_mask2 = best_mask2;
            }
        }
        
        // ビットマスクからsetに変換して返す
        set<int> best_trees = bitmask_to_trees(global_best_mask1, global_best_mask2);
        
        cerr << "Best score (partial): " << global_best_score << endl;
        cerr << "Trees to red: " << best_trees.size() << "/" << num_trees << endl;
        
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
         
         // ===== メインフェーズ: 木をRに変更しながらパターンを収集 =====
         
         while (step_count < T && no_progress_count < MAX_NO_PROGRESS) {
             int best_v = select_best_move(adj[current_pos], prev_pos, current_cone,
                                           shop_inventory, is_red, remaining_to_red, rng);
             if (best_v == -1) break;
             
             int old_score = 0;
             for (int s = 0; s < K; s++) {
                 old_score += shop_inventory[s].size();
             }
             
            move_to(best_v);
            
            // 焼きなましで決めた木を通ったら確定的にRに変更（シミュレーションと一致）
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
         
         // ===== 残りステップで追加収集 =====
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
          
         // 焼きなまし法で最適な木の配置を探索
         set<int> best_trees = simulated_annealing(timer, rng);
          
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
  