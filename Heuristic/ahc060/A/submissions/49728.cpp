/**
 * AHC060 - Ice Cream Collection
 * 貪欲法による解法
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

// Timer utility for time-limited optimization
class Timer {
public:
    chrono::high_resolution_clock::time_point start_time;
    double time_limit;
    
    Timer(double limit_sec = 1.9) : time_limit(limit_sec) {
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

// Fast Random using xorshift128+
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

class Solver {
public:
    int N, M, K, T;
    vector<vector<int>> adj;  // 隣接リスト
    vector<int> X, Y;         // 座標
    vector<bool> is_red;      // 木がRかどうか（初期は全てW=false）
    
    vector<set<string>> shop_inventory;  // 各ショップの在庫集合
    vector<int> actions;      // 出力する行動列
    
    int current_pos;          // 現在位置
    int prev_pos;             // 直前の位置（戻れない）
    string current_cone;      // 現在のコーンの状態
    int step_count;           // 現在のステップ数
    
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
    
    bool is_shop(int v) {
        return v < K;
    }
    
    bool is_tree(int v) {
        return v >= K;
    }
    
    // 行動1: 移動
    bool move_to(int v) {
        if (step_count >= T) return false;
        if (v == prev_pos) return false;  // 直前の位置には戻れない
        
        // vが隣接しているか確認
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
            // アイスを収穫してコーンに追加
            current_cone += (is_red[v] ? 'R' : 'W');
        } else {
            // ショップに納品
            shop_inventory[v].insert(current_cone);
            current_cone = "";
        }
        
        return true;
    }
    
    // 行動2: 木の味変更（W→R）
    bool change_to_red() {
        if (step_count >= T) return false;
        if (!is_tree(current_pos)) return false;
        if (is_red[current_pos]) return false;  // 既にR
        
        actions.push_back(-1);
        step_count++;
        is_red[current_pos] = true;
        
        return true;
    }
    
    // BFSで最短経路を求める（prev_posを避ける）
    vector<int> find_path(int from, int to) {
        if (from == to) return {};
        
        vector<int> parent(N, -1);
        queue<int> q;
        q.push(from);
        parent[from] = from;
        
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (parent[v] == -1) {
                    // 最初の一歩でprev_posには行けない
                    if (u == from && v == prev_pos) continue;
                    parent[v] = u;
                    if (v == to) {
                        // 経路を復元
                        vector<int> path;
                        int cur = to;
                        while (cur != from) {
                            path.push_back(cur);
                            cur = parent[cur];
                        }
                        reverse(path.begin(), path.end());
                        return path;
                    }
                    q.push(v);
                }
            }
        }
        return {};  // 到達不能（2-辺連結なのでありえないはず）
    }
    
    // 経路に沿って移動
    bool follow_path(const vector<int>& path) {
        for (int v : path) {
            if (!move_to(v)) return false;
        }
        return true;
    }
    
    void solve() {
        Timer timer(1.8);
        Random rng(42);
        
        cerr << "N=" << N << " M=" << M << " K=" << K << " T=" << T << endl;
        
        // 戦略: 
        // 1. まず各ショップに空文字列を納品（ショップ間を移動）
        // 2. 木を適当にR/Wに設定
        // 3. 各ショップに対して、様々な長さの文字列を納品
        
        // フェーズ1: 木の味を決定
        // 各ショップの近くの木をバランスよくW/Rに分ける
        set<int> should_be_red;
        
        // 各木について、ランダムに半分をRにする
        for (int t = K; t < N; t++) {
            if (rng.uniform() < 0.5) {
                should_be_red.insert(t);
            }
        }
        
        // フェーズ2: 貪欲に巡回
        // 戦略: 各ショップに対して、まだ納品していない文字列を作れる経路を探す
        
        // 全ショップへの最短経路を事前計算
        auto bfs_dist = [&](int start) -> vector<int> {
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
        };
        
        vector<vector<int>> shop_dist(K);
        for (int s = 0; s < K; s++) {
            shop_dist[s] = bfs_dist(s);
        }
        
        int iteration = 0;
        while (step_count < T) {
            iteration++;
            
            // 移動可能な隣接頂点を取得
            vector<int> candidates;
            for (int v : adj[current_pos]) {
                if (v != prev_pos) candidates.push_back(v);
            }
            if (candidates.empty()) break;
            
            // 各候補のスコアを計算
            int best_v = -1;
            double best_score = -1e18;
            
            for (int v : candidates) {
                double score = 0;
                
                if (is_shop(v)) {
                    // ショップへの移動
                    if (shop_inventory[v].count(current_cone) == 0) {
                        // 新しい文字列を納品できる
                        // 長い文字列ほど価値が高い（作るのが難しいため）
                        score = 1000.0 + current_cone.length() * 50;
                    } else {
                        // 既存の文字列（コーンをリセットするだけ）
                        // コーンが長い場合は無駄になるのでペナルティ
                        score = -100.0 - current_cone.length() * 30;
                    }
                } else {
                    // 木への移動
                    char ice = is_red[v] ? 'R' : 'W';
                    string new_cone = current_cone + ice;
                    
                    // この文字列がどのショップで新しいかカウント
                    int new_count = 0;
                    int min_dist_to_shop = INT_MAX;
                    for (int s = 0; s < K; s++) {
                        if (shop_inventory[s].count(new_cone) == 0) {
                            new_count++;
                            min_dist_to_shop = min(min_dist_to_shop, shop_dist[s][v]);
                        }
                    }
                    
                    if (new_count > 0) {
                        // 新しい文字列が作れる
                        score = 100.0 * new_count - min_dist_to_shop;
                    } else {
                        // 全ショップで既存
                        score = -50.0;
                    }
                    
                    // 味変更が必要な木にはボーナス
                    if (should_be_red.count(v) && !is_red[v]) {
                        score += 20.0;
                    }
                }
                
                // ランダム性を少し加える
                score += rng.uniform(-1, 1);
                
                if (score > best_score) {
                    best_score = score;
                    best_v = v;
                }
            }
            
            if (best_v == -1) {
                best_v = candidates[rng.randint(0, candidates.size() - 1)];
            }
            
            // 移動実行
            move_to(best_v);
            
            // 木に到着して味変更が必要なら実行
            if (is_tree(current_pos) && should_be_red.count(current_pos) && !is_red[current_pos]) {
                if (step_count < T) {
                    change_to_red();
                }
            }
        }
        
        cerr << "Iterations: " << iteration << endl;
        
        // スコア計算
        int total_score = 0;
        for (int s = 0; s < K; s++) {
            total_score += shop_inventory[s].size();
        }
        cerr << "Score: " << total_score << endl;
        cerr << "Steps: " << step_count << "/" << T << endl;
        cerr << "Time: " << timer.elapsed() << "s" << endl;
    }
    
    void output() {
        for (int a : actions) {
            cout << a << "\n";
        }
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
