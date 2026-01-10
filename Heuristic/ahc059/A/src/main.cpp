#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <cmath>
#include <chrono>
#include <random>
#include <set>
using namespace std;

// ===== Constants =====
constexpr int N = 20;
constexpr int INF = 1e9;

// ===== Timer =====
struct Timer {
    chrono::high_resolution_clock::time_point start;
    double limit;
    
    Timer(double limit_sec = 1.95) : limit(limit_sec) {
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
    int board[N][N];           // 各マスのカード番号
    pair<Pos, Pos> card_pos[N * N / 2];  // 各カード番号の2枚の位置
    
    void read() {
        int n;
        if (!(cin >> n)) return;  // N=20固定
        vector<bool> first_seen(N * N / 2, true);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cin >> board[i][j];
                int card = board[i][j];
                
                // Safety check
                if (card < 0 || card >= N * N / 2) {
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

// ===== Utility Functions =====
int manhattan_dist(const Pos& a, const Pos& b) {
    return abs(a.r - b.r) + abs(a.c - b.c);
}

// 2点間の移動操作を現在の操作列に追記する
void append_path(string& path, int& move_count, const Pos& from, const Pos& to) {
    int dr = to.r - from.r;
    int dc = to.c - from.c;
    char r_char = (dr > 0) ? 'D' : 'U';
    char c_char = (dc > 0) ? 'R' : 'L';
    dr = abs(dr);
    dc = abs(dc);
    for (int i = 0; i < dr; i++) {
        path += r_char;
        move_count++;
    }
    for (int i = 0; i < dc; i++) {
        path += c_char;
        move_count++;
    }
}

// ===== Solver =====
class Solver {
public:
    Input input;
    Timer timer;
    string best_operations;
    int best_move_count;
    
    // For current iteration
    string operations;
    int move_count;
    Pos current_pos;
    
    vector<bool> used;
    
    struct Candidate {
        int card;
        bool visit_first;
        int cost;
        int density_bonus;  // 周囲のカード密集度（高いほど良い）
        double score;       // 総合スコア（低いほど良い）
    };
    vector<Candidate> candidates;

    mt19937 rng;
    
    Solver() : best_move_count(INF), rng(42) {
        used.resize(N * N / 2);
        candidates.reserve(N * N); // Max candidates (2 per card)
    }
    
    void solve() {
        input.read();
        
        int iteration = 0;
        
        while (!timer.is_over()) {
            // 初回は必ずdeterministic (greedy) を実行し、ベースラインスコアを保証する
            bool deterministic = (iteration == 0);
            run_iteration(deterministic);
            
            if (move_count < best_move_count) {
                best_move_count = move_count;
                best_operations = operations;
                cerr << "[DEBUG] New Best: " << best_move_count << " (Iter " << iteration + 1 << ")" << endl;
            }
            iteration++;
        }
        
        cerr << "[DEBUG] Total Iterations: " << iteration << ", Time: " << timer.elapsed() << "s" << endl;
        output();
    }
    
    // マージグループを実行する操作列を生成 (Iterative)
    void generate_merge_group_operations(const vector<int>& group, 
                                          const vector<bool>& group_visit_first) {
        // [Outer1 -> [Inner1 -> ... -> [Core] ... -> Inner2] -> Outer2]
        // = (Outer1 -> Inner1 -> ... -> Core1 -> Core2 -> ... -> Inner2 -> Outer2)
        // 順番:
        // 1. group[0] 1st -> group[1] 1st -> ... -> group[k] 1st
        // 2. group[k] 2nd -> ... -> group[1] 2nd -> group[0] 2nd
        
        // 1. 往路 (Outer -> Inner)
        for (size_t i = 0; i < group.size(); ++i) {
            int card = group[i];
            Pos p1 = input.card_pos[card].first;
            Pos p2 = input.card_pos[card].second;
            Pos target = group_visit_first[i] ? p1 : p2;
            
            append_path(operations, move_count, current_pos, target);
            current_pos = target;
            operations += 'Z';
        }

        // 2. 復路 (Inner -> Outer)
        for (int i = (int)group.size() - 1; i >= 0; --i) {
            int card = group[i];
            Pos p1 = input.card_pos[card].first;
            Pos p2 = input.card_pos[card].second;
            Pos target = group_visit_first[i] ? p2 : p1; // 往路と逆側
            
            append_path(operations, move_count, current_pos, target);
            current_pos = target;
            operations += 'Z';
        }
    }
    
    // 挿入コストを計算（寄り道コスト）
    // Outerの移動経路(O_start -> O_end)の間にInner(I_start -> I_end)を挟む場合の追加コスト
    // Cost = (dist(O_S, I_S) + dist(I_S, I_E) + dist(I_E, O_E)) - dist(O_S, O_E)
    // ただし、Inner自体の移動距離(dist(I_S, I_E))は必須なので、純粋な「寄り道分」は
    // (dist(O_S, I_S) + dist(I_E, O_E)) - (dist(O_S, O_E) - dist(I_S, I_E)) ... ではなく
    // 単純に、トータル移動距離の増加分を見るのが適切。
    // 元の移動距離: dist(O_S, O_E)
    // 新しい移動距離: dist(O_S, I_S) + dist(I_S, I_E) + dist(I_E, O_E)
    // 増加分 (Insertion Cost): 新しい移動距離 - 元の移動距離 - Innerの本来の移動距離
    // ... と考えたいが、ここでは「親のパスにどれだけスムーズに乗れるか」を評価したい。
    // 親のパスの中に完全に包含されるなら、追加コストは0になるべき。
    // 親: A -> B, 子: C -> D
    // A -> C -> D -> B
    // 距離: dist(A,C) + dist(C,D) + dist(D,B)
    // 親の本来: dist(A,B)
    // 子の本来: dist(C,D)
    // 増加分 = (dist(A,C) + dist(C,D) + dist(D,B)) - dist(A,B) - dist(C,D)
    //        = dist(A,C) + dist(D,B) - dist(A,B)
    // これが0なら、CとDはA->Bの最短経路上にあり、かつ順序も整合している。
    int calc_insertion_cost(const Pos& o_s, const Pos& o_e, const Pos& i_s, const Pos& i_e) {
        int d_total = manhattan_dist(o_s, i_s) + manhattan_dist(i_s, i_e) + manhattan_dist(i_e, o_e);
        int d_base = manhattan_dist(o_s, o_e) + manhattan_dist(i_s, i_e);
        return d_total - d_base;
    }

    // ===== 始点・終点近接グループ化戦略 (Iterative Nesting Check) =====
    // 戻り値を vector<int> から vector<pair<int, bool>> に変更
    // 各カードについて {card_id, visit_first} のペアを返す
    vector<pair<int, bool>> find_proximity_group(int main_card, bool main_visit_first, 
                                     const vector<bool>& used, int max_depth = 3) {
        vector<pair<int, bool>> group;
        group.push_back({main_card, main_visit_first});
        
        Pos current_start = main_visit_first ? input.card_pos[main_card].first : input.card_pos[main_card].second;
        Pos current_end = main_visit_first ? input.card_pos[main_card].second : input.card_pos[main_card].first;
        
        for (int depth = 1; depth < max_depth; depth++) {
            int best_inner = -1;
            int min_insertion_cost = INF;
            bool best_inner_visit_first = true;

            for (int card = 0; card < N * N / 2; card++) {
                if (used[card]) continue;
                if (card == main_card) continue;
                
                bool already_in_group = false;
                for (auto& g : group) if (g.first == card) already_in_group = true;
                if (already_in_group) continue;
                
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                
                int cost1 = calc_insertion_cost(current_start, current_end, p1, p2);
                int cost2 = calc_insertion_cost(current_start, current_end, p2, p1);
                
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

    void run_iteration(bool deterministic) {
        operations.clear();
        move_count = 0;
        current_pos = Pos(0, 0);
        fill(used.begin(), used.end(), false);
        int remaining = N * N / 2;
        while (remaining > 0) {
            candidates.clear();
            
            for (int card = 0; card < N * N / 2; card++) {
                if (used[card]) continue;
                
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                
                int cost1 = manhattan_dist(current_pos, p1);
                int cost2 = manhattan_dist(current_pos, p2);
                
                if (cost1 <= cost2) {
                    candidates.push_back({card, true, cost1});
                } else {
                    candidates.push_back({card, false, cost2});
                }
            }
            
            if (candidates.empty()) break;
            
            // コストでソート
            sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                return a.cost < b.cost;
            });
            
            Candidate best = candidates[0];
            if (!deterministic) {
                int K = 8;  // 候補数を増加
                int range = min((int)candidates.size(), K);
                
                // コストに基づくsoftmax（ボルツマン）選択
                double T = 1.5;  // 温度パラメータ（大きいほど探索的、小さいほどGreedy）
                
                // 重みを計算
                vector<double> weights(range);
                double sum_weights = 0.0;
                int min_cost = candidates[0].cost;
                for (int i = 0; i < range; i++) {
                    // コスト差に基づく重み (最小コストを基準に)
                    weights[i] = exp(-(candidates[i].cost - min_cost) / T);
                    sum_weights += weights[i];
                }
                
                // 確率的に選択
                double r = uniform_real_distribution<double>(0, sum_weights)(rng);
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
            
            int best_card = best.card;
            bool best_visit_first = best.visit_first;
            
            vector<pair<int, bool>> detailed_group = find_proximity_group(best_card, best_visit_first, used, 200);
            
            vector<int> merge_group;
            vector<bool> group_visit_first;
            for (auto& p : detailed_group) {
                merge_group.push_back(p.first);
                group_visit_first.push_back(p.second);
            }
            
            generate_merge_group_operations(merge_group, group_visit_first);
            
            for (int card : merge_group) {
                if (!used[card]) {
                    used[card] = true;
                    remaining--;
                }
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
