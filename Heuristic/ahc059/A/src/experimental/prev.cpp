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
    
    Timer(double limit_sec = 1.9) : limit(limit_sec) {
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
        cin >> n;  // N=20固定
        
        vector<bool> first_seen(N * N / 2, true);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cin >> board[i][j];
                int card = board[i][j];
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

// 2点間の移動操作列を生成（単純な直線移動）
string get_path(const Pos& from, const Pos& to) {
    string path;
    int dr_move = (to.r > from.r) ? 1 : -1;
    int dc_move = (to.c > from.c) ? 1 : -1;
    
    // 垂直移動
    for (int i = 0; i < abs(to.r - from.r); i++) {
        path += (dr_move == 1) ? 'D' : 'U';
    }
    // 水平移動
    for (int i = 0; i < abs(to.c - from.c); i++) {
        path += (dc_move == 1) ? 'R' : 'L';
    }
    return path;
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
    
    mt19937 rng;
    
    Solver() : best_move_count(INF), rng(42) {}
    
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
        
        cerr << "[DEBUG] Total Iterations: " << iteration << endl;
        output();
    }
    
    // マージグループを実行する操作列を生成
    void generate_merge_group_operations(const vector<int>& group, 
                                          const vector<bool>& group_visit_first) {
        if (group.size() == 1) {
            // 単独ペア
            int card = group[0];
            Pos p1 = input.card_pos[card].first;
            Pos p2 = input.card_pos[card].second;
            Pos first_pos = group_visit_first[0] ? p1 : p2;
            Pos second_pos = group_visit_first[0] ? p2 : p1;
            
            // 1枚目へ移動
            string path1 = get_path(current_pos, first_pos);
            for (char c : path1) {
                operations += c;
                move_count++;
            }
            current_pos = first_pos;
            operations += 'Z';
            
            // 2枚目へ移動
            string path2 = get_path(current_pos, second_pos);
            for (char c : path2) {
                operations += c;
                move_count++;
            }
            current_pos = second_pos;
            operations += 'Z';
            return;
        }
        
        // 複数ペアのマージ（再帰的にネスト）
        int outer_card = group[0];
        Pos o1 = input.card_pos[outer_card].first;
        Pos o2 = input.card_pos[outer_card].second;
        Pos outer_first = group_visit_first[0] ? o1 : o2;
        Pos outer_second = group_visit_first[0] ? o2 : o1;
        
        // 外側1枚目へ移動
        string path_to_outer = get_path(current_pos, outer_first);
        for (char c : path_to_outer) {
            operations += c;
            move_count++;
        }
        current_pos = outer_first;
        operations += 'Z';
        
        // 内側のグループを処理（再帰）
        vector<int> inner_group(group.begin() + 1, group.end());
        vector<bool> inner_visit_first(group_visit_first.begin() + 1, group_visit_first.end());
        generate_merge_group_operations(inner_group, inner_visit_first);
        
        // 外側2枚目へ移動
        string path_to_outer2 = get_path(current_pos, outer_second);
        for (char c : path_to_outer2) {
            operations += c;
            move_count++;
        }
        current_pos = outer_second;
        operations += 'Z';
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
    vector<int> find_proximity_group(int main_card, bool main_visit_first, 
                                     const vector<bool>& used, int max_depth = 3) {
        vector<int> group;
        group.push_back(main_card);
        
        // 現在の最も内側のペアの始点・終点（これを親として、さらに内側に挟めるものを探す）
        Pos current_start = main_visit_first ? input.card_pos[main_card].first : input.card_pos[main_card].second;
        Pos current_end = main_visit_first ? input.card_pos[main_card].second : input.card_pos[main_card].first;
        
        for (int depth = 1; depth < max_depth; depth++) {
            int best_inner = -1;
            int min_insertion_cost = INF;

            for (int card = 0; card < N * N / 2; card++) {
                if (used[card]) continue;
                
                bool already_in_group = false;
                for (int g : group) if (g == card) already_in_group = true;
                if (already_in_group) continue;
                
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                
                // 順方向 (current_start -> p1 -> p2 -> current_end)
                int cost1 = calc_insertion_cost(current_start, current_end, p1, p2);
                
                // 逆方向 (current_start -> p2 -> p1 -> current_end)
                int cost2 = calc_insertion_cost(current_start, current_end, p2, p1);
                
                if (cost1 < min_insertion_cost) {
                    min_insertion_cost = cost1;
                    best_inner = card;
                }
                if (cost2 < min_insertion_cost) {
                    min_insertion_cost = cost2;
                    best_inner = card;
                }
            }
            
            // 閾値: 完璧な包含(0)か、わずかな寄り道(2程度)を許容
            // 以前は近接和(<=6)だったが、今回は増加距離なので厳しめに0か2にする
            if (best_inner != -1 && min_insertion_cost <= 2) {
                group.push_back(best_inner);
                
                // 次のループのために、基準となる親を更新する
                // 最適な向きを再確認
                Pos p1 = input.card_pos[best_inner].first;
                Pos p2 = input.card_pos[best_inner].second;
                int cost1 = calc_insertion_cost(current_start, current_end, p1, p2);
                int cost2 = calc_insertion_cost(current_start, current_end, p2, p1);
                
                if (cost1 <= cost2) {
                    current_start = p1;
                    current_end = p2;
                } else {
                    current_start = p2;
                    current_end = p1;
                }
            } else {
                break;
            }
        }
        
        return group;
    }

    // ランダム要素を入れた1回の試行
    // deterministic=true の場合は完全な貪欲法（元のSolution 6）を実行
    void run_iteration(bool deterministic) {
        operations.clear();
        move_count = 0;
        current_pos = Pos(0, 0);
        vector<bool> used(N * N / 2, false);
        
        int remaining = N * N / 2;
        while (remaining > 0) {
            // 候補リストを作成 (Top-K Strategy)
            struct Candidate {
                int card;
                bool visit_first;
                int cost;
            };
            vector<Candidate> candidates;
            
            for (int card = 0; card < N * N / 2; card++) {
                if (used[card]) continue;
                
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                
                int cost1 = manhattan_dist(current_pos, p1);
                int cost2 = manhattan_dist(current_pos, p2);
                
                candidates.push_back({card, true, cost1});
                candidates.push_back({card, false, cost2});
            }
            
            // コストでソート
            sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                return a.cost < b.cost;
            });
            
            Candidate best = candidates[0];
            
            if (!deterministic) {
                // Top-Kからランダムに選択
                // ただし、コストが良いものを強く優先する (Geometric distribution like)
                int K = 4;
                int range = min((int)candidates.size(), K);
                
                // 確率的に上位を選ぶ: 50%で1位, 25%で2位, ... のような重み付け
                int pick_idx = 0;
                int r = uniform_int_distribution<int>(0, 100)(rng);
                
                if (range >= 2) {
                    if (r < 50) pick_idx = 0;       // 50%
                    else if (r < 80) pick_idx = 1;  // 30%
                    else if (r < 95) pick_idx = 2;  // 15%
                    else pick_idx = min(3, range-1); // 5%
                }
                
                best = candidates[pick_idx];
            }
            
            int best_card = best.card;
            bool best_visit_first = best.visit_first;
            
            // 近接グループを探す
            vector<int> merge_group = find_proximity_group(best_card, best_visit_first, used, 100);
            
            // 各ペアの訪問順を決定
            vector<bool> group_visit_first;
            group_visit_first.push_back(best_visit_first);
            
            Pos outer_start = best_visit_first ? input.card_pos[best_card].first : input.card_pos[best_card].second;
            Pos outer_end = best_visit_first ? input.card_pos[best_card].second : input.card_pos[best_card].first;

            for (size_t i = 1; i < merge_group.size(); i++) {
                int card = merge_group[i];
                Pos p1 = input.card_pos[card].first;
                Pos p2 = input.card_pos[card].second;
                
                int s_to_p1 = manhattan_dist(outer_start, p1);
                int e_to_p2 = manhattan_dist(outer_end, p2);
                int score1 = s_to_p1 + e_to_p2;
                
                int s_to_p2 = manhattan_dist(outer_start, p2);
                int e_to_p1 = manhattan_dist(outer_end, p1);
                int score2 = s_to_p2 + e_to_p1;
                
                group_visit_first.push_back(score1 <= score2);
            }
            
            generate_merge_group_operations(merge_group, group_visit_first);
            
            for (int card : merge_group) {
                used[card] = true;
                remaining--;
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
