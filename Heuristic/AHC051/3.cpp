#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <queue>
#include <tuple>
#include <algorithm>
#include <chrono>
#include <random>

// 周辺密度を計測する半径 (の2乗)
const double DENSITY_RADIUS_SQ = 2000.0 * 2000.0;
// 密度ペナルティの係数 C
const double DENSITY_PENALTY_COEFFICIENT = 0.1;


// 座標を扱うための構造体
struct Point {
    long long x, y;
};

// 幾何学関連の関数 (変更なし)
long long cross_product(Point a, Point b, Point c) { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }
int sign(long long val) { return (val > 0) - (val < 0); }
bool on_segment(Point p, Point q, Point r) { return (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y)); }
bool segments_intersect(Point p1, Point p2, Point q1, Point q2) {
    int o1 = sign(cross_product(p1, p2, q1)), o2 = sign(cross_product(p1, p2, q2)), o3 = sign(cross_product(q1, q2, p1)), o4 = sign(cross_product(q1, q2, p2));
    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && on_segment(p1, q1, p2)) return true; if (o2 == 0 && on_segment(p1, q2, p2)) return true;
    if (o3 == 0 && on_segment(q1, p1, q2)) return true; if (o4 == 0 && on_segment(q1, p2, q2)) return true;
    return false;
}

// エントロピーと情報利得の計算 (変更なし)
double entropy(const std::vector<double>& probs) {
    double total_prob = 0;
    for (double p : probs) total_prob += p;
    if (total_prob < 1e-9) return 0;
    double h = 0;
    for (double p : probs) {
        if (p > 1e-9) {
            double normalized_p = p / total_prob;
            h -= normalized_p * log2(normalized_p);
        }
    }
    return h;
}
double calculate_info_gain(const std::vector<double>& current_probs, int sorter_k, int N, const std::vector<std::vector<double>>& p_mat) {
    double current_h = entropy(current_probs);
    std::vector<double> probs1(N), probs2(N);
    double total_p1 = 0, total_p2 = 0;
    for (int i = 0; i < N; ++i) {
        probs1[i] = current_probs[i] * p_mat[sorter_k][i];
        probs2[i] = current_probs[i] * (1.0 - p_mat[sorter_k][i]);
        total_p1 += probs1[i]; total_p2 += probs2[i];
    }
    double total_prob = total_p1 + total_p2;
    if (total_prob < 1e-9) return 0;
    double h1 = entropy(probs1), h2 = entropy(probs2);
    double expected_h = (total_p1 / total_prob) * h1 + (total_p2 / total_prob) * h2;
    return current_h - expected_h;
}

long long distSq(Point p1, Point p2) {
    long long dx = p1.x - p2.x;
    long long dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    int N, M, K;
    std::cin >> N >> M >> K;

    std::vector<Point> disposal_coords(N);
    for (int i = 0; i < N; ++i) std::cin >> disposal_coords[i].x >> disposal_coords[i].y;

    std::vector<Point> sorter_coords(M);
    for (int i = 0; i < M; ++i) std::cin >> sorter_coords[i].x >> sorter_coords[i].y;

    std::vector<std::vector<double>> p_mat(K, std::vector<double>(N));
    for (int i = 0; i < K; ++i) for (int j = 0; j < N; ++j) std::cin >> p_mat[i][j];

    std::vector<int> d(N);
    std::vector<int> disposal_indices(N);
    std::iota(disposal_indices.begin(), disposal_indices.end(), 0);
    std::sort(disposal_indices.begin(), disposal_indices.end(), [&](int a, int b) {
        if (disposal_coords[a].y != disposal_coords[b].y) return disposal_coords[a].y < disposal_coords[b].y;
        return disposal_coords[a].x < disposal_coords[b].x;
    });
    for (int i = 0; i < N; ++i) d[disposal_indices[i]] = i;

    int s = -1;
    std::vector<int> c_k(M, -1), c_v1(M, -1), c_v2(M, -1);
    std::vector<bool> sorter_loc_used(M, false);
    std::vector<std::pair<Point, Point>> belts;

    using P_Queue_Element = std::tuple<double, int, int, std::vector<double>>;
    std::priority_queue<P_Queue_Element> q;

    std::vector<double> initial_probs(N, 1.0);
    q.emplace(entropy(initial_probs), -1, 1, initial_probs);

    const double PENALTY_THRESHOLD = 0.02;

    while (!q.empty()) {
        auto [current_entropy, parent_loc_idx, exit_num, current_probs] = q.top();
        q.pop();

        Point source_p = (parent_loc_idx == -1) ? Point{0, 5000} : sorter_coords[parent_loc_idx];
        double sum_p = 0.0, max_p = 0.0; int max_p_idx = -1;
        for (int i = 0; i < N; ++i) {
            sum_p += current_probs[i];
            if (current_probs[i] > max_p) {
                max_p = current_probs[i];
                max_p_idx = i;
            }
        }
        if (sum_p < 1e-9) continue;

        int best_dest_node_id = -1;

        double penalty = (sum_p > 0) ? (sum_p - max_p) / sum_p : 0;
        if (penalty < PENALTY_THRESHOLD) {
            int target_disposal_loc = -1;
            for(int i=0; i<N; ++i) if(d[i] == max_p_idx) target_disposal_loc = i;
            
            if (target_disposal_loc != -1) {
                Point dest_p = disposal_coords[target_disposal_loc];
                bool intersects = false;
                for (const auto& belt : belts) if (segments_intersect(source_p, dest_p, belt.first, belt.second)) { intersects = true; break; }
                if (!intersects) best_dest_node_id = target_disposal_loc;
            }
        }

        if (best_dest_node_id == -1) {
            double best_score = -1.0; int best_sorter_loc = -1; int best_sorter_type = -1;
            for (int m = 0; m < M; ++m) {
                if (sorter_loc_used[m]) continue;
                Point dest_p = sorter_coords[m];
                bool intersects = false;
                for (const auto& belt : belts) if (segments_intersect(source_p, dest_p, belt.first, belt.second)) { intersects = true; break; }
                if (intersects) continue;
                
                double dist = std::hypot(source_p.x - dest_p.x, source_p.y - dest_p.y);
                if (dist < 1e-9) continue;

                // ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                // ★ 対策: 周辺密度ペナルティの計算 ★
                // ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                int nearby_unused_count = 0;
                for (int other_m = 0; other_m < M; ++other_m) {
                    if (m == other_m || sorter_loc_used[other_m]) continue;
                    if (distSq(dest_p, sorter_coords[other_m]) < DENSITY_RADIUS_SQ) {
                        nearby_unused_count++;
                    }
                }
                // ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                
                for (int k_type = 0; k_type < K; ++k_type) {
                    double gain = calculate_info_gain(current_probs, k_type, N, p_mat);
                    
                    // ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    // ★ 対策: ペナルティを考慮したスコア計算 ★
                    // ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    double penalty_factor = 1.0 + DENSITY_PENALTY_COEFFICIENT * nearby_unused_count;
                    double score = gain / (dist * penalty_factor);
                    
                    if (score > best_score) { best_score = score; best_sorter_loc = m; best_sorter_type = k_type; }
                }
            }
            if (best_sorter_loc != -1) {
                best_dest_node_id = N + best_sorter_loc;
                sorter_loc_used[best_sorter_loc] = true;
                c_k[best_sorter_loc] = best_sorter_type;
                
                std::vector<double> probs1(N), probs2(N);
                for (int i = 0; i < N; ++i) {
                    probs1[i] = current_probs[i] * p_mat[best_sorter_type][i];
                    probs2[i] = current_probs[i] * (1.0 - p_mat[best_sorter_type][i]);
                }
                q.emplace(entropy(probs1), best_sorter_loc, 1, probs1);
                q.emplace(entropy(probs2), best_sorter_loc, 2, probs2);
            }
        }
        
        if (best_dest_node_id == -1) {
            long long min_dist_sq = -1; int fallback_dest_id = -1;
            for (int i = 0; i < N; ++i) {
                Point dest_p = disposal_coords[i];
                bool intersects = false;
                for (const auto& belt : belts) if (segments_intersect(source_p, dest_p, belt.first, belt.second)) { intersects = true; break; }
                if (intersects) continue;
                long long d_sq = distSq(source_p, dest_p);
                if (fallback_dest_id == -1 || d_sq < min_dist_sq) { min_dist_sq = d_sq; fallback_dest_id = i; }
            }
            best_dest_node_id = (fallback_dest_id != -1) ? fallback_dest_id : max_p_idx;
        }
        
        Point dest_p = (best_dest_node_id < N) ? disposal_coords[best_dest_node_id] : sorter_coords[best_dest_node_id - N];
        belts.push_back({source_p, dest_p});
        if (parent_loc_idx == -1) s = best_dest_node_id;
        else { if (exit_num == 1) c_v1[parent_loc_idx] = best_dest_node_id; else c_v2[parent_loc_idx] = best_dest_node_id; }
    }

    // 出力
    for (int i = 0; i < N; ++i) std::cout << d[i] << (i == N - 1 ? "" : " ");
    std::cout << std::endl;
    std::cout << s << std::endl;
    for (int i = 0; i < M; ++i) {
        if (c_k[i] == -1) std::cout << -1 << std::endl;
        else {
            if (c_v1[i] == -1) c_v1[i] = 0; 
            if (c_v2[i] == -1) c_v2[i] = 0;
            std::cout << c_k[i] << " " << c_v1[i] << " " << c_v2[i] << std::endl;
        }
    }
    return 0;
}