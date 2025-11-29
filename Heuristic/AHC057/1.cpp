#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <queue>
#include <tuple>

using namespace std;

// 定数定義
int L;
int HALF_L;

class Timer {
public:
    Timer() { reset(); }
    void reset() { start_ = std::chrono::steady_clock::now(); }
    double elapsed() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

inline double wrap_coord(double val) {
    while (val >= L) val -= L;
    while (val < 0) val += L;
    return val;
}

inline double get_dist_sq(double x1, double y1, double x2, double y2) {
    double dx = std::abs(x1 - x2);
    double dy = std::abs(y1 - y2);
    if (dx > HALF_L) dx = L - dx;
    if (dy > HALF_L) dy = L - dy;
    return dx * dx + dy * dy;
}

double calc_contribution(int p_idx, const vector<int>& members, 
                         const vector<int>& pred_x, const vector<int>& pred_y) {
    double cost = 0;
    for (int other_idx : members) {
        if (other_idx == p_idx) continue;
        double dx = std::abs(pred_x[p_idx] - pred_x[other_idx]);
        double dy = std::abs(pred_y[p_idx] - pred_y[other_idx]);
        if (dx > HALF_L) dx = L - dx;
        if (dy > HALF_L) dy = L - dy;
        cost += dx * dx + dy * dy;
    }
    return cost;
}

vector<int> pick_initial_centers(int N, int M, const vector<int>& pred_x, const vector<int>& pred_y, mt19937& rng) {
    vector<int> centers;
    uniform_int_distribution<int> dist_N(0, N - 1);
    centers.push_back(dist_N(rng));

    for (int k = 1; k < M; ++k) {
        vector<double> min_dist_sq(N, 1e18);
        double total_dist_sq = 0;
        for (int i = 0; i < N; ++i) {
            for (int c_idx : centers) {
                double d = get_dist_sq(pred_x[i], pred_y[i], pred_x[c_idx], pred_y[c_idx]);
                if (d < min_dist_sq[i]) min_dist_sq[i] = d;
            }
            total_dist_sq += min_dist_sq[i];
        }
        uniform_real_distribution<double> dist_prob(0, total_dist_sq);
        double r = dist_prob(rng);
        int next_center = -1;
        double current_sum = 0;
        for (int i = 0; i < N; ++i) {
            current_sum += min_dist_sq[i];
            if (current_sum >= r) {
                next_center = i;
                break;
            }
        }
        if (next_center == -1) next_center = N - 1;
        centers.push_back(next_center);
    }
    return centers;
}

struct Command {
    int t, p1, p2;
};

struct SimResult {
    double total_cost;
    vector<Command> commands;
    vector<double> point_costs;
    vector<int> degrees;
};

SimResult run_simulation(
    int N, int T, int M, int K,
    const vector<double>& init_x, const vector<double>& init_y, 
    const vector<double>& init_vx, const vector<double>& init_vy,
    const vector<int>& target_group,
    mt19937& rng
) {
    SimResult res;
    res.total_cost = 0;
    res.point_costs.assign(N, 0.0);
    res.degrees.assign(N, 0);

    vector<bool> comp_active(N, true);
    vector<vector<int>> comp_members(N);
    for(int i=0; i<N; ++i) comp_members[i].push_back(i);
    vector<double> comp_vx = init_vx;
    vector<double> comp_vy = init_vy;
    
    // 現在位置 (更新用)
    vector<double> x = init_x;
    vector<double> y = init_y;

    vector<vector<int>> group_cids(M);
    for (int i = 0; i < N; ++i) group_cids[target_group[i]].push_back(i);

    vector<int> uf_parent(N);
    for(int i=0; i<N; ++i) uf_parent[i] = i;
    auto find_root = [&](auto&& self, int i) -> int {
        if (uf_parent[i] == i) return i;
        return uf_parent[i] = self(self, uf_parent[i]);
    };

    for (int t = 0; t < T; ++t) {
        double progress = (double)t / T;
        bool is_panic = (progress >= 0.85);

        double threshold_loose;
        double panic_threshold = 0;
        if (is_panic) {
            double p_panic = (progress - 0.85) / 0.15;
            double current_max = 300.0 + (L - 300.0) * p_panic;
            threshold_loose = current_max * current_max;
            double panic_dist = 500.0 + (L - 500.0) * (p_panic * p_panic);
            panic_threshold = panic_dist * panic_dist;
        } else {
            double cur_loose = 60.0 + 240.0 * (progress * progress);
            threshold_loose = cur_loose * cur_loose;
        }
        
        double threshold_strict = 0;
        if (!is_panic) {
            double base_strict = 10.0 + 30.0 * progress;
            threshold_strict = base_strict * base_strict;
        }

        for (int gid = 0; gid < M; ++gid) {
            vector<int> cids;
            for(int id : group_cids[gid]) {
                int r = find_root(find_root, id);
                if(comp_active[r]) {
                    bool found = false;
                    for(int existing : cids) if(existing == r) { found = true; break; }
                    if(!found) cids.push_back(r);
                }
            }
            group_cids[gid] = cids;
            if (cids.size() < 2) continue;

            struct Candidate {
                int cid1, cid2, p1, p2; 
                double dist_sq; 
                int priority_type; // 2:塊x塊, 1:塊x点, 0:点x点
                
                bool operator<(const Candidate& other) const {
                    if (priority_type != other.priority_type) return priority_type > other.priority_type;
                    return dist_sq < other.dist_sq;
                }
            };
            vector<Candidate> candidates;

            for (size_t i = 0; i < cids.size(); ++i) {
                int cid1 = cids[i];
                double cvx1 = comp_vx[cid1]; double cvy1 = comp_vy[cid1];
                int size1 = (int)comp_members[cid1].size();

                for (size_t j = i + 1; j < cids.size(); ++j) {
                    int cid2 = cids[j];
                    double cvx2 = comp_vx[cid2]; double cvy2 = comp_vy[cid2];
                    int size2 = (int)comp_members[cid2].size();

                    if (size1 + size2 > K) continue;

                    double min_d_local = 1e18; int lp1 = -1, lp2 = -1;
                    for (int p1_idx : comp_members[cid1]) {
                        for (int p2_idx : comp_members[cid2]) {
                            double d_sq = get_dist_sq(x[p1_idx], y[p1_idx], x[p2_idx], y[p2_idx]);
                            if (d_sq < min_d_local) { min_d_local = d_sq; lp1 = p1_idx; lp2 = p2_idx; }
                        }
                    }

                    double nx1 = wrap_coord(x[lp1] + cvx1); double ny1 = wrap_coord(y[lp1] + cvy1);
                    double nx2 = wrap_coord(x[lp2] + cvx2); double ny2 = wrap_coord(y[lp2] + cvy2);
                    double nd = get_dist_sq(nx1, ny1, nx2, ny2);
                    bool is_leaving = (nd > min_d_local);
                    
                    bool ok = false;
                    bool big1 = (size1 >= 5);
                    bool big2 = (size2 >= 5);
                    bool is_big_big = (big1 && big2);
                    
                    if (is_panic) {
                        if (t >= 980) ok = true;
                        else if (is_big_big) {
                            if (min_d_local <= 10000.0) ok = true;
                        }
                        else if (is_leaving) {
                            if (min_d_local <= panic_threshold) ok = true;
                        } 
                        else {
                            if (min_d_local <= 400.0) ok = true;
                        }
                    } else {
                        if (min_d_local <= 5000.0) ok = true;
                        else {
                            if (!is_leaving) {
                                if (min_d_local <= threshold_strict) ok = true;
                            }
                        }
                    }

                    if (ok) {
                        int p_type = 0;
                        if (is_big_big) p_type = 2;
                        else if (big1 || big2) p_type = 1;
                        
                        candidates.push_back({cid1, cid2, lp1, lp2, min_d_local, p_type});
                    }
                }
            }

            sort(candidates.begin(), candidates.end());

            for (const auto& cand : candidates) {
                int root1 = find_root(find_root, cand.cid1);
                int root2 = find_root(find_root, cand.cid2);
                if (root1 == root2) continue;
                
                if (comp_members[root1].size() + comp_members[root2].size() > K) continue;

                res.commands.push_back({t, cand.p1, cand.p2});
                double cost = sqrt(cand.dist_sq);
                res.total_cost += cost;
                res.point_costs[cand.p1] += cost;
                res.point_costs[cand.p2] += cost;
                res.degrees[cand.p1]++;
                res.degrees[cand.p2]++;

                double s1 = (double)comp_members[root1].size();
                double s2 = (double)comp_members[root2].size();
                double denom = s1 + s2;
                comp_vx[root1] = (s1 * comp_vx[root1] + s2 * comp_vx[root2]) / denom;
                comp_vy[root1] = (s1 * comp_vy[root1] + s2 * comp_vy[root2]) / denom;
                comp_members[root1].insert(comp_members[root1].end(), comp_members[root2].begin(), comp_members[root2].end());
                comp_active[root2] = false;
                uf_parent[root2] = root1;
            }
        }

        for(int cid=0; cid<N; ++cid) {
            if(comp_active[cid]) {
                double cvx = comp_vx[cid]; double cvy = comp_vy[cid];
                for(int midx : comp_members[cid]) {
                    x[midx] = wrap_coord(x[midx] + cvx);
                    y[midx] = wrap_coord(y[midx] + cvy);
                }
            }
        }
    }
    return res;
}

struct Assignment {
    double dist;
    int p_idx;
    int center_group_idx;
    bool operator>(const Assignment& other) const { return dist > other.dist; }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Timer timer;
    double TIME_LIMIT = 1.85; 

    int N, T, M, K;
    if (!(cin >> N >> T >> M >> K)) return 0;
    cin >> L;
    HALF_L = L / 2;

    vector<double> x(N), y(N), vx(N), vy(N);
    for (int i = 0; i < N; ++i) cin >> x[i] >> y[i] >> vx[i] >> vy[i];

    vector<int> pred_x(N), pred_y(N);
    for (int i = 0; i < N; ++i) {
        long long px = (long long)(x[i] + vx[i] * T) % L;
        if (px < 0) px += L;
        long long py = (long long)(y[i] + vy[i] * T) % L;
        if (py < 0) py += L;
        pred_x[i] = (int)px;
        pred_y[i] = (int)py;
    }

    mt19937 rng(42);
    
    double min_total_cost = 1e18;
    vector<Command> best_commands;

    // --- Multi-Start Loop ---
    int starts = 0;
    while (timer.elapsed() < TIME_LIMIT) {
        starts++;
        
        // 1. 新しい初期配置の生成 (K-means++)
        vector<int> center_indices = pick_initial_centers(N, M, pred_x, pred_y, rng);
        vector<int> target_group(N, -1);
        vector<int> group_counts(M, 0);
        priority_queue<Assignment, vector<Assignment>, greater<Assignment>> pq;

        for (int i = 0; i < N; ++i) {
            for (int g = 0; g < M; ++g) {
                int c_idx = center_indices[g];
                double d = get_dist_sq(pred_x[i], pred_y[i], pred_x[c_idx], pred_y[c_idx]);
                pq.push({d, i, g});
            }
        }

        int assigned = 0;
        while (assigned < N && !pq.empty()) {
            Assignment a = pq.top(); pq.pop();
            if (target_group[a.p_idx] != -1) continue;
            if (group_counts[a.center_group_idx] >= K) continue;
            target_group[a.p_idx] = a.center_group_idx;
            group_counts[a.center_group_idx]++;
            assigned++;
        }
        for(int i=0; i<N; ++i) {
            if(target_group[i] == -1) {
                for(int g=0; g<M; ++g) {
                    if(group_counts[g] < K) {
                        target_group[i] = g;
                        group_counts[g]++;
                        break;
                    }
                }
            }
        }

        // 2. 初期Swap最適化 (Quick Polish)
        // 回数は控えめにして、シミュレーション回数を稼ぐ
        {
            vector<vector<int>> group_members(M);
            for (int i = 0; i < N; ++i) group_members[target_group[i]].push_back(i);
            
            int optimize_steps = 10000; // 短く
            uniform_int_distribution<int> dist_M(0, M - 1);
            
            for(int k=0; k<optimize_steps; ++k) {
                int g1 = dist_M(rng); int g2 = dist_M(rng);
                if(g1 == g2 || group_members[g1].empty() || group_members[g2].empty()) continue;
                
                int idx1 = uniform_int_distribution<int>(0, (int)group_members[g1].size()-1)(rng);
                int idx2 = uniform_int_distribution<int>(0, (int)group_members[g2].size()-1)(rng);
                int p1 = group_members[g1][idx1];
                int p2 = group_members[g2][idx2];
                
                double cur1 = calc_contribution(p1, group_members[g1], pred_x, pred_y);
                double cur2 = calc_contribution(p2, group_members[g2], pred_x, pred_y);
                double new1 = calc_contribution(p1, group_members[g2], pred_x, pred_y); 
                double new2 = calc_contribution(p2, group_members[g1], pred_x, pred_y);
                
                if (new1 + new2 < cur1 + cur2) {
                    swap(group_members[g1][idx1], group_members[g2][idx2]);
                    target_group[p1] = g2; target_group[p2] = g1;
                }
            }
        }

        // 3. Refinement Loop (この配置で何度か粘る)
        // 1回シミュレーションして、ダメなところを直して...を数回繰り返す
        // あまりやりすぎると時間がなくなるので、5回程度
        int refinement_steps = 5;
        
        for(int r_step = 0; r_step < refinement_steps; ++r_step) {
            // 時間切れチェック
            if (timer.elapsed() > TIME_LIMIT) break;

            // シミュレーション実行
            SimResult res = run_simulation(N, T, M, K, x, y, vx, vy, target_group, rng);

            // ベスト更新
            if (res.total_cost < min_total_cost) {
                min_total_cost = res.total_cost;
                best_commands = res.commands;
            }

            // 悪い点を修正 (Mutation)
            vector<pair<double, int>> bad_points;
            for(int i=0; i<N; ++i) {
                // コストが高く、かつ次数が低い(0 or 1)点
                if (res.point_costs[i] > 0 && res.degrees[i] < 2) {
                    bad_points.push_back({res.point_costs[i], i});
                }
            }
            sort(bad_points.rbegin(), bad_points.rend());

            // 上位を救済
            int modify_count = min((int)bad_points.size(), 10);
            for(int i=0; i<modify_count; ++i) {
                int p_idx = bad_points[i].second;
                int old_g = target_group[p_idx];
                
                int best_new_g = old_g;
                double best_g_dist = 1e18;
                
                for(int g=0; g<M; ++g) {
                    if (g == old_g) continue;
                    vector<int> members;
                    for(int j=0; j<N; ++j) if(target_group[j] == g) members.push_back(j);
                    if(members.empty()) continue;
                    
                    double c = calc_contribution(p_idx, members, pred_x, pred_y);
                    if (c < best_g_dist) {
                        best_g_dist = c;
                        best_new_g = g;
                    }
                }
                
                if (best_new_g != old_g) {
                    int swap_target = -1;
                    double max_c = -1;
                    for(int j=0; j<N; ++j) {
                        if(target_group[j] == best_new_g) {
                            if (res.point_costs[j] > max_c) { 
                                max_c = res.point_costs[j];
                                swap_target = j;
                            }
                        }
                    }
                    if (swap_target == -1) {
                        vector<int> cands;
                        for(int j=0; j<N; ++j) if(target_group[j] == best_new_g) cands.push_back(j);
                        if(!cands.empty()) swap_target = cands[rng() % cands.size()];
                    }
                    if (swap_target != -1) {
                        target_group[p_idx] = best_new_g;
                        target_group[swap_target] = old_g;
                    }
                }
            }
        }
    }

    for (const auto& cmd : best_commands) {
        cout << cmd.t << " " << cmd.p1 << " " << cmd.p2 << "\n";
    }

    return 0;
}