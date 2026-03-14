/**
 * AHC062 - King's Tour
 * Simulated Annealing w/ Optuna support
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <random>
using namespace std;

// ---------------------------------------------------------------------------
// Parameters (Can be overridden by cmd args for Optuna)
// ---------------------------------------------------------------------------
struct Params {
    double time_limit = 2.85;
    double t0 = 1844.45;
    double t1 = 0.02;
    int max_opt_len = 19510; // max length of 2-opt reversal
    int prob_2opt = 100;
    int prob_swap = 0;
    double w1 = -70.23, w2 = 11.21, w3 = 96.77, w4 = 99.76, w5 = 78.16, w6 = -33.36, w7 = 45.51, w8 = -38.85;
};

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
class Timer {
public:
    chrono::high_resolution_clock::time_point start_time;
    double time_limit;

    Timer(double limit_sec) : time_limit(limit_sec) {
        start_time = chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start_time).count();
    }

    bool has_time() const { return elapsed() < time_limit; }
};

// ---------------------------------------------------------------------------
// Fast Random (Xorshift)
// ---------------------------------------------------------------------------
struct Xorshift {
    uint64_t x = 88172645463325252ULL;
    inline uint64_t next() { x ^= x << 13; x ^= x >> 7; return x ^= x << 17; }
    inline int next_int(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }
    inline double next_double() { return (double)next() / UINT64_MAX; }
};

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------
class Solver {
public:
    int N;
    int M; // N*N
    vector<long long> A; // A[i*N+j]
    vector<int> path;    // path[k] = cell index (i*N+j)
    Params params;
    Xorshift rng;
    vector<int> pos;     // pos[cell] = day when that cell is visited
    vector<vector<int>> adj;
    vector<long long> prefA;  // prefA[k] = sum(A[path[0..k-1]])
    vector<long long> prefWA; // prefWA[k] = sum(i * A[path[0..k-1]])
    vector<double> evalA;
    vector<double> prefEvalA;
    vector<double> prefEvalWA;
    vector<double> dist_center;
    vector<int> cell_r;
    vector<int> cell_c;
    vector<uint8_t> next_edge_idx; // edge idx for (path[k] -> path[k+1])
    vector<vector<double>> pre_edgeW; // pre_edgeW[u][neighbor_idx]

    static constexpr int PREFIX_BLOCK = 256;
    int prefix_block_count = 0;
    vector<long long> prefWA_block_add;
    vector<double> prefEvalWA_block_add;

    inline int pref_block(int idx) const { return idx / PREFIX_BLOCK; }

    void init_prefix_blocks() {
        prefix_block_count = (M + 1 + PREFIX_BLOCK - 1) / PREFIX_BLOCK;
        prefWA_block_add.assign(prefix_block_count, 0);
        prefEvalWA_block_add.assign(prefix_block_count, 0.0);
    }

    inline long long get_prefWA(int idx) const {
        return prefWA[idx] + prefWA_block_add[pref_block(idx)];
    }

    inline double get_prefEvalWA(int idx) const {
        return prefEvalWA[idx] + prefEvalWA_block_add[pref_block(idx)];
    }

    inline void set_prefWA(int idx, long long actual) {
        int b = pref_block(idx);
        prefWA[idx] = actual - prefWA_block_add[b];
    }

    inline void set_prefEvalWA(int idx, double actual) {
        int b = pref_block(idx);
        prefEvalWA[idx] = actual - prefEvalWA_block_add[b];
    }

    inline void add_suffix_prefWA(int l, int r, long long delta) {
        if (l > r || delta == 0) return;
        int bl = pref_block(l), br = pref_block(r);
        if (bl == br) {
            for (int i = l; i <= r; i++) prefWA[i] += delta;
            return;
        }
        int end_bl = min(r, (bl + 1) * PREFIX_BLOCK - 1);
        for (int i = l; i <= end_bl; i++) prefWA[i] += delta;
        for (int b = bl + 1; b < br; b++) prefWA_block_add[b] += delta;
        int start_br = br * PREFIX_BLOCK;
        for (int i = start_br; i <= r; i++) prefWA[i] += delta;
    }

    inline void add_suffix_prefEvalWA(int l, int r, double delta) {
        if (l > r || abs(delta) <= 1e-15) return;
        int bl = pref_block(l), br = pref_block(r);
        if (bl == br) {
            for (int i = l; i <= r; i++) prefEvalWA[i] += delta;
            return;
        }
        int end_bl = min(r, (bl + 1) * PREFIX_BLOCK - 1);
        for (int i = l; i <= end_bl; i++) prefEvalWA[i] += delta;
        for (int b = bl + 1; b < br; b++) prefEvalWA_block_add[b] += delta;
        int start_br = br * PREFIX_BLOCK;
        for (int i = start_br; i <= r; i++) prefEvalWA[i] += delta;
    }

    inline int edge_idx_in_adj(int u, int v) const {
        const auto& nbrs = adj[u];
        for (int i = 0; i < (int)nbrs.size(); i++) {
            if (nbrs[i] == v) return i;
        }
        return -1;
    }

    inline double get_edgeW_idx(int u, int idx) const {
        return pre_edgeW[u][idx];
    }

    void build_next_edge_idx() {
        next_edge_idx.assign(max(0, M - 1), 0);
        for (int k = 0; k < M - 1; k++) {
            int idx = edge_idx_in_adj(path[k], path[k + 1]);
            next_edge_idx[k] = (idx >= 0 ? (uint8_t)idx : 0);
        }
    }

    void update_next_edge_idx_range(int l, int r) {
        int start = max(0, l - 1);
        int end = min(M - 2, r);
        for (int k = start; k <= end; k++) {
            int idx = edge_idx_in_adj(path[k], path[k + 1]);
            next_edge_idx[k] = (idx >= 0 ? (uint8_t)idx : 0);
        }
    }

    void init_evalA() {
        evalA.resize(M);
        dist_center.resize(M);
        for (int u = 0; u < M; u++) {
            int r = u / N, c = u % N;
            double dc = pow(r - N/2.0, 2) + pow(c - N/2.0, 2);
            dist_center[u] = dc;
            double d_corner = min(r, N-1-r)*min(r, N-1-r) + min(c, N-1-c)*min(c, N-1-c);
            double is_bd = (r == 0 || r == N-1 || c == 0 || c == N-1) ? 1.0 : 0.0;
            evalA[u] = A[u] + params.w1 * dc + params.w2 * d_corner + params.w3 * (A[u] * A[u] * 1e-3) + params.w4 * is_bd;
        }

        pre_edgeW.resize(M);
        for (int u = 0; u < M; u++) {
            pre_edgeW[u].resize(adj[u].size());
            int r1 = cell_r[u], c1 = cell_c[u];
            for (int i = 0; i < (int)adj[u].size(); i++) {
                int v = adj[u][i];
                int r2 = cell_r[v], c2 = cell_c[v];
                double cost = params.w5 * abs(A[u] - A[v]) + params.w6 * ((r1 != r2 && c1 != c2) ? 1.0 : 0.0) + params.w7 * min(A[u], A[v]);
                double dc_diff = dist_center[u] - dist_center[v];
                pre_edgeW[u][i] = cost + params.w8 * abs(dc_diff);
            }
        }
    }

    inline double get_edgeW(int u, int v) const {
        const auto& nbrs = adj[u];
        for (int i = 0; i < (int)nbrs.size(); i++) {
            if (nbrs[i] == v) return pre_edgeW[u][i];
        }
        return 0.0;
    }

    inline bool king_adj(int c1, int c2) const {
        int dr = cell_r[c1] - cell_r[c2];
        if (dr < 0) dr = -dr;
        int dc = cell_c[c1] - cell_c[c2];
        if (dc < 0) dc = -dc;
        return max(dr, dc) == 1;
    }

    void read_input() {
        cin >> N;
        M = N * N;
        A.resize(M);
        for (int i = 0; i < M; i++) cin >> A[i];
        cell_r.resize(M);
        cell_c.resize(M);

        // Precompute adjacencies
        adj.resize(M);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int u = i * N + j;
                cell_r[u] = i;
                cell_c[u] = j;
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
                            adj[u].push_back(ni * N + nj);
                        }
                    }
                }
            }
        }
    }

    // Greedy initialization: prefer smaller A values, use Warnsdorff's to avoid dead ends
    void init_greedy() {
        vector<pair<long long, int>> sorted_A;
        for (int i = 0; i < M; i++) sorted_A.push_back({A[i], i});
        sort(sorted_A.begin(), sorted_A.end());

        // Try starting from cells with small A values
        for (int attempt = 0; attempt < 10; attempt++) {
            int start_cell = sorted_A[attempt].second;
            path.assign(M, -1);
            pos.assign(M, -1);
            vector<bool> visited(M, false);

            int curr = start_cell;
            path[0] = curr;
            pos[curr] = 0;
            visited[curr] = true;

            bool success = true;
            for (int k = 1; k < M; k++) {
                int best_next = -1;
                long long best_score = 3e18; // safe upper bound

                for (int nxt : adj[curr]) {
                    if (!visited[nxt]) {
                        // Warnsdorff's heuristic: count unvisited neighbors
                        long long deg = 0;
                        for (int nn : adj[nxt]) {
                            if (!visited[nn]) deg++;
                        }
                        // Priority: fewer unvisited neighbors (avoid dead ends), then smaller A
                        long long score = deg * 10000000000LL + A[nxt];
                        if (score < best_score) {
                            best_score = score;
                            best_next = nxt;
                        }
                    }
                }

                if (best_next == -1) {
                    success = false;
                    break;
                }

                curr = best_next;
                path[k] = curr;
                pos[curr] = k;
                visited[curr] = true;
            }

            if (success) {
                return;
            }
        }

        // Fallback to snake initialization if greedy fails.
        cerr << "Greedy initialization failed. Fallback to snake." << endl;
        init_snake();
    }

    void build_prefix() {
        prefA.assign(M + 1, 0);
        prefWA.assign(M + 1, 0);
        prefEvalA.assign(M + 1, 0);
        prefEvalWA.assign(M + 1, 0);
        fill(prefWA_block_add.begin(), prefWA_block_add.end(), 0);
        fill(prefEvalWA_block_add.begin(), prefEvalWA_block_add.end(), 0.0);
        for (int k = 0; k < M; k++) {
            prefA[k + 1] = prefA[k] + A[path[k]];
            prefWA[k + 1] = prefWA[k] + (long long)k * A[path[k]];
            prefEvalA[k + 1] = prefEvalA[k] + evalA[path[k]];
            prefEvalWA[k + 1] = prefEvalWA[k] + (double)k * evalA[path[k]];
        }
    }

    void update_prefix(int l, int r, long long true_delta, double eval_delta_node) {
        long long curA = prefA[l];
        double curEvalA = prefEvalA[l];
        long long curWA = get_prefWA(l);
        double curEvalWA = get_prefEvalWA(l);

        for (int k = l; k <= r; k++) {
            curA += A[path[k]];
            curEvalA += evalA[path[k]];
            curWA += (long long)k * A[path[k]];
            curEvalWA += (double)k * evalA[path[k]];

            prefA[k + 1] = curA;
            prefEvalA[k + 1] = curEvalA;
            set_prefWA(k + 1, curWA);
            set_prefEvalWA(k + 1, curEvalWA);
        }

        if (true_delta != 0) {
            add_suffix_prefWA(r + 2, M, true_delta);
        }
        if (abs(eval_delta_node) > 1e-12) {
            add_suffix_prefEvalWA(r + 2, M, eval_delta_node);
        }
    }

    // Snake path initialization (robust fallback)
    void init_snake() {
        path.resize(M);
        pos.resize(M);
        int k = 0;
        for (int r = 0; r < N; r++) {
            if (r % 2 == 0) {
                for (int c = 0; c < N; c++) path[k++] = r * N + c;
            } else {
                for (int c = N - 1; c >= 0; c--) path[k++] = r * N + c;
            }
        }
        for (int k = 0; k < M; k++) pos[path[k]] = k;
    }

    long long calc_score() const {
        long long v = 0;
        for (int k = 0; k < M; k++) v += (long long)k * A[path[k]];
        return v;
    }

    void solve() {
        Timer timer(params.time_limit);

        init_greedy(); // Try greedy first, fallback to snake
        init_evalA();
        init_prefix_blocks();
        build_prefix();
        build_next_edge_idx();
        long long current_true_score = get_prefWA(M);
        double current_eval_score = get_prefEvalWA(M);
        for (int k = 0; k < M - 1; k++) {
            current_eval_score += get_edgeW_idx(path[k], next_edge_idx[k]);
        }
        long long best_score = current_true_score;
        vector<int> best_path = path;

        double t0 = params.t0;
        double t1 = params.t1;
        double temp = t0;
        double log_t = log(t1 / t0);

        int iter = 0;
        int max_len = params.max_opt_len;
        constexpr int REBUILD_PERIOD = 4096;
        constexpr int TEMP_UPDATE_PERIOD = 512;

        while (true) {
            iter++;
            if ((iter & (REBUILD_PERIOD - 1)) == 0) {
                build_prefix();
                build_next_edge_idx();
                current_eval_score = get_prefEvalWA(M);
                for (int k = 0; k < M - 1; k++) {
                    current_eval_score += get_edgeW_idx(path[k], next_edge_idx[k]);
                }
                current_true_score = get_prefWA(M);
            }
            if ((iter & (TEMP_UPDATE_PERIOD - 1)) == 0) {
                double elapsed = timer.elapsed();
                if (elapsed >= params.time_limit) break;
                double progress = elapsed / params.time_limit;
                temp = t0 * exp(log_t * progress);
            }

            // 1. Neighbor selection
            // --- 2-opt (Targeted) ---
            int l = rng.next_int(1, M - 2);
            int prev_cell = path[l - 1];

            int valid_rs[8];
            uint8_t valid_prev_r_edge_idx[8];
            uint8_t valid_r_edge_idx[8];
            int valid_count = 0;
            for (int i = 0; i < (int)adj[prev_cell].size(); i++) {
                int r = pos[adj[prev_cell][i]];
                if (r <= l || r >= M - 1 || r - l > max_len) continue;
                int idx_l_r1 = edge_idx_in_adj(path[l], path[r + 1]);
                if (idx_l_r1 >= 0) {
                    valid_rs[valid_count++] = r;
                    valid_prev_r_edge_idx[valid_count - 1] = (uint8_t)i;
                    valid_r_edge_idx[valid_count - 1] = (uint8_t)idx_l_r1;
                }
            }
            if (valid_count == 0) continue;
            int picked = rng.next_int(0, valid_count - 1);
            int r = valid_rs[picked];
            uint8_t idx_prev_r = valid_prev_r_edge_idx[picked];
            uint8_t idx_l_r1 = valid_r_edge_idx[picked];

            // Delta update O(1)
            long long sumA = prefA[r + 1] - prefA[l];
            long long sumWA = get_prefWA(r + 1) - get_prefWA(l);
            long long true_delta = (long long)(r + l) * sumA - 2LL * sumWA;

            double sumEvalA = prefEvalA[r + 1] - prefEvalA[l];
            double sumEvalWA = get_prefEvalWA(r + 1) - get_prefEvalWA(l);
            double eval_delta_node = (double)(r + l) * sumEvalA - 2.0 * sumEvalWA;
            double eval_delta_edge = - get_edgeW_idx(path[l - 1], next_edge_idx[l - 1])
                                     - get_edgeW_idx(path[r], next_edge_idx[r])
                                     + get_edgeW_idx(path[l - 1], idx_prev_r)
                                     + get_edgeW_idx(path[l], idx_l_r1);
            double eval_delta = eval_delta_node + eval_delta_edge;

            // Metropolis
            bool accept = false;
            if (eval_delta >= 0.0) {
                accept = true;
            } else {
                double ratio = eval_delta / temp;
                // Very unlikely moves are pruned to avoid expensive exp() calls.
                if (ratio >= -20.0) {
                    accept = (rng.next_double() < exp(ratio));
                }
            }

            if (accept) {
                current_true_score += true_delta;
                current_eval_score += eval_delta;
                reverse(path.begin() + l, path.begin() + r + 1);
                for (int k = l; k <= r; k++) pos[path[k]] = k;
                update_next_edge_idx_range(l, r);
                update_prefix(l, r, true_delta, eval_delta_node);
                if (current_true_score > best_score) {
                    best_score = current_true_score;
                    best_path = path;
                }
            }
        }

        path = best_path;
        cerr << "SA Iters: " << iter << ", Best Score: " << best_score << endl;
    }

    void output() {
        for (int k = 0; k < M; k++) {
            cout << path[k] / N << " " << path[k] % N << "\n";
        }
    }
};

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Solver solver;

    // Parse arguments for Optuna tuning
    // Usage: ./main --t0 2000 --t1 5 --limit 2.85 --maxlen 15000
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--t0" && i + 1 < argc) solver.params.t0 = stod(argv[++i]);
        else if (arg == "--t1" && i + 1 < argc) solver.params.t1 = stod(argv[++i]);
        else if (arg == "--limit" && i + 1 < argc) solver.params.time_limit = stod(argv[++i]);
        else if (arg == "--maxlen" && i + 1 < argc) solver.params.max_opt_len = stoi(argv[++i]);
        else if (arg == "--p2opt" && i + 1 < argc) solver.params.prob_2opt = stoi(argv[++i]);
        else if (arg == "--pswap" && i + 1 < argc) solver.params.prob_swap = stoi(argv[++i]);
        else if (arg == "--w1" && i + 1 < argc) solver.params.w1 = stod(argv[++i]);
        else if (arg == "--w2" && i + 1 < argc) solver.params.w2 = stod(argv[++i]);
        else if (arg == "--w3" && i + 1 < argc) solver.params.w3 = stod(argv[++i]);
        else if (arg == "--w4" && i + 1 < argc) solver.params.w4 = stod(argv[++i]);
        else if (arg == "--w5" && i + 1 < argc) solver.params.w5 = stod(argv[++i]);
        else if (arg == "--w6" && i + 1 < argc) solver.params.w6 = stod(argv[++i]);
        else if (arg == "--w7" && i + 1 < argc) solver.params.w7 = stod(argv[++i]);
        else if (arg == "--w8" && i + 1 < argc) solver.params.w8 = stod(argv[++i]);
    }

    solver.read_input();
    solver.solve();
    solver.output();

    return 0;
}
