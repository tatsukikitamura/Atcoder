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
    double t0 = 2000.0;
    double t1 = 5.0;
    int max_opt_len = 20000; // max length of 2-opt reversal
    int prob_2opt = 35;
    int prob_swap = 42;
    double w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0, w8 = 0;
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

    void init_evalA() {
        evalA.resize(M);
        for (int u = 0; u < M; u++) {
            int r = u / N, c = u % N;
            double d_center = pow(r - N/2.0, 2) + pow(c - N/2.0, 2);
            double d_corner = min(r, N-1-r)*min(r, N-1-r) + min(c, N-1-c)*min(c, N-1-c);
            double is_bd = (r == 0 || r == N-1 || c == 0 || c == N-1) ? 1.0 : 0.0;
            evalA[u] = A[u] + params.w1 * d_center + params.w2 * d_corner + params.w3 * (A[u] * A[u] * 1e-3) + params.w4 * is_bd;
        }
    }

    inline double get_edgeW(int u, int v) const {
        int r1 = u / N, c1 = u % N;
        int r2 = v / N, c2 = v % N;
        double cost = params.w5 * abs(A[u] - A[v]) + params.w6 * ((r1 != r2 && c1 != c2) ? 1.0 : 0.0) + params.w7 * min(A[u], A[v]);
        double dc = pow(r1 - N/2.0, 2) + pow(c1 - N/2.0, 2) - (pow(r2 - N/2.0, 2) + pow(c2 - N/2.0, 2));
        return cost + params.w8 * abs(dc);
    }

    // King-move check between cell indices
    inline bool king_adj(int c1, int c2) const {
        int r1 = c1 / N, col1 = c1 % N;
        int r2 = c2 / N, col2 = c2 % N;
        return max(abs(r1 - r2), abs(col1 - col2)) == 1;
    }

    void read_input() {
        cin >> N;
        M = N * N;
        A.resize(M);
        for (int i = 0; i < M; i++) cin >> A[i];

        // Precompute adjacencies
        adj.resize(M);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int u = i * N + j;
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
        for (int k = 0; k < M; k++) {
            prefA[k + 1] = prefA[k] + A[path[k]];
            prefWA[k + 1] = prefWA[k] + (long long)k * A[path[k]];
            prefEvalA[k + 1] = prefEvalA[k] + evalA[path[k]];
            prefEvalWA[k + 1] = prefEvalWA[k] + (double)k * evalA[path[k]];
        }
    }

    void update_prefix(int l, int r) {
        long long old_prefWA_r1 = prefWA[r + 1];
        double old_prefEvalWA_r1 = prefEvalWA[r + 1];
        for (int k = l; k <= r; k++) {
            prefA[k + 1] = prefA[k] + A[path[k]];
            prefWA[k + 1] = prefWA[k] + (long long)k * A[path[k]];
            prefEvalA[k + 1] = prefEvalA[k] + evalA[path[k]];
            prefEvalWA[k + 1] = prefEvalWA[k] + (double)k * evalA[path[k]];
        }
        long long dWA = prefWA[r + 1] - old_prefWA_r1;
        double dEvalWA = prefEvalWA[r + 1] - old_prefEvalWA_r1;
        if (dWA != 0 || abs(dEvalWA) > 1e-9) {
            for (int k = r + 1; k < M; k++) {
                prefWA[k + 1] += dWA;
                prefEvalWA[k + 1] += dEvalWA;
            }
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
        build_prefix();
        long long current_true_score = prefWA[M];
        double current_eval_score = prefEvalWA[M];
        for (int k = 0; k < M - 1; k++) current_eval_score += get_edgeW(path[k], path[k+1]);
        long long best_score = current_true_score;
        vector<int> best_path = path;

        double t0 = params.t0;
        double t1 = params.t1;
        double temp = t0;
        double log_t = log(t1 / t0);

        int iter = 0;
        int max_len = params.max_opt_len;

        while (true) {
            iter++;
            if ((iter & 1023) == 0) {
                build_prefix();
                current_eval_score = prefEvalWA[M];
                for (int k = 0; k < M - 1; k++) current_eval_score += get_edgeW(path[k], path[k+1]);
                current_true_score = prefWA[M];
            }
            if ((iter & 255) == 0) {
                double elapsed = timer.elapsed();
                if (elapsed >= params.time_limit) break;
                double progress = elapsed / params.time_limit;
                temp = t0 * exp(log_t * progress);
            }

            // 1. Neighbor selection
            int n_type = rng.next_int(0, 99);

            if (n_type < params.prob_2opt) {
                // --- 2-opt (Targeted) ---
                int l = rng.next_int(1, M - 2);
                int prev_cell = path[l - 1];

                int r_cand_count = adj[prev_cell].size();
                int r_idx = rng.next_int(0, r_cand_count - 1);
                int cand_cell = adj[prev_cell][r_idx];
                int r = pos[cand_cell];

                if (r <= l || r >= M - 1 || r - l > max_len) continue;
                if (!king_adj(path[r + 1], path[l])) continue;

                // Delta update O(1)
                long long sumA = prefA[r + 1] - prefA[l];
                long long sumWA = prefWA[r + 1] - prefWA[l];
                long long true_delta = (long long)(r + l) * sumA - 2LL * sumWA;

                double sumEvalA = prefEvalA[r + 1] - prefEvalA[l];
                double sumEvalWA = prefEvalWA[r + 1] - prefEvalWA[l];
                double eval_delta_node = (r + l) * sumEvalA - 2.0 * sumEvalWA;
                double eval_delta_edge = - get_edgeW(path[l-1], path[l]) - get_edgeW(path[r], path[r+1])
                                         + get_edgeW(path[l-1], path[r]) + get_edgeW(path[l], path[r+1]);
                double eval_delta = eval_delta_node + eval_delta_edge;

                // Metropolis
                if (eval_delta > 0 || rng.next_double() < exp(eval_delta / temp)) {
                    current_true_score += true_delta;
                    current_eval_score += eval_delta;
                    reverse(path.begin() + l, path.begin() + r + 1);
                    for (int k = l; k <= r; k++) pos[path[k]] = k;
                    update_prefix(l, r);
                    if (current_true_score > best_score) {
                        best_score = current_true_score;
                        best_path = path;
                    }
                }
            } else if (n_type < params.prob_2opt + params.prob_swap) {
                // --- Swap adjacent elements ---
                // Try swapping path[i] and path[i+1]
                int i = rng.next_int(1, M - 3); // avoid ends for simplicity
                int p1 = path[i], p2 = path[i + 1];

                // Connectivity after swap:
                // path[i-1] -> p2 -> p1 -> path[i+2]
                if (!king_adj(path[i - 1], p2) || !king_adj(p1, path[i + 2])) continue;
                // Note: p1 and p2 must be king_adj, which they are (currently path[i]->path[i+1])

                long long true_delta = A[p1] - A[p2];
                double eval_delta_node = evalA[p1] - evalA[p2];
                double eval_delta_edge = - get_edgeW(path[i-1], p1) - get_edgeW(p2, path[i+2])
                                         + get_edgeW(path[i-1], p2) + get_edgeW(p1, path[i+2]);
                double eval_delta = eval_delta_node + eval_delta_edge;

                if (eval_delta > 0 || rng.next_double() < exp(eval_delta / temp)) {
                    current_true_score += true_delta;
                    current_eval_score += eval_delta;
                    swap(path[i], path[i + 1]);
                    pos[p2] = i;
                    pos[p1] = i + 1;
                    update_prefix(i, i + 1);
                    if (current_true_score > best_score) {
                        best_score = current_true_score;
                        best_path = path;
                    }
                }
            } else {
                // --- Single-cell Or-opt (Move cell i to after j) ---
                int i = rng.next_int(1, M - 2);
                int j = rng.next_int(1, M - 2);
                if (i == j || i == j + 1) continue;

                int cell = path[i];
                // Removing i requires path[i-1] adj path[i+1]
                if (!king_adj(path[i - 1], path[i + 1])) continue;

                // Inserting element `cell` after position j in the *original* array indices:
                // Case 1: j < i. We shift [j+1 .. i-1] to the right. 
                //   Insert `cell` at index j+1.
                //   Requires: path[j] adj cell, cell adj path[j+1]
                
                // Case 2: j > i. We shift [i+1 .. j] to the left.
                //   Insert `cell` at index j.
                //   Requires: path[j] adj cell, cell adj path[j+1]

                if (!king_adj(path[j], cell) || !king_adj(cell, path[j + 1])) continue;

                long long true_delta = 0;
                double eval_delta_node = 0;
                double eval_delta_edge = - get_edgeW(path[i-1], path[i]) - get_edgeW(path[i], path[i+1]) + get_edgeW(path[i-1], path[i+1])
                                         - get_edgeW(path[j], path[j+1]) + get_edgeW(path[j], path[i]) + get_edgeW(path[i], path[j+1]);
                int min_mod, max_mod;
                if (j < i) {
                    long long sumA = prefA[i] - prefA[j + 1];
                    true_delta = sumA + (long long)(j + 1 - i) * A[cell];
                    double sumEvalA = prefEvalA[i] - prefEvalA[j + 1];
                    eval_delta_node = sumEvalA + (j + 1 - i) * evalA[cell];
                    min_mod = j + 1; max_mod = i;
                } else {
                    long long sumA = prefA[j + 1] - prefA[i + 1];
                    true_delta = (long long)(j - i) * A[cell] - sumA;
                    double sumEvalA = prefEvalA[j + 1] - prefEvalA[i + 1];
                    eval_delta_node = (j - i) * evalA[cell] - sumEvalA;
                    min_mod = i; max_mod = j;
                }
                double eval_delta = eval_delta_node + eval_delta_edge;

                if (eval_delta > 0 || rng.next_double() < exp(eval_delta / temp)) {
                    current_true_score += true_delta;
                    current_eval_score += eval_delta;
                    
                    if (j < i) {
                        for (int k = i; k > j + 1; k--) {
                            path[k] = path[k - 1];
                            pos[path[k]] = k;
                        }
                        path[j + 1] = cell;
                        pos[cell] = j + 1;
                    } else {
                        for (int k = i; k < j; k++) {
                            path[k] = path[k + 1];
                            pos[path[k]] = k;
                        }
                        path[j] = cell;
                        pos[cell] = j;
                    }
                    update_prefix(min_mod, max_mod);

                    if (current_true_score > best_score) {
                        best_score = current_true_score;
                        best_path = path;
                    }
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
