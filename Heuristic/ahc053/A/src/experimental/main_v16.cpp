#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <numeric>
#include <climits>

using namespace std;
using ll = long long;

class Timer {
    chrono::high_resolution_clock::time_point start_time;
    double time_limit;
public:
    Timer(double limit = 1.9) : time_limit(limit) {
        start_time = chrono::high_resolution_clock::now();
    }
    double elapsed() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start_time).count();
    }
    bool is_time_up() const {
        return elapsed() >= time_limit;
    }
    double progress() const {
        return min(1.0, elapsed() / time_limit);
    }
};

class Xorshift {
    uint32_t x = 123456789, y = 362436069, z = 521288629, w = 88675123;
public:
    uint32_t next() {
        uint32_t t = x ^ (x << 11);
        x = y; y = z; z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }
    int next_int(int n) { return next() % n; }
    ll next_ll() {
        return (((ll)next() << 32) | next()) & 0x7FFFFFFFFFFFFFFFLL;
    }
    ll next_ll(ll mod) { return next_ll() % mod; }
    double next_double() { return next() / 4294967296.0; }
};

int N, M;
ll L, U;
vector<ll> A, B;
vector<int> X;
vector<ll> S;
Xorshift rng;

ll calc_error() {
    ll error = 0;
    for (int j = 1; j <= M; j++) error += abs(S[j] - B[j - 1]);
    return error;
}

// Stable deterministic initialization
void deterministic_init() {
    X.assign(N, 0);
    S.assign(M + 1, 0);
    vector<bool> card_used(N, false);
    for (int j = 1; j <= M; j++) {
        int best_idx = -1;
        ll min_err = -1;
        for (int i = 0; i < N; i++) {
            if (card_used[i]) continue;
            ll err = abs(A[i] - (B[j - 1] - S[j]));
            if (min_err == -1 || err < min_err) {
                min_err = err;
                best_idx = i;
            }
        }
        if (best_idx != -1) {
            X[best_idx] = j;
            S[j] += A[best_idx];
            card_used[best_idx] = true;
        }
    }
    vector<int> remaining;
    for (int i = 0; i < N; i++) if (!card_used[i]) remaining.push_back(i);
    sort(remaining.begin(), remaining.end(), [&](int a, int b) { return A[a] > A[b]; });
    for (int i : remaining) {
        int best_p = 0;
        ll best_reduction = 0;
        for (int j = 1; j <= M; j++) {
            ll cur_err = abs(S[j] - B[j - 1]);
            ll diff = B[j - 1] - S[j];
            if (diff > 0) {
                ll new_err = abs(diff - A[i]);
                ll reduction = cur_err - new_err;
                if (reduction > best_reduction) {
                    best_reduction = reduction;
                    best_p = j;
                }
            }
        }
        if (best_p > 0) {
            X[i] = best_p;
            S[best_p] += A[i];
        }
    }
}

// Exploratory randomized initialization
void randomized_init() {
    X.assign(N, 0);
    S.assign(M + 1, 0);
    vector<bool> card_used(N, false);
    vector<int> pile_order(M);
    iota(pile_order.begin(), pile_order.end(), 1);
    for(int i = M-1; i > 0; --i) swap(pile_order[i], pile_order[rng.next_int(i+1)]);
    
    for (int j : pile_order) {
        int best_idx = -1;
        ll min_err = -1;
        for (int i = 0; i < N; i++) {
            if (card_used[i]) continue;
            ll err = abs(A[i] - (B[j - 1] - S[j]));
            if (min_err == -1 || err < min_err) {
                min_err = err;
                best_idx = i;
            }
        }
        if (best_idx != -1) {
            X[best_idx] = j;
            S[j] += A[best_idx];
            card_used[best_idx] = true;
        }
    }
    vector<int> remaining;
    for (int i = 0; i < N; i++) if (!card_used[i]) remaining.push_back(i);
    sort(remaining.begin(), remaining.end(), [&](int a, int b) { return A[a] > A[b]; });
    for(int i = M-1; i > 0; --i) swap(pile_order[i], pile_order[rng.next_int(i+1)]);
    for (int i : remaining) {
        int best_p = 0;
        ll best_reduction = 0;
        for (int j : pile_order) {
            ll cur_err = abs(S[j] - B[j - 1]);
            ll diff = B[j - 1] - S[j];
            if (diff > 0) {
                ll new_err = abs(diff - A[i]);
                ll reduction = cur_err - new_err;
                if (reduction > best_reduction) {
                    best_reduction = reduction;
                    best_p = j;
                }
            }
        }
        if (best_p > 0) {
            X[i] = best_p;
            S[best_p] += A[i];
        }
    }
}

struct Result {
    vector<int> X;
    ll error;
};

Result simulated_annealing(double limit_seconds) {
    Timer timer(limit_seconds);
    ll current_error = calc_error();
    ll best_local_error = current_error;
    vector<int> best_local_X = X;
    const double start_temp = 2e12, end_temp = 1e2;
    const double ln_ratio = log(end_temp / start_temp);
    int iterations = 0;
    double temp = start_temp;
    while (true) {
        iterations++;
        if ((iterations & 1023) == 0) {
            if (timer.is_time_up()) break;
        }
        if ((iterations & 127) == 0) temp = start_temp * exp(ln_ratio * timer.progress());
        int op = rng.next_int(100);
        if (op < 60) {
            int card = rng.next_int(N), old_p = X[card], new_p = rng.next_int(M + 1);
            if (old_p == new_p) continue;
            ll val = A[card], delta = 0;
            if (old_p > 0) delta += abs(S[old_p] - val - B[old_p - 1]) - abs(S[old_p] - B[old_p - 1]);
            if (new_p > 0) delta += abs(S[new_p] + val - B[new_p - 1]) - abs(S[new_p] - B[new_p - 1]);
            if (delta <= 0 || rng.next_double() < exp(-(double)delta / temp)) {
                if (old_p > 0) S[old_p] -= val;
                if (new_p > 0) S[new_p] += val;
                X[card] = new_p;
                current_error += delta;
                if (current_error < best_local_error) { best_local_error = current_error; best_local_X = X; }
            }
        } else {
            int c1 = rng.next_int(N), c2 = rng.next_int(N);
            if (c1 == c2) continue;
            int p1 = X[c1], p2 = X[c2];
            if (p1 == p2) continue;
            ll v1 = A[c1], v2 = A[c2], delta = 0;
            if (p1 > 0) delta += abs(S[p1] - v1 + v2 - B[p1 - 1]) - abs(S[p1] - B[p1 - 1]);
            if (p2 > 0) delta += abs(S[p2] - v2 + v1 - B[p2 - 1]) - abs(S[p2] - B[p2 - 1]);
            if (delta <= 0 || rng.next_double() < exp(-(double)delta / temp)) {
                if (p1 > 0) S[p1] = S[p1] - v1 + v2;
                if (p2 > 0) S[p2] = S[p2] - v2 + v1;
                swap(X[c1], X[c2]);
                current_error += delta;
                if (current_error < best_local_error) { best_local_error = current_error; best_local_X = X; }
            }
        }
    }
    cerr << "Iterations: " << iterations << ", Final Error: " << best_local_error << endl;
    return {best_local_X, best_local_error};
}

void generate_A() {
    A.resize(N);
    int idx = 0;
    for (; idx < M; ++idx) A[idx] = L;
    ll max_gap = U - L;
    auto add_random = [&](int count, ll min_v, ll max_v) {
        for (int i = 0; i < count && idx < N; ++i) A[idx++] = min_v + rng.next_ll(max_v - min_v + 1);
    };
    // v16 Distribution
    add_random(60, 1000000000000LL, max_gap);         // Top (Was 80)
    add_random(60, 100000000000LL, 999999999999LL);   // High-Mid (Same)
    add_random(80, 10000000000LL, 99999999999LL);     // Mid (Was 60) -> NEEDED
    add_random(70, 1000000000LL, 9999999999LL);       // Low-Mid (Was 60)
    add_random(60, 10000000LL, 999999999LL);         // Precision (Was 50)
    
    // Remaining micro cards
    while (idx < N) A[idx++] = 1000LL + rng.next_ll(9999999LL + 1);
    
    for (int i = 0; i < N; i++) cout << A[i] << (i + 1 < N ? " " : "\n");
    cout.flush();
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    if (!(cin >> N >> M >> L >> U)) return 0;
    generate_A();
    B.resize(M);
    for (int j = 0; j < M; j++) cin >> B[j];
    
    vector<int> final_X(N, 0);
    ll final_best_error = LLONG_MAX;
    double time_per_start = 1.95 / 2.0;

    // Start 1: Stable Deterministic
    deterministic_init();
    Result res1 = simulated_annealing(time_per_start);
    final_best_error = res1.error;
    final_X = res1.X;

    // Start 2: Elite Randomized
    ll best_init_err = LLONG_MAX;
    vector<int> elite_X;
    vector<ll> elite_S;
    Timer elite_timer(0.05);
    for(int t=0; t<200; ++t) {
        randomized_init();
        ll err = calc_error();
        if(err < best_init_err) { best_init_err = err; elite_X = X; elite_S = S; }
        if(elite_timer.is_time_up()) break;
    }
    X = elite_X; S = elite_S;
    Result res2 = simulated_annealing(time_per_start - elite_timer.elapsed());
    if (res2.error < final_best_error) { final_best_error = res2.error; final_X = res2.X; }

    for (int i = 0; i < N; i++) cout << final_X[i] << (i + 1 < N ? " " : "\n");
    cerr << "Final Best Error: " << final_best_error << endl;
    return 0;
}
