#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <numeric>
#include <climits>
#include <set>

using namespace std;
using ll = long long;

// 時間管理
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
    for (int j = 1; j <= M; j++) {
        error += abs(S[j] - B[j-1]);
    }
    return error;
}

// 戦略:
// 1. 各山に1枚、ほぼ確実にベースとなる L 付近のカードを入れる (M枚)
// 2. 残りの差分 (0 ~ U-L) を、ビット表現のように 2べきのカード等で埋める
void large_card_init() {
    X.assign(N, 0); 
    S.assign(M + 1, 0);
    
    vector<bool> card_used(N, false);
    
    // ベースカード(値がLのもの) を探して各山に割り当てる
    // Aの生成時に最初のM枚をLにしている前提
    // もしそうでなくても大きい順に割り当てればOK
    
    // 便宜上、大きい順にソートしたインデックスを持つ
    vector<int> sorted_indices(N);
    iota(sorted_indices.begin(), sorted_indices.end(), 0);
    sort(sorted_indices.begin(), sorted_indices.end(), [&](int i, int j){
        return A[i] > A[j];
    });

    // M個の山に、大きいカードを貪欲に割り当て
    for(int j=1; j<=M; ++j) {
        int best_idx = -1;
        // 未使用の中で一番大きいものを使う (L想定)
        for(int k=0; k<N; ++k) {
            int idx = sorted_indices[k];
            if(!card_used[idx]) {
                best_idx = idx;
                break;
            }
        }
        
        if(best_idx != -1) {
            X[best_idx] = j;
            S[j] += A[best_idx];
            card_used[best_idx] = true;
        }
    }

    // 残りのカード (調整用) で隙間を埋める
    // 貪欲法: 残りのカードを、最もエラー削減効果が高い山に入れる
    
    vector<int> remaining_cards;
    for(int i=0; i<N; ++i) {
        if(!card_used[i]) remaining_cards.push_back(i);
    }
    
    // 調整用カードは大きい順に試したほうが、大きな隙間を埋めやすい
    sort(remaining_cards.begin(), remaining_cards.end(), [&](int a, int b){
        return A[a] > A[b];
    });

    for(int c : remaining_cards) {
        int best_pile = -1;
        ll best_reduction = 0;
        
        for(int j=1; j<=M; ++j) {
            ll current_err = abs(S[j] - B[j-1]);
            ll current_diff = B[j-1] - S[j]; // 正なら足りてない
            
            // 調整カードは基本正の数なので、足りてない時のみ入れる価値がある
            // (過剰な時に入れると悪化するだけ)
            if(current_diff > 0) {
                 ll new_diff = current_diff - A[c];
                 ll new_err = abs(new_diff);
                 ll reduction = current_err - new_err;
                 
                 if(reduction > 0 && reduction > best_reduction) {
                     best_reduction = reduction;
                     best_pile = j;
                 }
            }
        }
        
        if(best_pile != -1) {
            X[c] = best_pile;
            S[best_pile] += A[c];
            card_used[c] = true;
        }
    }
}

// 調整用SA
void simulated_annealing(Timer& timer) {
    ll current_error = calc_error();
    ll best_error = current_error;
    vector<int> best_X = X;
    vector<ll> best_S = S;
    
    const double start_temp = 2e12; // 初期温度 高め
    const double end_temp = 1e2;
    
    int iterations = 0;
    double temp = start_temp;
    const double ln_ratio = log(end_temp / start_temp);
    
    while (true) {
        iterations++;
        
        // 1024回に1回時間チェック
        if ((iterations & 1023) == 0) {
            if (timer.is_time_up()) break;
        }
        
        // 128回に1回温度更新
        if ((iterations & 127) == 0) {
            double progress = timer.progress();
            temp = start_temp * exp(ln_ratio * progress);
        }
        
        int op = rng.next_int(100);
        
        if (op < 60) { // Move
            int card = rng.next_int(N);
            int old_pile = X[card];
            int new_pile = rng.next_int(M + 1); 
            
            if (old_pile == new_pile) continue;
            
            ll delta = 0;
            ll val = A[card];
            // old_pile から抜ける影響
            if (old_pile > 0) {
                ll target = B[old_pile - 1];
                ll current_s = S[old_pile];
                delta += abs(current_s - val - target) - abs(current_s - target);
            }
            // new_pile に入る影響
            if (new_pile > 0) {
                ll target = B[new_pile - 1];
                ll current_s = S[new_pile];
                delta += abs(current_s + val - target) - abs(current_s - target);
            }
            
            if (delta <= 0 || rng.next_double() < exp(-(double)delta / temp)) {
                if (old_pile > 0) S[old_pile] -= val;
                if (new_pile > 0) S[new_pile] += val;
                X[card] = new_pile;
                current_error += delta;
                
                if (current_error < best_error) {
                    best_error = current_error;
                    best_X = X;
                    best_S = S;
                }
            }
        } else { // Swap
            int card1 = rng.next_int(N);
            int card2 = rng.next_int(N);
            if (card1 == card2) continue;
            
            int pile1 = X[card1];
            int pile2 = X[card2];
            if (pile1 == pile2) continue;
            
            ll delta = 0;
            ll v1 = A[card1];
            ll v2 = A[card2];
            
            // pile1 変化
            if(pile1 > 0) {
                ll target = B[pile1 - 1];
                ll current_s = S[pile1];
                delta += abs(current_s - v1 + v2 - target) - abs(current_s - target);
            }
            // pile2 変化
            if(pile2 > 0) {
                ll target = B[pile2 - 1];
                ll current_s = S[pile2];
                delta += abs(current_s - v2 + v1 - target) - abs(current_s - target);
            }

            if (delta <= 0 || rng.next_double() < exp(-(double)delta / temp)) {
                if (pile1 > 0) S[pile1] = S[pile1] - v1 + v2;
                if (pile2 > 0) S[pile2] = S[pile2] - v2 + v1;
                swap(X[card1], X[card2]);
                current_error += delta;
                
                if (current_error < best_error) {
                    best_error = current_error;
                    best_X = X;
                    best_S = S;
                }
            }
        }
    }
    
    X = best_X;
    S = best_S;
    
    cerr << "Iterations: " << iterations << ", Best Error: " << best_error << endl;
}

double start_temp = 2e12;
double end_temp = 1e2;

int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (argc >= 2) start_temp = atof(argv[1]);
    if (argc >= 3) end_temp = atof(argv[2]);
    
    cin >> N >> M >> L >> U;
    
    A.resize(N);
    
    // A生成戦略 (Hybrid: Power-of-2 + v16 Dist)
    int idx = 0;
    // Base cards
    for(; idx < M; ++idx) {
        A[idx] = L;
    }
    
    ll max_gap = U - L;
    
    // Small Powers 2^10 ... 2^29
    for(int s=0; s<3; ++s) {
        ll p = 1024; 
        while(p < 1000000000LL) {
             if(idx < N) A[idx++] = p;
             p *= 2;
        }
    }
    
    // Middle Powers 2^30 ... near max_gap
    for(int s=0; s<3; ++s) {
        ll p = 1073741824LL; // 2^30
        while(p <= max_gap) {
            if(idx < N) A[idx++] = p;
            if (max_gap / 2 < p) break;
            p *= 2;
        }
    }
    
    auto add_random = [&](int count, ll min_v, ll max_v) {
        for(int i=0; i<count && idx < N; ++i) {
            A[idx++] = min_v + rng.next_ll(max_v - min_v + 1);
        }
    };

    // v16 Distribution (Modified for Hybrid)
    // Reduce Top (1e12) to match v16 improvement (80 -> 60)
    // Boost Mid (1e10, 1e9) to match v16 improvement
    add_random(60, 1000000000000LL, max_gap);      // 10^12 ~ (v16: 60)
    add_random(60, 100000000000LL, 999999999999LL); // 10^11 ~ (v16: 60)
    add_random(80, 10000000000LL, 99999999999LL);   // 10^10 ~ (v16: 80 - Boost!)
    add_random(70, 1000000000LL, 9999999999LL);     // 10^9 ~ (v16: 70 - Boost!)
    add_random(60, 10000000LL, 999999999LL);        // 10^7 ~ (v16: 60 - Boost!)
    
    // Remaining (Micro)
    while(idx < N) {
        A[idx++] = 1000LL + rng.next_ll(9999999LL + 1);
    }
    
    for (int i = 0; i < N; i++) {
        cout << A[i] << (i + 1 < N ? " " : "\n");
    }
    cout.flush();
    
    B.resize(M);
    for (int j = 0; j < M; j++) {
        cin >> B[j];
    }
    
    vector<int> final_X;
    ll final_best_error = -1;

    int NUM_STARTS = 3;
    double time_per_start = 1.9 / NUM_STARTS;

    for(int attempt=0; attempt < NUM_STARTS; ++attempt) {
        // Reset state for each attempt
        Timer local_timer(time_per_start);
        large_card_init();
        
        simulated_annealing(local_timer);
        
        // Check if this run is better
        ll current_error = calc_error();
        if (final_best_error == -1 || current_error < final_best_error) {
            final_best_error = current_error;
            final_X = X;
        }
    }
    
    // Output best X
    for (int i = 0; i < N; i++) {
        cout << final_X[i] << (i + 1 < N ? " " : "\n");
    }
    cout.flush();
    
    // Debug info
    cerr << "Final Best Error: " << final_best_error << endl;
    
    return 0;
}
