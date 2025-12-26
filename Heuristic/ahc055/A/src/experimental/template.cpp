#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <cmath>
#include <cstring>
#include <cassert>
#include <random>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <tuple>

using namespace std;

// ============================================================================
// Debug Macros & Output Overloads
// ============================================================================
#define F0(i,n) for (int i=0; i<n; i++)
#define F1(i,n) for (int i=1; i<=n; i++)
#define SZ(x) ((int)x.size())
#define CL(a,x) memset(x, a, sizeof(x));

template<class A, class B>
ostream& operator<<(ostream& os, const pair<A, B>& p) { os << "(" << p.first << "," << p.second << ")"; return os; }
template<class A, class B, class C>
ostream& operator<<(ostream& os, const tuple<A, B, C>& p) { os << "(" << get<0>(p) << "," << get<1>(p) << "," << get<2>(p) << ")"; return os; }
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
    os << "["; F0(i,SZ(v)) { if (i>0) os << ","; os << v[i]; } os << "]"; return os;
}
template<class T>
ostream& operator<<(ostream& os, const set<T>& v) {
    os << "{"; int f=1; for(auto i:v) { if(f)f=0;else os << ","; cerr << i; } os << "}"; return os;
}
template<class T, class R>
ostream& operator<<(ostream& os, const map<T,R>& v) {
    os << "{"; int f=1; for(auto i:v) { if(f)f=0;else os << ", "; cerr << i.first << ":" << i.second; } os << "}"; return os;
}
void print_all() { cerr << endl; }
template <typename H, typename... T>
void print_all(H head, T... tail) { cerr << " " << head; print_all(tail...); }

#ifdef LOCAL
#define PR(...) cerr << #__VA_ARGS__ << " =", print_all(__VA_ARGS__)
#else
#define PR(...)
#endif

// ============================================================================
// Timer / Benchmark (Universal: Mac M1/M2/M3 & AtCoder Linux)
// ============================================================================
// グローバルで開始時間を保持
// steady_clock はシステム時間を変更しても影響を受けないため計測に最適
auto init_time = std::chrono::steady_clock::now();

inline double GetSeconds() {
    auto now = std::chrono::steady_clock::now();
    // ナノ秒単位で差分を取得し、秒(double)に変換
    return std::chrono::duration<double>(now - init_time).count();
}

// ============================================================================
// AutoTimer for Profiling
// ============================================================================
// Usage: 
// double profile_times[20];
// { AT(0); ...code... } // adds elapsed time to profile_times[0]
double profile_times[20]; 
struct AutoTimer {
    int x;
    double t;
    AutoTimer(int x) : x(x) { t = GetSeconds(); }
    ~AutoTimer() { profile_times[x] += GetSeconds() - t; }
};
#define AT(i) AutoTimer a##i(i)

// ============================================================================
// Random Number Generator with Weighted Sampling (Walker's Alias-Method)
// ============================================================================
const int MAX_RAND = 1 << 30;
struct Rand {
    long long x, y, z, w, o;
    Rand() {}
    Rand(long long seed) { reseed(seed); o = 0; }
    inline void reseed(long long seed) { x = 0x498b3bc5 ^ seed; y = 0; z = 0; w = 0;  F0(i, 20) mix(); }
    inline void mix() { long long t = x ^ (x << 11); x = y; y = z; z = w; w = w ^ (w >> 19) ^ t ^ (t >> 8); }
    inline long long rand() { mix(); return x & (MAX_RAND - 1); }
    inline int nextInt(int n) { return rand() % n; }
    inline int nextInt(int L, int R) { return rand()%(R - L + 1) + L; }
    inline double nextDouble() { return rand() * 1.0 / MAX_RAND; }
    inline double nextDouble(double L, double R) { return L + (R - L) * nextDouble(); }
    
    // Weighted Sampling using Alias Method
    // O(N) setup, O(1) sampling
    int ws_n; vector<double> ws_prob; vector<int> ws_alias;
    inline void PrepareWeightedSample(const vector<double>& p) {
        ws_n = SZ(p); ws_prob.assign(ws_n, 0); ws_alias.assign(ws_n, 0);
        vector<double> sc(ws_n);
        vector<int> small, large;
        double sum = 0; F0(i, ws_n) sum += p[i];
        F0(i, ws_n) {
            sc[i] = p[i]/sum * ws_n; if (sc[i] < 1) small.push_back(i); else large.push_back(i);
        }
        while(!small.empty() && !large.empty()){
            int l = small.back(); small.pop_back();
            int g = large.back(); large.pop_back();
            ws_prob[l] = sc[l]; ws_alias[l] = g;
            sc[g] = sc[g] - (1 - sc[l]);
            if(sc[g] < 1) small.push_back(g); else large.push_back(g);
        }
        for(int g : large){ ws_prob[g]=1; ws_alias[g]=g; }
        for(int l : small){ ws_prob[l]=1; ws_alias[l]=l; }
    }
    inline int WeightedSample() { int i = nextInt(ws_n); return nextDouble() < ws_prob[i] ? i : ws_alias[i]; }
};

// ============================================================================
// Main Template
// ============================================================================
/*
int main(int argc, char* argv[]) {
    // Standard setup
    // Init();
    
    if (argc > 1) {
        // Local Batch Execution
        // Usage: ./a.out <start_seed> <end_seed>
        int seed1 = atoi(argv[1]);
        int seed2 = (argc == 2) ? seed1 : atoi(argv[2]);
        
        for (int seed = seed1; seed <= seed2; seed++) {
             // Redirect I/O
             char inp[128], outp[128];
             sprintf(inp, "in/%04d.txt", seed);
             sprintf(outp, "out/%04d.txt", seed);
             if (freopen(inp, "r", stdin) == NULL) break;
             freopen(outp, "w", stdout);

             // Reset Solution State
             // Solve();
             
             fprintf(stderr, "Seed #%d Done\n", seed);
        }
    } else {
        // Online Judge Mode
        // ReadInput();
        // Solve();
    }
    return 0;
}
*/
