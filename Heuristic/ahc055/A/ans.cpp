double TL = 1.99;
//#pragma GCC optimize("O3,Ofast,omit-frame-pointer,unroll-all-loops,tree-loop-vectorize,tree-slp-vectorize")
#pragma GCC optimize("O3,unroll-all-loops")
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <cmath>
#include <cstring>
#include <cmath>
#include <cassert>
#include <random>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <bitset>

int STANDARD = 1;
using namespace std;
#define F0(i,n) for (int i=0; i<n; i++)
#define F1(i,n) for (int i=1; i<=n; i++)
#define CL(a,x) memset(x, a, sizeof(x));
#define SZ(x) ((int)x.size())
const int inf = 1000000000;
const double pi = acos(-1.0);
typedef pair<int, int> pii;
typedef long long ll;
typedef unsigned long long ull;
const double EPS = 1e-9;
template<class A, class B>
ostream& operator<<(ostream& os, const pair<A, B>& p) { os << "(" << p.first << "," << p.second << ")"; return os; }
template<class A, class B, class C>
ostream& operator<<(ostream& os, const tuple<A, B, C>& p) { os << "(" << get<0>(p) << "," << get<1>(p) << "," << get<2>(p) << ")"; return os; }
istream& operator>>(istream& is, pii& p) { is>>p.first>>p.second; return is; }
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

#ifdef ATCODERX
inline ll GetTSC() {
    ll lo, hi;
    asm volatile ("rdtsc": "=a"(lo), "=d"(hi));
    return lo + (hi << 32);
}
inline double GetSeconds() {
    return GetTSC() / 3.0e9;
}
#else
chrono::system_clock::time_point init_time = chrono::system_clock::now();
inline double GetSeconds() {
     chrono::system_clock::time_point current_time = chrono::system_clock::now();
     double ret = chrono::duration_cast<std::chrono::nanoseconds>(current_time - init_time).count();
     return ret * 1e-9;
}
#endif

const int MAX_RAND = 1 << 30;
struct Rand {
    ll x, y, z, w, o;
    Rand() {}
    Rand(ll seed) { reseed(seed); o = 0; }
    inline void reseed(ll seed) { x = 0x498b3bc5 ^ seed; y = 0; z = 0; w = 0;  F0(i, 20) mix(); }
    inline void mix() { ll t = x ^ (x << 11); x = y; y = z; z = w; w = w ^ (w >> 19) ^ t ^ (t >> 8); }
    inline ll rand() { mix(); return x & (MAX_RAND - 1); }
    inline ll zh() { mix(); return x; }
    inline int nextInt(int n) { return rand() % n; }
    inline int nextInt(int L, int R) { return rand()%(R - L + 1) + L; }
    inline int nextBool() { if (o < 4) o = rand(); o >>= 2; return o & 1; }
    inline double nextDouble() { return rand() * 1.0 / MAX_RAND; }
    inline double nextDouble(double L, double R) { return L + (R - L) * nextDouble(); }
    template<class T> auto REL(T& v) { return v[nextInt(SZ(v))]; }
    template<class T> void RS(vector<T>& v) { F1(i, SZ(v)-1) { int j = nextInt(0, i); swap(v[i], v[j]); } }
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
Rand my(2029);
double saveTime;
double t_o[20];
ll c_o[20];
void Init() {
    saveTime = GetSeconds();
    F0(i, 20) t_o[i] = 0.0;
    F0(i, 20) c_o[i] = 0;
}
double Elapsed() { return GetSeconds() - saveTime; }
void Report() {
    double tmp = Elapsed();
    cerr << "-------------------------------------" << endl;
    cerr << "Elapsed time: " << tmp << " sec" << endl;
    double total = 0.0; F0(i, 20) { if (t_o[i] > 0) cerr << "t_o[" << i << "] = " << t_o[i] << endl; total += t_o[i]; } cerr << endl; //if (total > 0) cerr << "Total spent: " << total << endl;
    F0(i, 20) if (c_o[i] > 0) cerr << "c_o[" << i << "] = " << c_o[i] << endl;
    cerr << "-------------------------------------" << endl;
}
struct AutoTimer {
    int x;
    double t;
    AutoTimer(int x) : x(x) {
        t = Elapsed();
    }
    ~AutoTimer() {
        t_o[x] += Elapsed() - t;
    }
};
#define AT(i) AutoTimer a##i(i)
//#define AT(i)

// CONSTANTS
const int N = 256;
const int n = 200;
Rand rng[N], rng2[N];
ll bscore, score;
int H[N], C[N], h[N], c[N], HSUM;
int a[N][N];
vector<pair<int, pii>> bsol, sol;
vector<int> o;
vector<pii> oute[N], ine[N];
int f[N][N], bf[N][N];
pii olist[N][6], blist[N][6];
pii elist[N][N];
int on[N], bn[N], en[N];

const int LOGN = 1 << 16;
double logs[LOGN];

template<class T> T sqr(T x) { return x*x; }

void Prepare() {    
    HSUM = 0;
    F0(i, n) HSUM += H[i];
    F0(i, n) {
        vector<pii> v;
        F0(j, n) if (j != i && a[i][j] > 1) v.push_back({a[i][j], j});
        sort(v.rbegin(), v.rend());
        oute[i] = v;
    }
    F0(j, n) {
        vector<pii> v;
        F0(i, n) if (j != i && a[i][j] > 1) v.push_back({a[i][j], i});
        sort(v.rbegin(), v.rend());
        ine[j] = v;
    }
    int pw = 1.33;
    F0(i, n) {
        rng[i].reseed(i);
        rng2[i].reseed(i);
        vector<double> v, v2;
        F0(j, SZ(oute[i])) v.push_back(pow((oute[i][j].first - 1), pw));
        rng[i].PrepareWeightedSample(v);
        F0(j, SZ(ine[i])) v2.push_back(pow((ine[i][j].first - 1), pw));
        rng2[i].PrepareWeightedSample(v2);
    }
    bscore = inf;
    F0(i, LOGN) logs[i] = -log((i+0.5)/LOGN);
}

void UpdateBest() {
    if (score < bscore) {
        bscore = score;
        F0(i, n) {
            bn[i] = on[i];
            F0(j, on[i]) blist[i][j] = olist[i][j];
        }
    }
}

void PreSA() {
    score = HSUM;
    F0(i, n) {
        h[i] = H[i];
        c[i] = 0;
        on[i] = en[i] = 0;
    }
    CL(0, f);
}

void AddRev(int i, int j, int cnt) {
    F0(ii, en[i]) if (elist[i][ii].first == j) {
        elist[i][ii].second+=cnt;
        return;
    }
    elist[i][en[i]++] = {j, cnt};
}

void DelRev(int i, int j, int cnt) {
    F0(ii, en[i]) if (elist[i][ii].first == j) {
        if ((elist[i][ii].second -= cnt) == 0) {
            elist[i][ii] = elist[i][--en[i]];
        }
        return;
    }
    throw;
}

void AddEdge(int i, int j, int cnt) {
    AddRev(j, i, cnt);
    F0(ii, on[i]) if (olist[i][ii].first == j) {
        olist[i][ii].second += cnt;
        return;
    }
    olist[i][on[i]++] = {j, cnt};
}

void DelEdge(int i, int j, int cnt) {
    DelRev(j, i, cnt);
    F0(ii, on[i]) if (olist[i][ii].first == j) {
        if ((olist[i][ii].second -= cnt) == 0) {
            olist[i][ii] = olist[i][--on[i]];
        }
        return;
    }
    throw;
}

int q[N], qi, qn, vis[N], vz;
void BFS(int start, int finish = -1) {
    qn = qi = 0;
    q[qn++] = start;
    vis[start] = ++vz;
    F0(qi, qn) {
        int i = q[qi];
        //c_o[5] += on[i];
        F0(ii, on[i]) {
            int j = olist[i][ii].first;
            if (vis[j] != vz) {
                vis[j] = vz;
                if (j == finish) return;
                q[qn++] = j;
            }
        }
    }
}

void SA(double endTime) {
    int tot = 1, acc = 0;
    double T2 = 15.00, T1 = 0.00, TEMP = 0.0, r = 0.0;
    double delta = 0.0;
    int score_delta = 0, edge_delta = 0, used_delta = 0;

    int i = 0, j = 0, tp = 0, index = 0, adding = 0, cnt = 0;
    PR(score);

    Rand jrng(my.nextInt(100));
    vector<double> v;
    F0(i, n) v.push_back(pow(H[i], 1.0));
    jrng.PrepareWeightedSample(v);

    int itc = 10000000;
#define TIME_BASED
#ifdef TIME_BASED
    double startTime = Elapsed();
    itc = 1000000000;
#endif
    F0(iter, itc) {
        if ((iter & 127) == 0) {
#ifdef TIME_BASED
            r = (endTime - Elapsed()) / (endTime - startTime);
#else
            r = 1.0 * (itc - iter) / itc;
#endif
            if (r <= 0) { PR(iter); break; }
            TEMP = T1 + (T2 - T1) * r;
            //TEMP = pow(T1, 1-r) * pow(T2, r);
        }

        tp = my.nextInt(2);
        cnt = 1;

        if (tp == 0) {
            i = my.nextInt(n);
            if (c[i] == C[i]) {
                adding = 0;
                index = my.nextInt(on[i]);
                j = olist[i][index].first;
                if (my.nextInt(2)) cnt = olist[i][index].second;
            } else {
                adding = 1;
                if (on[i] && my.nextInt(4) == 0) {
                    index = my.nextInt(on[i]);
                    j = olist[i][index].first;
                } else {
                    index = rng[i].WeightedSample();
                    j = oute[i][index].second;
                }
            }
        } else {
            j = jrng.WeightedSample();
            if (h[j] > 0 || my.nextInt(8) == 0) {
                adding = 1;
                if (en[j] && my.nextInt(8) == 0) {
                    index = my.nextInt(en[j]);
                    i = elist[j][index].first;
                } else {
                    index = rng2[j].WeightedSample();
                    i = ine[j][index].second;
                }
                if (c[i] == C[i]) continue;
                if (h[j] >= 2 * a[i][j]) cnt = min(C[i] - c[i], h[j] / a[i][j]);
            } else {
                adding = 0;
                index = my.nextInt(en[j]);
                i = elist[j][index].first;
                if (my.nextInt(8) == 0) cnt = elist[j][index].second;
            }
        }

        c_o[1]++;

        score_delta = -max(h[j], 0);
        if (adding) {
            used_delta = cnt;
            if (c[i] + cnt == C[i]) used_delta++;
        } else {
            used_delta = -cnt;
            if (c[i] == C[i]) used_delta--;
        }
        edge_delta = 0;
        if (adding) {
            if (!f[i][j]) edge_delta = 1;
        } else {
            if (f[i][j] == cnt) edge_delta = -1;
        }
        if (adding) {
            score_delta += max(h[j] - a[i][j] * cnt, 0) + cnt;
        } else {
            score_delta += max(h[j] + a[i][j] * cnt, 0) - cnt;
        }

        delta = score_delta + used_delta * r * r * 0.1 + edge_delta * r * r * 15;

        if (delta > 0) tot++;
        if (delta <= 0 || delta <= TEMP * logs[my.nextInt(LOGN)]) {
            if (adding && !f[i][j]) {
                BFS(j, i);
                c_o[3]++;
                if (vis[i] == vz) {
                    c_o[4]++;
                    continue;
                }
            }

            score += score_delta;
            if (delta > 0) acc++;
            if (adding) {
                AddEdge(i, j, cnt);
                c[i]+=cnt;
                f[i][j]+=cnt;
                h[j] -= a[i][j]*cnt;
            } else {
                DelEdge(i, j, cnt);
                c[i]-=cnt;
                f[i][j]-=cnt;
                h[j] += a[i][j]*cnt;
            }
            UpdateBest();
        } else {
        }
    }
    if (1) {
        cerr << acc << "/" << tot << " "
             << 100.0 * acc / tot << " "
             << score << "/" << bscore << endl;
    }
}

void DFS(int i) {
    vis[i] = vz;
    F0(ii, bn[i]) {
        int j = blist[i][ii].first;
        if (vis[j] != vz) DFS(j);
    }
    o.push_back(i);
}

void Solve() {
    Prepare();

    int tries = 2;
    F1(x, tries) {
        PreSA();
        SA(TL * x / tries);
    }

    vz++;
    F0(i, n) if (vis[i] != vz) DFS(i);

    reverse(o.begin(), o.end());
    F0(i, n) h[i] = H[i];
    assert(SZ(o) == n);

    for (int i : o) {
        if (h[i] > 0) sol.push_back({h[i], {-1, i}});
        F0(ii, bn[i]) {
            int j = blist[i][ii].first;
            int cnt = blist[i][ii].second;
            F0(x, cnt) if (h[j] > 0) {
                h[j] -= a[i][j];
                sol.push_back({1, {i, j}});
            } else {
                bscore--;
            }
        }
    }
    bsol = sol;

    PR(bscore, HSUM + 1 - bscore);

    F0(i, SZ(bsol)) {
        auto p = bsol[i];
        F0(j, p.first) {
            cout << p.second.first << " " << p.second.second << endl;
        }
    }

    Report();
}


void ReadInput() {
    int tmp;
    cin >> tmp;
    F0(i, n) cin >> H[i];
    F0(i, n) cin >> C[i];
    F0(i, n) F0(j, n) cin >> a[i][j];
}

int main(int argc, char* argv[]) {
    Init();

    int seed1 = 0, seed2 = 0;
    if (argc>1) {
        if (argc == 2) {
            seed1 = seed2 = atoi(argv[1]);
        } else {
            seed1 = atoi(argv[1]);
            seed2 = atoi(argv[2]);
        }
        STANDARD=0;
    }

    if (STANDARD) {
        ReadInput();
        Solve();
        return 0;
    }

    for (int seed=seed1; seed<=seed2; seed++) {
        if (seed>=0 && seed<10000) {
            char inp[128];
            sprintf(inp, "in/%04d.txt", seed);
            char outp[128];
            sprintf(outp, "out/%04d.txt", seed);
            ignore = freopen(inp, "r", stdin);
            ignore = freopen(outp, "w", stdout);
            ReadInput();
            Solve();
            cerr << "Seed #" << seed << " ";
            cerr << bscore << endl;
            //cout << "Score would be " << bscore << endl;
        } else {
            // Generate
            throw;
            Rand my(seed);
        }
    }

    return 0;
}
