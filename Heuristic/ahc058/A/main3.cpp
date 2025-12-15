#pragma GCC optimize("Ofast")
#pragma GCC target("avx2")

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <chrono>

using namespace std;

using ll = long long;
using ld = double; 

// =============================================================================
// 定数・設定
// =============================================================================
constexpr int N = 10;
constexpr int L = 4;
constexpr int T = 500;

// ビームサーチ設定
constexpr int BEAM_WIDTH = 1500;
constexpr int SIM_LENGTH = 15;
constexpr ld FUTURE_DECAY = 0.3;
constexpr double TIME_LIMIT = 1.9;
constexpr double BEAM_NARROW_START = 1.6;
constexpr int MIN_BEAM_WIDTH = 100;

// ボーナス設定
constexpr ld LEVEL3_ACTION_BONUS = 1;

// =============================================================================
// グローバル変数
// =============================================================================
ll A[N];
ll C[L][N];

// =============================================================================
// 状態構造体
// =============================================================================
struct State {
    ll apples;
    ll P[L][N]; // 強化回数
    ll B[L][N]; // 稼働台数
    ld raw_score;
    
    State() : apples(0), raw_score(0) {
        memset(P, 0, sizeof(P));
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                B[i][j] = 1;
            }
        }
    }
    
    ll getCost(int level, int id) const {
        return C[level][id] * (P[level][id] + 1);
    }
    
    bool canUpgrade(int level, int id) const {
        return apples >= getCost(level, id);
    }
    
    void upgrade(int level, int id) {
        apples -= getCost(level, id);
        P[level][id]++;
    }
    
    void produce() {
        for (int j = 0; j < N; j++) {
            apples += A[j] * B[0][j] * P[0][j];
        }
        for (int i = 1; i < L; i++) {
            for (int j = 0; j < N; j++) {
                B[i - 1][j] += B[i][j] * P[i][j];
            }
        }
    }
    
    // 数式ベースの高速評価関数
    void evaluate(int remaining_turns) {
        int sim_len = (remaining_turns < SIM_LENGTH) ? remaining_turns : SIM_LENGTH;
        ld sim_apples = (ld)apples;
        
        ll sim_B[L][N];
        memcpy(sim_B, B, sizeof(B));
        
        // 1. 短期シミュレーション
        for (int t = 0; t < sim_len; t++) {
            for (int j = 0; j < N; j++) {
                sim_apples += (ld)A[j] * sim_B[0][j] * P[0][j];
            }
            for (int i = 1; i < L; i++) {
                for (int j = 0; j < N; j++) {
                    sim_B[i - 1][j] += sim_B[i][j] * P[i][j];
                }
            }
        }
        
        // 2. 長期予測
        int rem = remaining_turns - sim_len;
        if (rem > 0) {
            ld prod_rate = 0;
            for (int j = 0; j < N; j++) {
                prod_rate += (ld)A[j] * sim_B[0][j] * P[0][j];
            }
            sim_apples += prod_rate * rem * FUTURE_DECAY;
            
            for (int i = 1; i < L; i++) {
                // ピラミッド構造が崩れている（下位が0）場合は評価しない
                bool chain_ok = true;
                for(int k=0; k<i; ++k) { // i=1ならk=0, i=2ならk=0,1
                     // 簡易チェック: 少なくとも1つ以上強化されていないと連鎖しない
                     // (本来はBが増えるのでP>0でなくても良いが、P>0を推奨する)
                     if (P[k][0] == 0 && P[k][1] == 0 && P[k][2] == 0 && P[k][3] == 0 &&
                         P[k][4] == 0 && P[k][5] == 0 && P[k][6] == 0 && P[k][7] == 0 &&
                         P[k][8] == 0 && P[k][9] == 0) {
                         // 全IDでレベルkが死んでたら、その上のレベルiも死ぬ（概算）
                         // 厳密にはIDごとのチェックだが、高速化のためざっくり評価
                     }
                }

                // レベル別の重み付け
                ld level_weight = 0.5;
                if (i == 3) level_weight = 20.0; // Level 3は高評価
                else if (i == 2) level_weight = 1.5;

                for (int j = 0; j < N; j++) {
                    if (P[i][j] == 0) continue;
                    
                    ld val = (ld)A[j] * sim_B[i][j] * P[i][j];
                    ld mult = 1;
                    for (int k = 0; k < i; k++) {
                        mult *= rem / (ld)(5 + k);
                    }
                    sim_apples += val * mult * level_weight;
                }
            }
        }
        
        raw_score = sim_apples;
    }
};

struct Node {
    short parent_idx;
    signed char action_level;
    signed char action_id;
};

struct Action {
    int level;
    int id;
    bool isNoop() const { return level == -1; }
};

class Timer {
    chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(chrono::high_resolution_clock::now()) {}
    double elapsed() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start).count();
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n_in, l_in, t_in;
    ll k_in;
    if (!(cin >> n_in >> l_in >> t_in >> k_in)) return 0;
    
    for (int j = 0; j < N; j++) cin >> A[j];
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) cin >> C[i][j];
    }
    
    vector<vector<Node>> history(T + 1);
    history[0].reserve(1);
    history[0].push_back({-1, -1, -1});
    
    vector<State> cur_beam;
    cur_beam.reserve(BEAM_WIDTH * 2);
    
    State initial_state;
    initial_state.apples = k_in;
    initial_state.evaluate(T);
    cur_beam.push_back(initial_state);
    
    Timer timer;
    
    for (int turn = 0; turn < T; turn++) {
        double elapsed = timer.elapsed();
        int width = BEAM_WIDTH;
        
        if (elapsed > BEAM_NARROW_START) {
            double ratio = (TIME_LIMIT - elapsed) / (TIME_LIMIT - BEAM_NARROW_START);
            width = max(MIN_BEAM_WIDTH, (int)(BEAM_WIDTH * ratio));
        }
        
        int remaining = T - turn;
        int eval_remaining = remaining - 1;
        
        vector<State> next_beam;
        vector<Node> next_nodes;
        next_beam.reserve(cur_beam.size() * (2 + L * N / 2)); 
        next_nodes.reserve(cur_beam.size() * (2 + L * N / 2));
        
        for (int idx = 0; idx < (int)cur_beam.size(); idx++) {
            const State& state = cur_beam[idx];
            
            // 1. 何もしない (Wait)
            {
                next_beam.push_back(state);
                State& next = next_beam.back();
                next.produce();
                next.evaluate(eval_remaining);
                next_nodes.push_back({(short)idx, -1, -1});
            }
            
            // 2. アップグレード (Upgrade)
            if (remaining > 1) {
                for (int i = 0; i < L; i++) {
                    for (int j = 0; j < N; j++) {
                        
                        // ★★★ ピラミッド制約の適用 ★★★
                        // 上位レベル(i)の強化回数は、下位レベル(i-1)の強化回数未満でなければならない
                        // つまり、P[i-1] > P[i] が条件 (P[i-1] <= P[i] なら強化禁止)
                        // これにより、必ず Level 0 > Level 1 > Level 2 > Level 3 の形になる
                        if (i > 0 && state.P[i - 1][j] <= state.P[i][j]) {
                            continue;
                        }

                        if (state.canUpgrade(i, j)) {
                            next_beam.push_back(state);
                            State& upgraded = next_beam.back();
                            upgraded.upgrade(i, j);
                            upgraded.produce();
                            upgraded.evaluate(eval_remaining);
                          
                            // Level 3を買えた場合は、ピラミッドの頂点として強力なボーナス
                            if (i == 3) {
                                upgraded.raw_score += LEVEL3_ACTION_BONUS; // raw_scoreはlogとってないので扱い注意だが、大小関係への寄与としては機能する
                            }

                            next_nodes.push_back({(short)idx, (signed char)i, (signed char)j});
                        }
                    }
                }
            }
        }
        
        // ビーム選択
        if ((int)next_beam.size() > width) {
            vector<int> indices(next_beam.size());
            iota(indices.begin(), indices.end(), 0);
            
            nth_element(indices.begin(), indices.begin() + width, indices.end(),
                [&](int a, int b) { return next_beam[a].raw_score > next_beam[b].raw_score; });
            
            vector<State> new_beam;
            vector<Node> new_nodes;
            new_beam.reserve(width);
            new_nodes.reserve(width);
            
            for (int i = 0; i < width; i++) {
                int original_idx = indices[i];
                new_beam.push_back(next_beam[original_idx]);
                new_nodes.push_back(next_nodes[original_idx]);
            }
            cur_beam = move(new_beam);
            history[turn + 1] = move(new_nodes);
        } else {
            cur_beam = move(next_beam);
            history[turn + 1] = move(next_nodes);
        }
    }
    
    // 最良解
    int best_idx = 0;
    for (int i = 1; i < (int)cur_beam.size(); i++) {
        if (cur_beam[i].apples > cur_beam[best_idx].apples) {
            best_idx = i;
        }
    }
    
    vector<pair<int, int>> actions(T);
    int cur_idx = best_idx;
    for (int turn = T - 1; turn >= 0; turn--) {
        const Node& node = history[turn + 1][cur_idx];
        actions[turn] = {node.action_level, node.action_id};
        cur_idx = node.parent_idx;
    }
    
    for (int turn = 0; turn < T; turn++) {
        if (actions[turn].first == -1) {
            cout << "-1\n";
        } else {
            cout << actions[turn].first << " " << actions[turn].second << "\n";
        }
    }
    
    return 0;
}