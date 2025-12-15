#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <random>
#include <cmath>
using namespace std;
using lint = long long int;
using PA = pair<int, int>;
using PL = pair<lint, lint>;
#define FOR(i, begin, end) for(int i=(begin),i##_end_=(end);i<i##_end_;i++)
#define IFOR(i, begin, end) for(int i=(end)-1,i##_begin_=(begin);i>=i##_begin_;i--)
#define REP(i, n) FOR(i,0,n)
#define IREP(i, n) IFOR(i,0,n)
#define ALL(a)  (a).begin(),(a).end()
constexpr int MOD = 1000000007;
vector<lint> RH_B = {1532834020, 1388622299};
vector<lint> RH_M = {2147482409, 2147478017};
constexpr int INF = 2147483647;
void yes(bool expr) {cout << (expr ? "Yes" : "No") << "\n";}
template<class T>void chmax(T &a, const T &b) { if (a<b) a=b; }
template<class T>void chmin(T &a, const T &b) { if (b<a) a=b; }
int N, L, T, K;
vector<lint> A;
vector<vector<lint>> C;
vector<vector<lint>> B, P;
mt19937_64 mt(0);
vector<int> fact;
lint get_cost(int l, int i, vector<vector<lint>> &P) {
  return C[l][i] * (P[l][i] + 1);
}

tuple<lint, vector<PA>, int> simulate(vector<PA> &actions, bool build_turns = false) {
  //操作列を元にシミュレーションする
  //操作が実行できるまで待つことにする
  lint score = 1;
  vector<PA> turns;
  int action_index = 0;
  REP(i, L) REP(j, N) {
    B[i][j] = 1;
    P[i][j] = 0;
  }
  REP(t, T) {
    if(action_index < actions.size()) {
      int i = actions[action_index].first;
      int j = actions[action_index].second;
      lint cost = get_cost(i, j, P);
      if(score >= cost) {
        score -= cost;
        P[i][j]++;
        action_index++;
        if(build_turns) turns.emplace_back(i, j);
      } else {
        if(build_turns) turns.emplace_back(-1, -1);
      }
    } else {
      if(build_turns) turns.emplace_back(-1, -1);
    }
    REP(j, N) REP(i, L) {
      if(P[i][j] == 0) break;
      if(i == 0) score += A[j] * B[i][j] * P[i][j];
      else B[i-1][j] += B[i][j] * P[i][j];
    }
  }
  return {score, turns, action_index};
}

pair<lint, vector<PA>> solve(
  int target1,
  int target2,
  int patience_0,
  int patience_1,
  int patience_2,
  int patience_3,
  int max_level_count_lim) {
  lint score = 1;
  REP(i, L) REP(j, N) {
    B[i][j] = 1;
    P[i][j] = 0;
  }
  vector<PA> actions;
  int max_level = 0;
  int max_level_count = 0;
  REP(t, T) {
    lint score_per_turn = 0;
    REP(j, N) score_per_turn += A[j] * B[0][j] * P[0][j];
    double best_value = 1e-9;
    int best_i = -1, best_j = -1;
    int wait_level = L;
    IREP(i, L) REP(j, N) {
      if(!(j == target1 || j == target2 || (i == 0 && max_level == 0))) continue;
      if((max_level > 1 || (max_level == 1 && max_level_count > max_level_count_lim)) && j != target1) continue;
      lint cost = get_cost(i, j, P);
      double cost_normalized = max((double)cost, score_per_turn/5.0);
      double value = (double)A[j] / cost_normalized * pow((T-t-1) * 0.8, i+1) / fact[i] * B[i][j];
      REP(l, i) value *= P[l][j];
      if(j != target1 && j != target2) value *= 0.5;
      else if(j != target2) value *= 0.9;
      if(value > best_value && score >= cost && i < wait_level) {
        best_value = value;
        best_i = i;
        best_j = j;
      }
      //P=0について、patienceターン以内に変えるなら、何もしないで強制終了
      if(value > best_value && i == 0 && P[i][j] < 1 && score_per_turn * patience_0 + score >= cost && score < cost) {
        wait_level = -1;
        best_i = -1;
        best_j = -1;
      }
      if(value > best_value && i == 1 && P[i][j] < 1 && score_per_turn * patience_1 + score >= cost && score < cost) {
        wait_level = 0;
        best_i = -1;
        best_j = -1;
      }
      if(value > best_value && i == 2 && P[i][j] < 1 && score_per_turn * patience_2 + score >= cost && score < cost) {
        wait_level = 1;
        best_i = -1;
        best_j = -1;
      }
      if(value > best_value && i == 3 && P[i][j] < 1 && score_per_turn * patience_3 + score >= cost && score < cost) {
        wait_level = 2;
        best_i = -1;
        best_j = -1;
      }
    }
    if(best_i == -1 || best_j == -1) {
      //actions.emplace_back(-1, -1);
    } else {
      lint cost = get_cost(best_i, best_j, P);
      if(score >= cost) {
        actions.emplace_back(best_i, best_j);
        score -= get_cost(best_i, best_j, P);
        P[best_i][best_j]++;
        if(best_i > max_level) {
          max_level = best_i;
          max_level_count = 1;
        } else if(best_i == max_level) {
          max_level_count++;
        }
      } else {
        //actions.emplace_back(-1, -1);
      }
    }
    REP(i, L) REP(j, N) {
      if(i == 0) score += A[j] * B[i][j] * P[i][j];
      else B[i-1][j] += B[i][j] * P[i][j];
    }
  }
  return {score, actions};
}

int main()
{
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  cin >> N >> L >> T >> K;
  A.resize(N);
  REP(i, N) cin >> A[i];
  C.resize(L, vector<lint>(N));
  REP(i, L) REP(j, N) cin >> C[i][j];
  B.resize(L, vector<lint>(N, 1));
  P.resize(L, vector<lint>(N, 0));

  auto start_time = chrono::system_clock::now();

  fact.resize(L);
  fact[0] = 1;
  FOR(i, 1, L) fact[i] = fact[i-1] * (i+1);

  //0と0以外のあるジェネレーターに絞って強化を行う
  //上位のレベルのジェネレーターにpatienceターンで届きそうなら強化を止める
  lint best_score = 0;
  vector<PA> best_actions;
  int best_patience_1 = 0;
  int best_patience_2 = 60;
  int best_patience_3 = 40;
  int best_target1 = -1;
  int best_target2 = -1;
  int best_patience_0 = 5;
  int best_max_level_count_lim = 20;
  REP(target1, N) REP(target2, N) for(int patience_1 = 0; patience_1 <= 30; patience_1 += 10) for(int max_level_count_lim = 0; max_level_count_lim <= 30; max_level_count_lim += 10) {
    auto [score, actions] = solve(target1, target2, best_patience_0, patience_1, best_patience_2, best_patience_3, max_level_count_lim);

    if(score > best_score) {
      best_score = score;
      best_actions = actions;
      best_patience_1 = patience_1;
      best_target1 = target1;
      best_target2 = target2;
      best_max_level_count_lim = max_level_count_lim;
    }
  }

  //操作列を焼きなましで改善する
  auto sa_start_time = chrono::system_clock::now();
  double sa_time_limit = 1980 - chrono::duration_cast<chrono::milliseconds>(sa_start_time - start_time).count();
  int loop = 0;
  float start_temp = 2.0, end_temp = 0.8;
  float temp = start_temp;
  float time;
  double sa_best_score = 100000 * log2(best_score);
  vector<PA> best_turns;
  double sa_best_score_2 = 0;
  while(true) {
    if(loop%100 == 0) {
      time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - sa_start_time).count();
      if(time > sa_time_limit) break;
      cerr << loop << " " << sa_best_score << "\n";
      temp = start_temp + (end_temp - start_temp) * time / (float)sa_time_limit;
    }
    loop++;
    
    vector<PA> last_actions(best_actions);
    //追加・削除・変更
    int mode = mt() % 100;
    if(mode < 10) {
      //追加
      int pos = mt() % (best_actions.size()) + 1;
      int i = mt() % L;
      if(mt()%2 == 0) i = best_actions[pos-1].first;
      int j = mt() % N;
      if(mt()%2 == 0) j = best_actions[pos-1].second;
      best_actions.insert(best_actions.begin() + pos, PA(i, j));
    } else if(mode < 30) {
      //削除
      if(best_actions.size() < 2) continue;
      int pos = mt() % (best_actions.size() - 1) + 1;
      best_actions.erase(best_actions.begin() + pos);
    } else if(mode < 40){
      //変更
      if(best_actions.size() < 2) continue;
      int pos = mt() % (best_actions.size() - 1) + 1;
      int i = mt() % L;
      if(mt()%2 == 0) i = best_actions[pos-1].first;
      int j = mt() % N;
      if(mt()%2 == 0) j = best_actions[pos-1].second;
      best_actions[pos] = PA(i, j);
    } else if(mode < 60) {
      //複製
      if(best_actions.size() < 2) continue;
      int pos = mt() % (best_actions.size() - 1) + 1;
      best_actions.insert(best_actions.begin() + pos, best_actions[pos]);
    } else if(mode < 90) {
      //swap
      if(best_actions.size() < 3) continue;
      int pos = mt() % (best_actions.size() - 2) + 1;
      int stride = 1;
      if(pos + stride >= best_actions.size()) continue;
      if(best_actions[pos] == best_actions[pos+stride]) continue;
      swap(best_actions[pos], best_actions[pos+stride]);
    } else {
      //範囲target変更
      if(best_actions.size() < 2) continue;
      int pos1 = mt() % (best_actions.size() - 1) + 1;
      int pos2 = mt() % (best_actions.size() - 1) + 1;
      if(pos1 > pos2) swap(pos1, pos2);
      pos2 += 1;
      int before_target = best_actions[mt()%best_actions.size()].second;
      int new_target = best_actions[mt()%best_actions.size()].second;
      if(before_target == new_target) continue;
      FOR(pos, pos1, pos2) {
        if(best_actions[pos].second == before_target) {
          best_actions[pos].second = new_target;
        }
      }

    }
    auto [score, turns, action_idx] = simulate(best_actions, true);
    double sa_score = 100000 * log2(score);
    double diff = sa_score - sa_best_score;
    float prob = exp(diff*pow(0.1, temp));
    if(diff > 0 || prob*(float)INF > (mt()%INF)) {
      sa_best_score = sa_score;
      while(action_idx < best_actions.size()) {
        best_actions.pop_back();
      }
      if(sa_best_score > sa_best_score_2) {
        sa_best_score_2 = sa_best_score;
        best_turns = turns;
      }
    } else {
      //戻す
      best_actions = last_actions;
    }
  }

  for(auto p : best_turns) {
    if(p.first == -1) cout << "-1\n";
    else cout << p.first << " " << p.second << "\n";
  }
  cerr << (lint)round(sa_best_score_2) << "\n";
}