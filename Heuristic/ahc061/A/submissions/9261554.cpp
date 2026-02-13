/**
 * AHC061 - Multi-Player Territory Game
 *
 * Strategy: Greedy evaluation with phase-dependent weights
 * - Early game: expand aggressively to claim high-value cells
 * - Late game: strengthen owned high-value cells to maximize V*L
 * - Predict AI targets to avoid collisions
 */

 #include <iostream>
 #include <vector>
 #include <algorithm>
 #include <queue>
 #include <cmath>
 using namespace std;
 
 const int DX[] = {-1, 1, 0, 0};
 const int DY[] = {0, 0, -1, 1};
 
 struct Solver {
     int N, M, T, U;
     vector<vector<int>> V;
     vector<pair<int,int>> pos;
     vector<vector<int>> owner;
     vector<vector<int>> lv;
 
     // ---- Input ----
     void read_initial() {
         cin >> N >> M >> T >> U;
         V.assign(N, vector<int>(N));
         for (int i = 0; i < N; i++)
             for (int j = 0; j < N; j++)
                 cin >> V[i][j];
         pos.resize(M);
         for (int p = 0; p < M; p++)
             cin >> pos[p].first >> pos[p].second;
 
         owner.assign(N, vector<int>(N, -1));
         lv.assign(N, vector<int>(N, 0));
         for (int p = 0; p < M; p++) {
             owner[pos[p].first][pos[p].second] = p;
             lv[pos[p].first][pos[p].second] = 1;
         }
     }
 
     void read_turn() {
         int a, b;
         for (int p = 0; p < M; p++) cin >> a >> b; // selected moves
         vector<pair<int,int>> ex(M);
         for (int p = 0; p < M; p++) cin >> ex[p].first >> ex[p].second;
         for (int i = 0; i < N; i++)
             for (int j = 0; j < N; j++)
                 cin >> owner[i][j];
         for (int i = 0; i < N; i++)
             for (int j = 0; j < N; j++)
                 cin >> lv[i][j];
         pos = ex;
     }
 
     // ---- Move generation ----
     // BFS through own territory from current position.
     // Returns all reachable cells + adjacent cells (excluding cells with enemy pieces).
     vector<pair<int,int>> get_candidates(int player) {
         vector<vector<bool>> visited(N, vector<bool>(N, false));
         vector<pair<int,int>> result;
         queue<pair<int,int>> q;
 
         int sx = pos[player].first, sy = pos[player].second;
         q.push({sx, sy});
         visited[sx][sy] = true;
 
         while (!q.empty()) {
             auto [x, y] = q.front();
             q.pop();
 
             // Check if another player's piece is on this cell
             bool blocked = false;
             for (int p = 0; p < M; p++) {
                 if (p != player && pos[p].first == x && pos[p].second == y) {
                     blocked = true;
                     break;
                 }
             }
             if (!blocked) result.push_back({x, y});
 
             // Expand BFS only through own territory
             if (owner[x][y] == player) {
                 for (int d = 0; d < 4; d++) {
                     int nx = x + DX[d], ny = y + DY[d];
                     if (nx >= 0 && nx < N && ny >= 0 && ny < N && !visited[nx][ny]) {
                         visited[nx][ny] = true;
                         q.push({nx, ny});
                     }
                 }
             }
         }
         return result;
     }
 
     // ---- AI prediction ----
     // Simple prediction: AI most likely moves to highest-eval candidate
     pair<int,int> predict_ai_target(int player) {
         auto cands = get_candidates(player);
         if (cands.empty()) return pos[player];
 
         double best = -1e18;
         pair<int,int> target = cands[0];
         const double w = 0.65; // average AI weight
 
         for (auto [x, y] : cands) {
             double eval = 0;
             double v = V[x][y];
             if (owner[x][y] == -1) {
                 eval = v * w;
             } else if (owner[x][y] == player) {
                 eval = (lv[x][y] < U) ? v * w : 0;
             } else {
                 eval = (lv[x][y] == 1) ? v * w : v * w * 0.5;
             }
             if (eval > best) {
                 best = eval;
                 target = {x, y};
             }
         }
         return target;
     }
 
     // ---- Evaluation & move selection ----
     pair<int,int> choose_move(int turn) {
         auto candidates = get_candidates(0);
         if (candidates.empty()) return pos[0];
         if (candidates.size() == 1) return candidates[0];
 
         double phase = (double)turn / T; // 0.0 (early) -> 1.0 (late)
 
         // Predict AI targets for collision avoidance
         vector<pair<int,int>> ai_targets(M);
         for (int p = 1; p < M; p++) {
             ai_targets[p] = predict_ai_target(p);
         }
 
         double best_score = -1e18;
         pair<int,int> best_move = candidates[0];
 
         for (auto [tx, ty] : candidates) {
             double score = 0;
             double v = V[tx][ty];
 
             // === 1. Immediate action value ===
             if (owner[tx][ty] == -1) {
                 // Claim empty cell (higher value early)
                 score += v * (1.5 - 0.5 * phase);
             } else if (owner[tx][ty] == 0) {
                 if (lv[tx][ty] < U) {
                     // Strengthen own cell (higher value late)
                     score += v * (0.5 + 1.0 * phase);
                 } else {
                     // Already max level - waste of turn
                     score -= 100000;
                 }
             } else {
                 // Enemy territory
                 if (lv[tx][ty] == 1) {
                     // Capture in one hit (very valuable: we gain V, enemy loses V)
                     score += v * 2.0;
                 } else {
                     // Multi-hit attack, piece gets recalled - usually bad
                     score += v * 0.05 - 200;
                 }
             }
 
             // === 2. Proximity to valuable unclaimed/capturable cells ===
             if (phase < 0.7) {
                 double prox = 0;
                 for (int i = 0; i < N; i++) {
                     for (int j = 0; j < N; j++) {
                         int dist = abs(i - tx) + abs(j - ty);
                         if (dist == 0 || dist > 5) continue;
                         if (owner[i][j] == -1) {
                             prox += V[i][j] / (1.0 + dist);
                         } else if (owner[i][j] > 0 && lv[i][j] == 1) {
                             prox += V[i][j] * 0.5 / (1.0 + dist);
                         }
                     }
                 }
                 score += prox * 0.08 * (1.0 - phase);
             }
 
             // === 3. Nearby strengthen potential (late game bonus) ===
             if (phase > 0.3) {
                 double str_pot = 0;
                 for (int i = 0; i < N; i++) {
                     for (int j = 0; j < N; j++) {
                         if (owner[i][j] == 0 && lv[i][j] < U) {
                             int dist = abs(i - tx) + abs(j - ty);
                             if (dist <= 4) {
                                 str_pot += (double)V[i][j] * (U - lv[i][j]) / (1.0 + dist);
                             }
                         }
                     }
                 }
                 score += str_pot * 0.02 * phase;
             }
 
             // === 4. Collision avoidance ===
             for (int p = 1; p < M; p++) {
                 // Distance-based penalty
                 int d = abs(pos[p].first - tx) + abs(pos[p].second - ty);
                 if (d <= 1) score -= 300;
                 else if (d == 2) score -= 50;
 
                 // Predicted target collision
                 if (ai_targets[p].first == tx && ai_targets[p].second == ty) {
                     score -= 500;
                 }
             }
 
             if (score > best_score) {
                 best_score = score;
                 best_move = {tx, ty};
             }
         }
 
         return best_move;
     }
 
     // ---- Main loop ----
     void run() {
         read_initial();
         for (int turn = 0; turn < T; turn++) {
             auto [tx, ty] = choose_move(turn);
             cout << tx << " " << ty << endl;
             read_turn();
         }
     }
 };
 
 int main() {
     ios::sync_with_stdio(false);
     cin.tie(nullptr);
 
     Solver solver;
     solver.run();
 
     return 0;
 }
 