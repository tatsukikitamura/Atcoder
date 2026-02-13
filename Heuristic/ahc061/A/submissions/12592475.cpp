/**
 * AHC061 - Multi-Player Territory Game
 *
 * Algorithm: Chokudai Search with opponent simulation
 *   - Time-limited iterative beam search using priority queues
 *   - Simulates opponent greedy moves for accurate board prediction
 *   - Fixed-size int8_t arrays for fast state copying (~230 bytes/state)
 *   - Correct recall mechanic: piece stays at pre-move position (not init_pos)
 *   - Phase-adaptive evaluation (V*L, potential, safety, proximity, opp gap)
 *   - Collision avoidance via opponent prediction (post-processing)
 *
 * Key fixes over previous Chokudai attempt (new.cpp):
 *   1. int8_t fixed arrays instead of vector<int> -> ~10x faster state copy
 *   2. Correct recall: piece returns to turn-start position, not init_pos
 *   3. Opponent simulation for accurate multi-turn board prediction
 */

 #include <iostream>
 #include <vector>
 #include <algorithm>
 #include <queue>
 #include <cmath>
 #include <chrono>
 #include <cstring>
 using namespace std;
 
 const int DX[] = {-1, 1, 0, 0};
 const int DY[] = {0, 0, -1, 1};
 
 int N, M, T, U;
 int VV[10][10];
 
 // True game state (read from tester each turn)
 int g_owner[10][10], g_lv[10][10];
 int g_px[8], g_py[8];
 
 chrono::high_resolution_clock::time_point G_START;
 
 inline double elapsed_sec() {
     return chrono::duration<double>(chrono::high_resolution_clock::now() - G_START).count();
 }
 
 void read_initial() {
     cin >> N >> M >> T >> U;
     for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
             cin >> VV[i][j];
     for (int p = 0; p < M; p++)
         cin >> g_px[p] >> g_py[p];
 
     for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
             g_owner[i][j] = -1;
     memset(g_lv, 0, sizeof(g_lv));
     for (int p = 0; p < M; p++) {
         g_owner[g_px[p]][g_py[p]] = p;
         g_lv[g_px[p]][g_py[p]] = 1;
     }
 }
 
 void read_turn() {
     int a, b;
     for (int p = 0; p < M; p++) cin >> a >> b;
     for (int p = 0; p < M; p++) cin >> g_px[p] >> g_py[p];
     for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
             cin >> g_owner[i][j];
     for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++)
             cin >> g_lv[i][j];
 }
 
 // ===== Beam State: compact fixed-size struct for fast copy =====
 struct BState {
     int8_t own[100];     // owner[r*10+c]: -1 or 0..M-1
     int8_t lev[100];     // level[r*10+c]: 0..U
     int8_t px[8], py[8]; // all player positions
     double score;
     int8_t fx, fy;       // first move of player 0 from root
 
     bool operator<(const BState& o) const { return score < o.score; }
 };
 
 // ===== BFS: reachable candidates for a player =====
 // Returns cells reachable via own territory + 1-step frontier.
 // Excludes cells occupied by other players' pieces.
 int get_cands(const BState& bs, int player, pair<int8_t,int8_t>* out) {
     bool vis[100];
     memset(vis, 0, 100);
     pair<int8_t,int8_t> bq[100];
     int qh = 0, qt = 0, cnt = 0;
 
     int sx = bs.px[player], sy = bs.py[player];
     bq[qt++] = {(int8_t)sx, (int8_t)sy};
     vis[sx * 10 + sy] = true;
 
     while (qh < qt) {
         int x = bq[qh].first, y = bq[qh].second;
         qh++;
 
         // Exclude cells with another player's piece
         bool blk = false;
         for (int p = 0; p < M; p++) {
             if (p != player && bs.px[p] == x && bs.py[p] == y) {
                 blk = true;
                 break;
             }
         }
         if (!blk) out[cnt++] = {(int8_t)x, (int8_t)y};
 
         // Expand BFS only through own territory
         if (bs.own[x * 10 + y] == player) {
             for (int d = 0; d < 4; d++) {
                 int nx = x + DX[d], ny = y + DY[d];
                 if (nx >= 0 && nx < N && ny >= 0 && ny < N && !vis[nx * 10 + ny]) {
                     vis[nx * 10 + ny] = true;
                     bq[qt++] = {(int8_t)nx, (int8_t)ny};
                 }
             }
         }
     }
     return cnt;
 }
 
 // ===== Apply a single player's move to state =====
 // Correct recall: on failed attack, piece stays at pre-move position (unchanged).
 inline void apply_move(BState& ns, int player, int tx, int ty) {
     int id = tx * 10 + ty;
 
     if (ns.own[id] == -1) {
         // Claim empty cell
         ns.own[id] = player;
         ns.lev[id] = 1;
         ns.px[player] = tx;
         ns.py[player] = ty;
     } else if (ns.own[id] == player) {
         // Strengthen own cell
         if (ns.lev[id] < U) ns.lev[id]++;
         ns.px[player] = tx;
         ns.py[player] = ty;
     } else {
         // Attack enemy cell
         ns.lev[id]--;
         if (ns.lev[id] == 0) {
             // Captured!
             ns.own[id] = player;
             ns.lev[id] = 1;
             ns.px[player] = tx;
             ns.py[player] = ty;
         }
         // Failed attack: px[player], py[player] unchanged = pre-move position
         // This correctly implements the recall rule (return to turn-start position)
     }
 }
 
 // ===== Simulate opponent moves using greedy prediction =====
 // Each opponent picks the cell with highest evaluation.
 // Uses approximate AI weights (wa ≈ wb ≈ wc ≈ 0.65, wd ≈ 0.30).
 void sim_opponents(BState& ns) {
     pair<int8_t,int8_t> opcands[100];
 
     for (int p = 1; p < M; p++) {
         int nc = get_cands(ns, p, opcands);
         if (nc == 0) continue;
 
         double best = -1e18;
         int bx = ns.px[p], by = ns.py[p];
 
         for (int i = 0; i < nc; i++) {
             int x = opcands[i].first, y = opcands[i].second;
             int id = x * 10 + y;
             double v = VV[x][y];
             double ev = 0;
 
             if (ns.own[id] == -1)          ev = v * 0.65;
             else if (ns.own[id] == p)      ev = (ns.lev[id] < U) ? v * 0.65 : 0;
             else                           ev = (ns.lev[id] == 1) ? v * 0.65 : v * 0.30;
 
             if (ev > best) { best = ev; bx = x; by = y; }
         }
         apply_move(ns, p, bx, by);
     }
 }
 
 // ===== Quick eval: fast heuristic for candidate pre-filtering =====
 inline double quick_eval(const BState& bs, int tx, int ty, int turn) {
     double v = VV[tx][ty];
     double phase = (double)turn / T;
     int id = tx * 10 + ty;
 
     if (bs.own[id] == -1) {
         // Claim: immediate V + future strengthen potential V*(U-1)
         return v * (1.0 + 0.3 * (U - 1)) * (1.5 - 0.5 * phase);
     }
     if (bs.own[id] == 0) {
         if (bs.lev[id] < U) {
             // Strengthen: value increases late, safety bonus
             int safe = 0;
             for (int d = 0; d < 4; d++) {
                 int nx = tx + DX[d], ny = ty + DY[d];
                 if (nx < 0 || nx >= N || ny < 0 || ny >= N) safe++;
                 else if (bs.own[nx * 10 + ny] == 0) safe++;
             }
             return v * (0.8 + 0.6 * phase) * (0.5 + 0.5 * safe / 4.0);
         }
         return -1e9; // already max level
     }
     // Enemy territory
     if (bs.lev[id] == 1) return v * 2.0; // one-hit capture
     int rem = T - turn;
     if (rem <= 5) return v * 0.8 - max(0.0, (rem - 1.0) * 20.0);
     return v * 0.05 - 200; // multi-hit: avoid (recall penalty)
 }
 
 // ===== Full state evaluation (6 components) =====
 double evaluate(const BState& bs, int turn) {
     double phase = (double)turn / T;
     int rem = T - turn;
 
     double vl = 0;    // Component 1: Current V*L score
     double fut = 0;   // Component 2: Future strengthen potential
     double saf = 0;   // Component 3: Safety / defensibility
     double pex = 0;   // Component 4: Proximity to claimable/capturable cells
     double pst = 0;   // Component 5: Proximity to upgradable own cells
     double ovl = 0;   // Component 6: Opponent V*L (penalize to reward attacks)
 
     int p0x = bs.px[0], p0y = bs.py[0];
 
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             int id = i * 10 + j;
             double v = VV[i][j];
 
             if (bs.own[id] == 0) {
                 int l = bs.lev[id];
 
                 // C1: Current V*L
                 vl += v * l;
 
                 // C2: Future strengthen potential
                 if (l < U) {
                     int dist = abs(i - p0x) + abs(j - p0y);
                     double reach = 1.0 / (1.0 + dist * 0.3);
                     int gap = U - l;
                     double feas = min(1.0, (double)rem / max(1, gap + dist));
                     fut += v * gap * reach * feas;
 
                     // C5: Proximity to strengthen targets (mid-late game)
                     if (phase > 0.3 && dist > 0 && dist <= 4)
                         pst += v * gap / (1.0 + dist);
                 }
 
                 // C3: Safety (surrounded by walls/own territory)
                 int safe = 0;
                 for (int d = 0; d < 4; d++) {
                     int nx = i + DX[d], ny = j + DY[d];
                     if (nx < 0 || nx >= N || ny < 0 || ny >= N) safe++;
                     else if (bs.own[nx * 10 + ny] == 0) safe++;
                 }
                 saf += v * l * safe * 0.25;
 
             } else if (bs.own[id] > 0) {
                 // C6: Opponent V*L (larger = worse for us)
                 ovl += v * bs.lev[id];
             }
 
             // C4: Expansion/capture proximity guide (early-mid game)
             if (phase < 0.7) {
                 int dist = abs(i - p0x) + abs(j - p0y);
                 if (dist > 0 && dist <= 5) {
                     if (bs.own[id] == -1)
                         pex += v / (1.0 + dist);
                     else if (bs.own[id] > 0 && bs.lev[id] == 1)
                         pex += v * 0.5 / (1.0 + dist);
                 }
             }
         }
     }
 
     // Phase-adaptive weights
     double w1 = 1.0;                                       // V*L: always dominant
     double w2 = 0.4 * (1.0 - phase * 0.5);                // future potential: fades late
     double w3 = 0.3 * (0.3 + phase * 0.7);                // safety: grows late
     double w4 = 0.08 * max(0.0, 1.0 - phase * 1.4);       // expansion guide: early only
     double w5 = 0.02 * max(0.0, phase - 0.3) / 0.7;       // strengthen guide: mid-late
     double w6 = (phase > 0.5) ? 0.3 * min(1.0, (phase - 0.5) / 0.5) : 0.0;
 
     return w1 * vl + w2 * fut + w3 * saf + w4 * pex + w5 * pst - w6 * ovl;
 }
 
 // ===== AI prediction using current game state (for collision avoidance) =====
 pair<int,int> predict_ai_now(int player) {
     bool vis[100];
     memset(vis, 0, 100);
     pair<int8_t,int8_t> bq[100], cds[100];
     int qh = 0, qt = 0, cnt = 0;
 
     int sx = g_px[player], sy = g_py[player];
     bq[qt++] = {(int8_t)sx, (int8_t)sy};
     vis[sx * 10 + sy] = true;
 
     while (qh < qt) {
         int x = bq[qh].first, y = bq[qh].second;
         qh++;
 
         // Exclude cells with other pieces
         bool blk = false;
         for (int p = 0; p < M; p++) {
             if (p != player && g_px[p] == x && g_py[p] == y) {
                 blk = true;
                 break;
             }
         }
         if (!blk) cds[cnt++] = {(int8_t)x, (int8_t)y};
 
         if (g_owner[x][y] == player) {
             for (int d = 0; d < 4; d++) {
                 int nx = x + DX[d], ny = y + DY[d];
                 if (nx >= 0 && nx < N && ny >= 0 && ny < N && !vis[nx * 10 + ny]) {
                     vis[nx * 10 + ny] = true;
                     bq[qt++] = {(int8_t)nx, (int8_t)ny};
                 }
             }
         }
     }
 
     double best = -1e18;
     int bx = sx, by = sy;
     for (int i = 0; i < cnt; i++) {
         int x = cds[i].first, y = cds[i].second;
         double v = VV[x][y];
         double ev = 0;
         if (g_owner[x][y] == -1)          ev = v * 0.65;
         else if (g_owner[x][y] == player) ev = (g_lv[x][y] < U) ? v * 0.65 : 0;
         else                              ev = (g_lv[x][y] == 1) ? v * 0.65 : v * 0.30;
         if (ev > best) { best = ev; bx = x; by = y; }
     }
     return {bx, by};
 }
 
 // ===== Chokudai Search: main decision function =====
 pair<int,int> choose_move(int turn) {
     const int MAX_D = min(6, T - turn);   // max lookahead depth
     const int MAX_C = 8;                  // max candidates per expansion (deep)
     const int MAX_C0 = 20;               // max candidates at depth 0 (wider)
 
     // === Time management ===
     double t0 = elapsed_sec();
     double t_rem = 1.90 - t0;
     int t_left = T - turn;
     double t_per = t_rem / max(1, t_left);
     double phase = (double)turn / T;
     // Early game bonus: decisions have more long-term impact
     if (phase < 0.5) t_per *= 1.2;
     double deadline = t0 + max(t_per, 0.002); // at least 2ms
 
     // === Priority queues for each depth (max-heap by score) ===
     vector<priority_queue<BState>> beams(MAX_D + 1);
 
     // === Initialize root state from current game state ===
     BState root;
     for (int i = 0; i < N; i++)
         for (int j = 0; j < N; j++) {
             root.own[i * 10 + j] = (int8_t)g_owner[i][j];
             root.lev[i * 10 + j] = (int8_t)g_lv[i][j];
         }
     for (int p = 0; p < M; p++) {
         root.px[p] = (int8_t)g_px[p];
         root.py[p] = (int8_t)g_py[p];
     }
     root.fx = -1;
     root.fy = -1;
     root.score = evaluate(root, turn);
     beams[0].push(root);
 
     // === Chokudai iterations ===
     pair<int8_t,int8_t> cds[100];
     pair<double, pair<int8_t,int8_t>> sc[100];
 
     while (elapsed_sec() < deadline) {
         bool any = false;
         for (int d = 0; d < MAX_D; d++) {
             if (beams[d].empty()) continue;
             if (elapsed_sec() >= deadline) break;
 
             // Pop best state at this depth
             BState bs = beams[d].top();
             beams[d].pop();
 
             // Get candidates for player 0
             int nc = get_cands(bs, 0, cds);
             if (nc == 0) continue;
 
             // Score and filter candidates
             int nsc = 0;
             for (int i = 0; i < nc; i++) {
                 int cx = cds[i].first, cy = cds[i].second;
                 int id = cx * 10 + cy;
                 // Skip already-maxed own cells (waste of turn)
                 if (bs.own[id] == 0 && bs.lev[id] >= U) continue;
                 sc[nsc++] = {quick_eval(bs, cx, cy, turn + d), {cds[i].first, cds[i].second}};
             }
 
             // Sort by quick_eval descending, take top MAX_C (or MAX_C0 at depth 0)
             sort(sc, sc + nsc, [](const auto& a, const auto& b) {
                 return a.first > b.first;
             });
             int lim = min(nsc, (d == 0) ? MAX_C0 : MAX_C);
 
             for (int i = 0; i < lim; i++) {
                 int cx = sc[i].second.first, cy = sc[i].second.second;
                 BState ns = bs;
 
                 // Apply player 0's move
                 apply_move(ns, 0, cx, cy);
 
                 // Simulate opponent greedy moves
                 sim_opponents(ns);
 
                 // Track first move
                 ns.fx = (d == 0) ? (int8_t)cx : bs.fx;
                 ns.fy = (d == 0) ? (int8_t)cy : bs.fy;
 
                 // Evaluate resulting state
                 ns.score = evaluate(ns, turn + d + 1);
 
                 beams[d + 1].push(ns);
                 any = true;
             }
         }
         if (!any) break;
     }
 
     // === Extract best result from deepest non-empty beam ===
     BState best;
     best.fx = -1;
     best.fy = -1;
     best.score = -1e18;
 
     for (int d = MAX_D; d >= 1; d--) {
         if (beams[d].empty()) continue;
 
         // Collect top candidates for collision avoidance post-processing
         vector<BState> top;
         while (!beams[d].empty() && (int)top.size() < 30) {
             top.push_back(beams[d].top());
             beams[d].pop();
         }
 
         // Predict AI targets for collision avoidance
         pair<int,int> ai_tgt[8];
         for (int p = 1; p < M; p++)
             ai_tgt[p] = predict_ai_now(p);
 
         // Apply collision penalties to first moves
         for (auto& bs : top) {
             for (int p = 1; p < M; p++) {
                 int dist = abs(g_px[p] - (int)bs.fx) + abs(g_py[p] - (int)bs.fy);
                 if (dist <= 1) bs.score -= 400;
                 else if (dist == 2) bs.score -= 60;
                 if (ai_tgt[p].first == (int)bs.fx && ai_tgt[p].second == (int)bs.fy)
                     bs.score -= 500;
             }
         }
 
         // Pick best after collision penalty
         for (const auto& bs : top) {
             if (bs.score > best.score) best = bs;
         }
         break; // use deepest non-empty level
     }
 
     if (best.fx >= 0) return {best.fx, best.fy};
 
     // === Fallback: greedy (if search produced no results) ===
     pair<int8_t,int8_t> fb_cds[100];
     int fnc = get_cands(root, 0, fb_cds);
     double fb_best = -1e18;
     int fbx = g_px[0], fby = g_py[0];
     for (int i = 0; i < fnc; i++) {
         int cx = fb_cds[i].first, cy = fb_cds[i].second;
         double ev = quick_eval(root, cx, cy, turn);
         if (ev > fb_best) { fb_best = ev; fbx = cx; fby = cy; }
     }
     return {fbx, fby};
 }
 
 // ===== Main Loop =====
 int main() {
     ios::sync_with_stdio(false);
     cin.tie(nullptr);
     G_START = chrono::high_resolution_clock::now();
 
     read_initial();
     for (int turn = 0; turn < T; turn++) {
         auto [tx, ty] = choose_move(turn);
         cout << tx << " " << ty << endl;
         read_turn();
     }
     return 0;
 }
 