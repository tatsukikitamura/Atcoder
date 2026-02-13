/**
 * AHC061 - Multi-Player Territory Game
 *
 * Algorithm: Beam Search (depth 4, width 10)
 *
 * Evaluation Components:
 *   1. Future V*L Potential: remaining turns & reachability-weighted strengthen potential
 *   2. Strengthen Priority: high-V own cells prioritized for strengthening
 *   3. Safety/Defensibility: cells surrounded by walls/own territory are more valuable
 *
 * Collision avoidance via opponent prediction (post-processing on first moves)
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
     vector<pair<int,int>> init_pos;  // recall destination
     vector<vector<int>> owner;
     vector<vector<int>> lv;
 
     // ========== Beam State (flat arrays for fast copy) ==========
     struct BState {
         vector<int> own, lev;      // N*N flat grids
         int px, py;                // my piece position
         int fx, fy;                // first move of this sequence
         double score;
     };
 
     int NN;  // N*N cache
     inline int idx(int x, int y) const { return x * N + y; }
 
     // ========== Input ==========
     void read_initial() {
         cin >> N >> M >> T >> U;
         NN = N * N;
         V.assign(N, vector<int>(N));
         for (int i = 0; i < N; i++)
             for (int j = 0; j < N; j++)
                 cin >> V[i][j];
         pos.resize(M);
         for (int p = 0; p < M; p++)
             cin >> pos[p].first >> pos[p].second;
         init_pos = pos;
 
         owner.assign(N, vector<int>(N, -1));
         lv.assign(N, vector<int>(N, 0));
         for (int p = 0; p < M; p++) {
             owner[pos[p].first][pos[p].second] = p;
             lv[pos[p].first][pos[p].second] = 1;
         }
     }
 
     void read_turn() {
         int a, b;
         for (int p = 0; p < M; p++) cin >> a >> b;
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
 
     // ========== Create initial BState from current game state ==========
     BState make_state() {
         BState bs;
         bs.own.resize(NN);
         bs.lev.resize(NN);
         for (int i = 0; i < N; i++)
             for (int j = 0; j < N; j++) {
                 int id = idx(i, j);
                 bs.own[id] = owner[i][j];
                 bs.lev[id] = lv[i][j];
             }
         bs.px = pos[0].first;
         bs.py = pos[0].second;
         bs.fx = -1; bs.fy = -1;
         bs.score = 0;
         return bs;
     }
 
     // ========== BFS: reachable candidates through own territory ==========
     vector<pair<int,int>> get_cands(const BState& bs) {
         vector<bool> visited(NN, false);
         vector<pair<int,int>> result;
         queue<pair<int,int>> q;
 
         q.push({bs.px, bs.py});
         visited[idx(bs.px, bs.py)] = true;
 
         while (!q.empty()) {
             auto [x, y] = q.front();
             q.pop();
             result.push_back({x, y});
 
             // Expand only through own territory (player 0)
             if (bs.own[idx(x, y)] == 0) {
                 for (int d = 0; d < 4; d++) {
                     int nx = x + DX[d], ny = y + DY[d];
                     if (nx >= 0 && nx < N && ny >= 0 && ny < N && !visited[idx(nx, ny)]) {
                         visited[idx(nx, ny)] = true;
                         q.push({nx, ny});
                     }
                 }
             }
         }
         return result;
     }
 
     // ========== Simulate player 0's move ==========
     BState sim(const BState& bs, int tx, int ty) {
         BState ns = bs;
         int id = idx(tx, ty);
 
         if (ns.own[id] == -1) {
             // Claim empty cell
             ns.own[id] = 0;
             ns.lev[id] = 1;
             ns.px = tx; ns.py = ty;
         } else if (ns.own[id] == 0) {
             // Strengthen own cell
             if (ns.lev[id] < U) ns.lev[id]++;
             ns.px = tx; ns.py = ty;
         } else {
             // Attack enemy cell
             ns.lev[id]--;
             if (ns.lev[id] == 0) {
                 // Captured!
                 ns.own[id] = 0;
                 ns.lev[id] = 1;
                 ns.px = tx; ns.py = ty;
             } else {
                 // Failed attack -> piece recalled to initial position
                 ns.px = init_pos[0].first;
                 ns.py = init_pos[0].second;
             }
         }
         return ns;
     }
 
     // ========== Safety metric: safe sides (wall or own territory) ==========
     int safe_sides(int x, int y, const vector<int>& own) {
         int safe = 0;
         for (int d = 0; d < 4; d++) {
             int nx = x + DX[d], ny = y + DY[d];
             if (nx < 0 || nx >= N || ny < 0 || ny >= N)
                 safe++;       // wall = safe
             else if (own[idx(nx, ny)] == 0)
                 safe++;       // own territory = safe
         }
         return safe;
     }
 
     // ========== Quick eval for candidate pre-filtering ==========
     double quick_eval(const BState& bs, int tx, int ty, int turn) {
         double v = V[tx][ty];
         double phase = (double)turn / T;
         int id = idx(tx, ty);
 
         if (bs.own[id] == -1) {
             // Claim: value = immediate V + future strengthen potential V*(U-1)
             return v * (1.0 + 0.3 * (U - 1)) * (1.3 - 0.4 * phase);
         } else if (bs.own[id] == 0) {
             if (bs.lev[id] < U) {
                 // Strengthen: direct +V to score, more valuable late
                 int safe = safe_sides(tx, ty, bs.own);
                 return v * (0.8 + 0.6 * phase) * (0.5 + 0.5 * safe / 4.0);
             }
             return -1e9;  // already max level
         } else {
             // Enemy territory
             if (bs.lev[id] == 1) return v * 2.0;  // one-hit capture
             return -200 + v * 0.05;                // multi-hit recall
         }
     }
 
     // ========== Full state evaluation (3 components) ==========
     double evaluate(const BState& bs, int turn) {
         double phase = (double)turn / T;
         int remaining = T - turn;
 
         double current_vl = 0;    // Component 1: Current V*L score
         double future_pot = 0;    // Component 2: Future V*L potential (strengthen priority)
         double safety_val = 0;    // Component 3: Safety-weighted value
 
         for (int i = 0; i < N; i++) {
             for (int j = 0; j < N; j++) {
                 int id = idx(i, j);
                 if (bs.own[id] != 0) continue;
 
                 double v = V[i][j];
                 int l = bs.lev[id];
 
                 // --- Component 1: Current V*L ---
                 current_vl += v * l;
 
                 // --- Component 2: Future strengthen potential ---
                 // High-V cells with low level have high potential
                 if (l < U) {
                     int dist = abs(i - bs.px) + abs(j - bs.py);
                     // Reachability: closer cells are easier to reach and strengthen
                     double reachability = 1.0 / (1.0 + dist * 0.3);
                     // Feasibility: can we actually do this in remaining turns?
                     int gap = U - l;
                     double feasibility = min(1.0, (double)remaining / max(1, gap + dist));
                     future_pot += v * gap * reachability * feasibility;
                 }
 
                 // --- Component 3: Safety / Defensibility ---
                 // Cells with more safe borders (wall/own territory) contribute reliably
                 int safe = safe_sides(i, j, bs.own);
                 double safety_factor = (double)safe / 4.0;
                 safety_val += v * l * safety_factor;
             }
         }
 
         // Phase-adaptive weights
         // Early: future potential matters most (expand & plan)
         // Late: current score + safety matters most (protect & strengthen)
         double w1 = 1.0;
         double w2 = 0.4 * (1.0 - phase * 0.5);   // future potential decreases late
         double w3 = 0.3 * (0.3 + phase * 0.7);    // safety increases late
 
         return w1 * current_vl + w2 * future_pot + w3 * safety_val;
     }
 
     // ========== AI prediction for collision avoidance ==========
     pair<int,int> predict_ai(int player) {
         // Simple BFS candidates for this player
         vector<vector<bool>> visited(N, vector<bool>(N, false));
         vector<pair<int,int>> cands;
         queue<pair<int,int>> q;
         int sx = pos[player].first, sy = pos[player].second;
         q.push({sx, sy});
         visited[sx][sy] = true;
         while (!q.empty()) {
             auto [x, y] = q.front(); q.pop();
             cands.push_back({x, y});
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
 
         double best = -1e18;
         pair<int,int> target = {sx, sy};
         for (auto [x, y] : cands) {
             double v = V[x][y];
             double eval = 0;
             if (owner[x][y] == -1)          eval = v * 0.65;
             else if (owner[x][y] == player) eval = (lv[x][y] < U) ? v * 0.65 : 0;
             else                            eval = (lv[x][y] == 1) ? v * 0.65 : v * 0.3;
             if (eval > best) { best = eval; target = {x, y}; }
         }
         return target;
     }
 
     // ========== Beam Search ==========
     pair<int,int> choose_move(int turn) {
         const int BEAM_W = 10;
         const int MAX_D  = min(4, T - turn);
         const int MAX_C  = 12;  // max candidates per beam state
 
         auto init_bs = make_state();
         vector<BState> beam = {init_bs};
 
         for (int d = 0; d < MAX_D; d++) {
             vector<BState> next;
 
             for (auto& bs : beam) {
                 auto cands = get_cands(bs);
                 if (cands.empty()) continue;
 
                 // Depth 0: exclude cells occupied by opponent pieces
                 if (d == 0) {
                     vector<pair<int,int>> filtered;
                     for (auto [cx, cy] : cands) {
                         bool blocked = false;
                         for (int p = 1; p < M; p++) {
                             if (pos[p].first == cx && pos[p].second == cy) {
                                 blocked = true; break;
                             }
                         }
                         if (!blocked) filtered.push_back({cx, cy});
                     }
                     cands = filtered;
                 }
 
                 // Score candidates with quick_eval, take top MAX_C
                 vector<pair<double, pair<int,int>>> scored;
                 for (auto [cx, cy] : cands) {
                     int id = idx(cx, cy);
                     if (bs.own[id] == 0 && bs.lev[id] >= U) continue;  // skip maxed cells
                     scored.push_back({quick_eval(bs, cx, cy, turn + d), {cx, cy}});
                 }
                 sort(scored.rbegin(), scored.rend());
                 int lim = min((int)scored.size(), MAX_C);
 
                 for (int i = 0; i < lim; i++) {
                     auto [cx, cy] = scored[i].second;
                     BState ns = sim(bs, cx, cy);
                     ns.fx = (d == 0) ? cx : bs.fx;
                     ns.fy = (d == 0) ? cy : bs.fy;
                     ns.score = evaluate(ns, turn + d + 1);
                     next.push_back(move(ns));
                 }
             }
 
             if (next.empty()) break;
 
             sort(next.begin(), next.end(),
                 [](const BState& a, const BState& b) { return a.score > b.score; });
             if ((int)next.size() > BEAM_W) next.resize(BEAM_W);
             beam = move(next);
         }
 
         if (beam.empty() || beam[0].fx == -1) return pos[0];
 
         // ========== Collision avoidance (post-processing) ==========
         // Predict AI targets
         vector<pair<int,int>> ai_targets(M);
         for (int p = 1; p < M; p++)
             ai_targets[p] = predict_ai(p);
 
         for (auto& bs : beam) {
             for (int p = 1; p < M; p++) {
                 int d = abs(pos[p].first - bs.fx) + abs(pos[p].second - bs.fy);
                 if (d <= 1) bs.score -= 400;
                 else if (d == 2) bs.score -= 60;
 
                 // Predicted target collision
                 if (ai_targets[p].first == bs.fx && ai_targets[p].second == bs.fy)
                     bs.score -= 500;
             }
         }
         sort(beam.begin(), beam.end(),
             [](const BState& a, const BState& b) { return a.score > b.score; });
 
         return {beam[0].fx, beam[0].fy};
     }
 
     // ========== Main Loop ==========
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
 