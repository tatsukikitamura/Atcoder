#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <cmath>
#include <chrono>
#include <random>
#include <set>

using namespace std;

// ===== Constants =====
constexpr int N = 20;
constexpr int INF = 1e9;

// ===== Timer =====
struct Timer {
    chrono::high_resolution_clock::time_point start;
    double limit;

    Timer(double limit_sec = 1.8) : limit(limit_sec) {
        start = chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start).count();
    }

    bool is_over() const {
        return elapsed() >= limit;
    }

    double progress() const {
        return min(1.0, elapsed() / limit);
    }
};

// ===== Position =====
struct Pos {
    int r, c;
    Pos() : r(0), c(0) {}
    Pos(int r, int c) : r(r), c(c) {}
    bool operator==(const Pos& o) const { return r == o.r && c == o.c; }
    bool operator!=(const Pos& o) const { return !(*this == o); }
};

int manhattan_dist(const Pos& a, const Pos& b) {
    return abs(a.r - b.r) + abs(a.c - b.c);
}

// ===== Input =====
struct Input {
    int board[N][N];
    pair<Pos, Pos> card_pos[N * N / 2];

    void read() {
        int n;
        if (!(cin >> n)) return;
        vector<bool> first_seen(N * N / 2, true);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cin >> board[i][j];
                int card = board[i][j];
                if (first_seen[card]) {
                    card_pos[card].first = Pos(i, j);
                    first_seen[card] = false;
                } else {
                    card_pos[card].second = Pos(i, j);
                }
            }
        }
    }
};

// ===== State Management =====
struct Group {
    vector<int> cards;
    vector<bool> visit_first; // true if first visited before second

    Pos get_entry_pos(const Input& input) const {
        int c = cards[0];
        return visit_first[0] ? input.card_pos[c].first : input.card_pos[c].second;
    }

    Pos get_exit_pos(const Input& input) const {
        int c = cards[0];
        return visit_first[0] ? input.card_pos[c].second : input.card_pos[c].first;
    }

    int internal_cost(const Input& input) const {
        int cost = 0;
        Pos cur = get_entry_pos(input);
        for (size_t i = 0; i < cards.size(); ++i) {
            int c = cards[i];
            Pos p1 = input.card_pos[c].first;
            Pos p2 = input.card_pos[c].second;
            Pos target1 = visit_first[i] ? p1 : p2;
            cost += manhattan_dist(cur, target1);
            cur = target1;
        }
        for (int i = (int)cards.size() - 1; i >= 0; --i) {
            int c = cards[i];
            Pos p1 = input.card_pos[c].first;
            Pos p2 = input.card_pos[c].second;
            Pos target2 = visit_first[i] ? p2 : p1;
            cost += manhattan_dist(cur, target2);
            cur = target2;
        }
        return cost;
    }
};

// ===== Solver =====
class Solver {
public:
    Input input;
    Timer timer;
    mt19937 rng;

    vector<Group> current_groups;
    int current_cost;

    vector<Group> best_groups;
    int best_cost;

    Solver() : rng(42), current_cost(INF), best_cost(INF) {}

    void solve() {
        input.read();
        init_greedy();
        
        best_groups = current_groups;
        best_cost = current_cost;
        cerr << "[DEBUG] Initial Cost: " << current_cost << endl;

        double start_temp = 20.0; 
        double end_temp = 0.01;
        long long iterations = 0;

        while (!timer.is_over()) {
            iterations++;
            double progress = timer.progress();
            double temp = start_temp + (end_temp - start_temp) * progress;

            int op = uniform_int_distribution<int>(0, 6)(rng);
            vector<Group> prev_groups = current_groups;
            int old_cost = current_cost;

            bool possible = true;
            if (op == 0) { // Swap Groups
                int i = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                int j = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                if (i == j) { possible = false; }
                else { swap(current_groups[i], current_groups[j]); }
            } 
            else if (op == 1) { // Flip Group
                int i = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                for (size_t k = 0; k < current_groups[i].visit_first.size(); ++k) {
                    current_groups[i].visit_first[k] = !current_groups[i].visit_first[k];
                }
            }
            else if (op == 2) { // Move Group
                int i = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                int j = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                if (i == j) { possible = false; }
                else {
                    Group g = current_groups[i];
                    current_groups.erase(current_groups.begin() + i);
                    current_groups.insert(current_groups.begin() + j, g);
                }
            }
            else if (op == 3) { // Merge Adjacent
                if (current_groups.size() < 2) { possible = false; }
                else {
                    int i = uniform_int_distribution<int>(0, (int)current_groups.size() - 2)(rng);
                    for (size_t k = 0; k < current_groups[i+1].cards.size(); ++k) {
                        current_groups[i].cards.push_back(current_groups[i+1].cards[k]);
                        current_groups[i].visit_first.push_back(current_groups[i+1].visit_first[k]);
                    }
                    current_groups.erase(current_groups.begin() + i + 1);
                }
            }
            else if (op == 4) { // Split
                int i = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                if (current_groups[i].cards.size() < 2) { possible = false; }
                else {
                    Group new_g;
                    new_g.cards.push_back(current_groups[i].cards.back());
                    new_g.visit_first.push_back(current_groups[i].visit_first.back());
                    current_groups[i].cards.pop_back();
                    current_groups[i].visit_first.pop_back();
                    current_groups.insert(current_groups.begin() + i + 1, new_g);
                }
            }
            else if (op == 5) { // Merge Random
                if (current_groups.size() < 2) { possible = false; }
                else {
                    int i = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                    int j = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                    if (i == j) { possible = false; }
                    else {
                        for (size_t k = 0; k < current_groups[j].cards.size(); ++k) {
                            current_groups[i].cards.push_back(current_groups[j].cards[k]);
                            current_groups[i].visit_first.push_back(current_groups[j].visit_first[k]);
                        }
                        current_groups.erase(current_groups.begin() + j);
                    }
                }
            }
            else if (op == 6) { // Move Card between groups
                int i = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                int j = uniform_int_distribution<int>(0, (int)current_groups.size() - 1)(rng);
                if (i == j || current_groups[i].cards.empty()) { possible = false; }
                else {
                    int card_idx = uniform_int_distribution<int>(0, (int)current_groups[i].cards.size() - 1)(rng);
                    int c = current_groups[i].cards[card_idx];
                    bool vf = current_groups[i].visit_first[card_idx];
                    current_groups[i].cards.erase(current_groups[i].cards.begin() + card_idx);
                    current_groups[i].visit_first.erase(current_groups[i].visit_first.begin() + card_idx);
                    if (current_groups[i].cards.empty()) {
                        current_groups.erase(current_groups.begin() + i);
                        if (j > i) j--;
                    }
                    current_groups[j].cards.push_back(c);
                    current_groups[j].visit_first.push_back(vf);
                }
            }

            if (!possible) { continue; }

            int new_cost = 0;
            Pos p(0, 0);
            for (const auto& g : current_groups) {
                new_cost += manhattan_dist(p, g.get_entry_pos(input));
                new_cost += g.internal_cost(input);
                p = g.get_exit_pos(input);
            }

            if (accept(new_cost - old_cost, temp)) {
                current_cost = new_cost;
                update_best();
            } else {
                current_groups = prev_groups;
            }
        }

        cerr << "[DEBUG] Iterations: " << iterations << ", Best Cost: " << best_cost << endl;
        output(best_groups);
    }

private:
    bool accept(int delta, double temp) {
        if (delta <= 0) return true;
        return uniform_real_distribution<double>(0, 1)(rng) < exp(-delta / temp);
    }

    void update_best() {
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_groups = current_groups;
        }
    }

    int calc_segment_cost(int i) {
        if (i < 0 || i >= (int)current_groups.size()) return 0;
        Pos prev_exit = (i == 0) ? Pos(0, 0) : current_groups[i-1].get_exit_pos(input);
        return manhattan_dist(prev_exit, current_groups[i].get_entry_pos(input)) + current_groups[i].internal_cost(input);
    }

    // Technique 3 & 5: Calculate score difference locally
    int calc_swap_delta(int i, int j) {
        if (i > j) swap(i, j);
        int old_cost = 0;
        int new_cost = 0;

        // segments affected: i-1->i, i->i+1, j-1->j, j->j+1
        auto get_dist = [&](int idx, Pos prev_exit) {
            if (idx < 0 || idx >= (int)current_groups.size()) return 0;
            return manhattan_dist(prev_exit, current_groups[idx].get_entry_pos(input)) + current_groups[idx].internal_cost(input);
        };

        // Old
        old_cost += get_dist(i, (i == 0) ? Pos(0, 0) : current_groups[i-1].get_exit_pos(input));
        old_cost += get_dist(i+1, current_groups[i].get_exit_pos(input));
        if (j > i + 1) {
            old_cost += get_dist(j, current_groups[j-1].get_exit_pos(input));
        }
        if (j != i) {
            old_cost += get_dist(j+1, current_groups[j].get_exit_pos(input));
        }

        // New (Simulate swap)
        swap(current_groups[i], current_groups[j]);
        new_cost += get_dist(i, (i == 0) ? Pos(0, 0) : current_groups[i-1].get_exit_pos(input));
        new_cost += get_dist(i+1, current_groups[i].get_exit_pos(input));
        if (j > i + 1) {
            new_cost += get_dist(j, current_groups[j-1].get_exit_pos(input));
        }
        if (j != i) {
            new_cost += get_dist(j+1, current_groups[j].get_exit_pos(input));
        }
        swap(current_groups[i], current_groups[j]); // Revert simulation

        return new_cost - old_cost;
    }

    int calc_flip_delta(int i) {
        int old_c = calc_segment_cost(i);
        if (i + 1 < (int)current_groups.size()) old_c += manhattan_dist(current_groups[i].get_exit_pos(input), current_groups[i+1].get_entry_pos(input));

        for (size_t k = 0; k < current_groups[i].visit_first.size(); ++k) current_groups[i].visit_first[k] = !current_groups[i].visit_first[k];
        int new_c = calc_segment_cost(i);
        if (i + 1 < (int)current_groups.size()) new_c += manhattan_dist(current_groups[i].get_exit_pos(input), current_groups[i+1].get_entry_pos(input));
        for (size_t k = 0; k < current_groups[i].visit_first.size(); ++k) current_groups[i].visit_first[k] = !current_groups[i].visit_first[k];

        return new_c - old_c;
    }

    int calc_move_delta(int i, int j) {
        auto full_cost = [&](const vector<Group>& gs) {
            int c = 0;
            Pos cur(0, 0);
            for (const auto& g : gs) {
                c += manhattan_dist(cur, g.get_entry_pos(input));
                c += g.internal_cost(input);
                cur = g.get_exit_pos(input);
            }
            return c;
        };

        int old_f = full_cost(current_groups);
        Group gi = current_groups[i];
        vector<Group> next_gs = current_groups;
        next_gs.erase(next_gs.begin() + i);
        next_gs.insert(next_gs.begin() + j, gi);
        int new_f = full_cost(next_gs);
        return new_f - old_f;
    }

    int calc_merge_delta(int i, int j) {
        auto full_cost = [&](const vector<Group>& gs) {
            int c = 0;
            Pos cur(0, 0);
            for (const auto& g : gs) {
                c += manhattan_dist(cur, g.get_entry_pos(input));
                c += g.internal_cost(input);
                cur = g.get_exit_pos(input);
            }
            return c;
        };

        int old_f = full_cost(current_groups);
        vector<Group> next_gs = current_groups;
        for (size_t k = 0; k < next_gs[i+1].cards.size(); ++k) {
            next_gs[i].cards.push_back(next_gs[i+1].cards[k]);
            next_gs[i].visit_first.push_back(next_gs[i+1].visit_first[k]);
        }
        next_gs.erase(next_gs.begin() + i + 1);
        int new_f = full_cost(next_gs);
        
        return new_f - old_f;
    }

    int calc_split_delta(int i) {
        auto full_cost = [&](const vector<Group>& gs) {
            int c = 0;
            Pos cur(0, 0);
            for (const auto& g : gs) {
                c += manhattan_dist(cur, g.get_entry_pos(input));
                c += g.internal_cost(input);
                cur = g.get_exit_pos(input);
            }
            return c;
        };

        int old_f = full_cost(current_groups);
        vector<Group> next_gs = current_groups;
        Group new_g;
        new_g.cards.push_back(next_gs[i].cards.back());
        new_g.visit_first.push_back(next_gs[i].visit_first.back());
        next_gs[i].cards.pop_back();
        next_gs[i].visit_first.pop_back();
        next_gs.insert(next_gs.begin() + i + 1, new_g);
        int new_f = full_cost(next_gs);
        
        return new_f - old_f;
    }

    void init_greedy() {
        current_groups.clear();
        vector<bool> used(N * N / 2, false);
        int remaining = N * N / 2;
        Pos cur(0, 0);

        while (remaining > 0) {
            struct Candidate { int card; bool visit_first; int cost; };
            vector<Candidate> cands;
            for (int c = 0; c < N * N / 2; ++c) {
                if (used[c]) continue;
                int d1 = manhattan_dist(cur, input.card_pos[c].first);
                int d2 = manhattan_dist(cur, input.card_pos[c].second);
                cands.push_back({c, true, d1});
                cands.push_back({c, false, d2});
            }
            sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
            Candidate best = cands[0];
            
            Group g;
            g.cards.push_back(best.card);
            g.visit_first.push_back(best.visit_first);
            used[best.card] = true;
            remaining--;
            
            Pos inner_start = best.visit_first ? input.card_pos[best.card].first : input.card_pos[best.card].second;
            Pos inner_end = best.visit_first ? input.card_pos[best.card].second : input.card_pos[best.card].first;
            
            while (true) {
                int best_nest = -1;
                bool nest_first = true;
                int min_nest = INF;
                for (int c = 0; c < N * N / 2; ++c) {
                    if (used[c]) continue;
                    int cost1 = manhattan_dist(inner_start, input.card_pos[c].first) + manhattan_dist(input.card_pos[c].second, inner_end) - manhattan_dist(inner_start, inner_end);
                    int cost2 = manhattan_dist(inner_start, input.card_pos[c].second) + manhattan_dist(input.card_pos[c].first, inner_end) - manhattan_dist(inner_start, inner_end);
                    if (cost1 < min_nest) { min_nest = cost1; best_nest = c; nest_first = true; }
                    if (cost2 < min_nest) { min_nest = cost2; best_nest = c; nest_first = false; }
                }

                if (best_nest != -1 && min_nest <= 2) {
                    g.cards.push_back(best_nest);
                    g.visit_first.push_back(nest_first);
                    used[best_nest] = true;
                    remaining--;
                    inner_start = nest_first ? input.card_pos[best_nest].first : input.card_pos[best_nest].second;
                    inner_end = nest_first ? input.card_pos[best_nest].second : input.card_pos[best_nest].first;
                } else break;
            }
            cur = g.get_exit_pos(input);
            current_groups.push_back(g);
        }

        current_cost = 0;
        Pos p(0, 0);
        for (const auto& g : current_groups) {
            current_cost += manhattan_dist(p, g.get_entry_pos(input));
            current_cost += g.internal_cost(input);
            p = g.get_exit_pos(input);
        }
    }

    void append_path(string& path, int& move_count, const Pos& from, const Pos& to) {
        int dr = to.r - from.r;
        int dc = to.c - from.c;
        char r_char = (dr > 0) ? 'D' : 'U';
        char c_char = (dc > 0) ? 'R' : 'L';
        dr = abs(dr); dc = abs(dc);
        for (int i = 0; i < dr; i++) { path += r_char; move_count++; }
        for (int i = 0; i < dc; i++) { path += c_char; move_count++; }
    }

    void output(const vector<Group>& groups) {
        Pos cur(0, 0);
        int total_moves = 0;
        string ops;
        for (const auto& g : groups) {
            // Forward
            for (size_t i = 0; i < g.cards.size(); ++i) {
                Pos target = g.visit_first[i] ? input.card_pos[g.cards[i]].first : input.card_pos[g.cards[i]].second;
                append_path(ops, total_moves, cur, target);
                cur = target;
                ops += 'Z';
            }
            // Backward
            for (int i = (int)g.cards.size() - 1; i >= 0; --i) {
                Pos target = g.visit_first[i] ? input.card_pos[g.cards[i]].second : input.card_pos[g.cards[i]].first;
                append_path(ops, total_moves, cur, target);
                cur = target;
                ops += 'Z';
            }
        }
        for (char c : ops) cout << c << "\n";
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Solver solver;
    solver.solve();
    return 0;
}
