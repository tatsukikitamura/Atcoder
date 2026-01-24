/**
 * Heuristic Contest Solution Template
 * 
 * Usage:
 *   - Read input, solve, and output
 *   - Use cerr for debug output (won't affect scoring)
 */

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

// Timer utility for time-limited optimization
class Timer {
public:
    chrono::high_resolution_clock::time_point start_time;
    double time_limit;
    
    Timer(double limit_sec = 1.9) : time_limit(limit_sec) {
        start_time = chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start_time).count();
    }
    
    bool has_time() const {
        return elapsed() < time_limit;
    }
    
    double remaining() const {
        return max(0.0, time_limit - elapsed());
    }
};

// Fast Random using xorshift128+
class Random {
    uint64_t s[2];
public:
    Random(uint64_t seed = 42) {
        s[0] = seed;
        s[1] = seed * 0x123456789ABCDEF;
    }
    
    uint64_t next() {
        uint64_t x = s[0], y = s[1];
        s[0] = y;
        x ^= x << 23;
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s[1] + y;
    }
    
    int randint(int lo, int hi) {
        return lo + next() % (uint64_t)(hi - lo + 1);
    }
    
    double uniform(double lo = 0.0, double hi = 1.0) {
        return lo + (hi - lo) * (next() * (1.0 / UINT64_MAX));
    }
};

class Solver {
public:
    // Input variables
    int N;
    
    void read_input() {
        cin >> N;
        // TODO: Read problem-specific input
    }
    
    void solve() {
        Timer timer(1.9);  // Adjust time limit as needed
        Random rng(42);
        
        // TODO: Implement your solution
        // - Greedy initialization
        // - Local search / Simulated Annealing / etc.
        
        cerr << "Time: " << timer.elapsed() << "s" << endl;
    }
    
    void output() {
        // TODO: Output your solution
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    Solver solver;
    solver.read_input();
    solver.solve();
    solver.output();
    
    return 0;
}
