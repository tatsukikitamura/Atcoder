#include <bits/stdc++.h>
using namespace std;

// ===== Timer =====
struct Timer {
    chrono::high_resolution_clock::time_point start;
    double limit;
    
    Timer(double limit_sec = 1.9) : limit(limit_sec) {
        start = chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start).count();
    }
    
    bool is_over() const { return elapsed() >= limit; }
    double progress() const { return min(1.0, elapsed() / limit); }
};

// ===== Random =====
struct Rng {
    mt19937 gen;
    
    Rng() : gen(random_device{}()) {}
    Rng(unsigned seed) : gen(seed) {}
    
    int randint(int lo, int hi) {
        return uniform_int_distribution<int>(lo, hi)(gen);
    }
    
    double uniform(double lo = 0.0, double hi = 1.0) {
        return uniform_real_distribution<double>(lo, hi)(gen);
    }
    
    template<typename T>
    void shuffle(vector<T>& v) {
        std::shuffle(v.begin(), v.end(), gen);
    }
};

// ===== Input =====
struct Input {
    // TODO: Define input variables
    
    void read() {
        // TODO: Read input
    }
};

// ===== State =====
struct State {
    // TODO: Define state variables
    long long current_score = 0;
    
    void init(const Input& input) {
        // TODO: Initialize state (greedy or random)
    }
    
    long long score() const { return current_score; }
    
    void output() const {
        // TODO: Output solution
    }
};

// ===== Simulated Annealing =====
class SimulatedAnnealing {
public:
    Input input;
    State best;
    Timer timer;
    Rng rng;
    
    // SA parameters
    double start_temp = 1e4;
    double end_temp = 1e-1;
    int iteration_count = 0;
    
    double get_temp() {
        double t = timer.progress();
        return start_temp * pow(end_temp / start_temp, t);
    }
    
    bool accept(long long diff, double temp) {
        if (diff >= 0) return true;
        return rng.uniform() < exp(diff / temp);
    }
    
    void solve() {
        input.read();
        
        State current;
        current.init(input);
        best = current;
        
        while (!timer.is_over()) {
            double temp = get_temp();
            
            // Choose random operation
            int op = rng.randint(0, 2);
            
            // Calculate score difference (delta)
            long long diff = 0;
            
            switch (op) {
                case 0: {
                    // TODO: Operation 1 (e.g., swap)
                    break;
                }
                case 1: {
                    // TODO: Operation 2 (e.g., insert)
                    break;
                }
                case 2: {
                    // TODO: Operation 3 (e.g., move)
                    break;
                }
            }
            
            if (accept(diff, temp)) {
                // Apply the move
                current.current_score += diff;
                
                if (current.score() > best.score()) {
                    best = current;
                }
            } else {
                // Revert the move
                // TODO: Undo changes
            }
            
            iteration_count++;
        }
        
        cerr << "Iterations: " << iteration_count << endl;
        cerr << "Best Score: " << best.score() << endl;
        
        best.output();
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    SimulatedAnnealing sa;
    sa.solve();
    
    return 0;
}
