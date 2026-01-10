#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
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
};

// ===== Input =====
struct Input {
    int N; // Number of turns/steps
    // TODO: Add more input variables
    
    void read() {
        cin >> N;
        // TODO: Read input
    }
};

// ===== State for Beam Search =====
struct State {
    // TODO: Define state
    vector<int> actions; // History of actions
    long long score = 0;
    
    // For priority queue (higher = better)
    bool operator<(const State& other) const {
        return score < other.score;
    }
    
    // Generate next states from this state
    vector<State> next_states(const Input& input) const {
        vector<State> result;
        
        // TODO: Generate all possible next states
        // Example:
        // for (int action = 0; action < num_actions; action++) {
        //     State next = *this;
        //     next.apply(action);
        //     next.actions.push_back(action);
        //     result.push_back(next);
        // }
        
        return result;
    }
    
    void output() const {
        for (int a : actions) {
            cout << a << "\n";
        }
    }
};

// ===== Beam Search =====
class BeamSearch {
public:
    Input input;
    int beam_width = 1000;
    
    State solve() {
        input.read();
        
        vector<State> current_beam;
        State initial;
        current_beam.push_back(initial);
        
        for (int turn = 0; turn < input.N; turn++) {
            vector<State> next_beam;
            
            for (const State& s : current_beam) {
                auto nexts = s.next_states(input);
                for (auto& ns : nexts) {
                    next_beam.push_back(move(ns));
                }
            }
            
            // Sort by score (descending) and keep top beam_width
            sort(next_beam.rbegin(), next_beam.rend());
            if ((int)next_beam.size() > beam_width) {
                next_beam.resize(beam_width);
            }
            
            current_beam = move(next_beam);
            
            if (current_beam.empty()) break;
        }
        
        if (current_beam.empty()) {
            return State();
        }
        
        return current_beam[0];
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    BeamSearch bs;
    State best = bs.solve();
    
    cerr << "Best Score: " << best.score << endl;
    best.output();
    
    return 0;
}
