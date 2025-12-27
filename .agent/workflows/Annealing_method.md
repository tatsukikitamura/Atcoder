---
description: Simulated Annealing (SA) Best Practices & Implementation Guide for AI Agents
---

# Simulated Annealing (SA) Master Guide

This document is a comprehensive reference for AI agents implementing Simulated Annealing.
It combines a high-velocity code template with a detailed explanation of heuristics and optimizations.

---

## Part 1: AI Agent Quick Reference

### 1.1 When to use SA?
- **Good Targets**: Problems with "smooth" landscapes (TSP, Scheduling, Graph Coloring).
- **Bad Targets**: 
    - **Exact Cover / Subset Sum**: "Almost correct" is useless. Use Meet-in-the-Middle or DLX.
    - **Heavily Constrained**: Use Constructive/Greedy with Beam Search.

### 1.2 High-Performance Template (C++)

```cpp
// Fast Random (Xorshift)
struct Xorshift {
    uint64_t x = 88172645463325252ULL;
    inline uint64_t next() { x ^= x << 13; x ^= x >> 7; return x ^= x << 17; }
    inline int next_int(int n) { return next() % n; }
    inline double next_double() { return (double)next() / UINT64_MAX; }
};

// Global Context
Xorshift rng;
Timer timer;

void simulated_annealing() {
    // Constants
    const double TIME_LIMIT = 1.95;
    const double START_TEMP = 1000.0; // Tune this!
    const double END_TEMP = 1.0;      // Tune this!

    // State Init
    auto current_score = calc_score();
    auto best_score = current_score;
    // auto best_state = current_state; // Deep copy only if necessary

    double temp = START_TEMP;
    double log_ratio = log(END_TEMP / START_TEMP);
    
    int iter = 0;
    while (true) {
        iter++;
        
        // Batched Timer Check (Fast) - Every 256 iters
        if ((iter & 255) == 0) {
            double t = timer.elapsed();
            if (t >= TIME_LIMIT) break;
            double progress = t / TIME_LIMIT;
            temp = START_TEMP * exp(log_ratio * progress);
        }

        // 1. Neighbor Selection
        // ... select neighbor (swap, move, etc.) ...
        
        // 2. Delta Update (Critical: O(1))
        double delta = estimate_delta();

        // 3. Metropolis Criterion
        bool accept = (delta <= 0); // Minimization (Energy)
        if (!accept) {
            // Fast Probability Check
            // if (delta < temp * -log(rng.next_double())) accept = true; 
            if (rng.next_double() < exp(-delta / temp)) accept = true;
        }

        if (accept) {
            apply_move();
            current_score += delta;
            
            // Log Best
            if (current_score < best_score) {
                best_score = current_score;
                // save_best_state();
            }
        }
    }
}
```

---

## Part 2: Detailed Heuristics & Tricks

*(Detailed English translation of "Tips for Simulated Annealing" by shindannin)*

### 1. Speed Optimization
The more iterations, the better the score. Identify and crush bottlenecks.

*   **Profile Your Code**: Use a profiler (Perf, Visual Studio, etc.) before guessing.
*   **Time vs Score Correlation**: Experiment by extending time (e.g., 10s to 100s).
    *   *Score improves?* -> Optimize speed and neighbor selection (SA is working).
    *   *Score stuck & High?* -> You might have found the optimal; cut time short.
    *   *Score stuck & Low?* -> Bad neighborhood or score function. Rethink strategy.
*   **Delta Updates**: Never recalculate the full score $O(N)$. Calculate only the difference $O(1)$.
*   **Efficient Rollback**:
    *   Instead of copying the whole state to backup, store only the changed variable.
    *   Better yet, use **Inverse Operations** (e.g., call `swap(a,b)` again) to revert.
*   **Early Pruning**: If `delta` calculation exceeds a "hopeless" threshold mid-calculation, abort immediately.
*   **State-less Delta**: Calculate `delta` *before* modifying the state. If rejected, you save the cost of modifying and reverting the state.

### 2. Neighbor Selection Strategy
How you move determines how fast you climb.

*   **Move vs Swap**:
    *   *Move* (A -> B): Often creates imbalances. Requires 2 steps (A->B, B->A) to fix.
    *   *Swap* (A <-> B): Preserves balance. Can "tunnel" through bad states that *Move* would require.
    *   *Recommendation*: Use **Swap** as the primary operator, with occasional **Move** for diversity.
*   **Multiple Neighborhoods**: Don't limit to one type. Mix "Swap", "Shift", "Reverse", "Random Change".
    *   Collect stats on acceptance rates for each type to tune their probabilities.
*   **Greedy Initialization**: Start with a greedy solution. SA is better at refining than building from scratch.
*   **Adaptive Magnitude**:
    *   Early Phase: Large changes (Swap 100 times, Shift large range).
    *   Late Phase: Small changes (Swap 1 time, adjacent shift).
*   **"Tunneling" Moves**: Look for neighbors that change the state significantly but change the score very little. These allow traversing flat plateaus to find new basins.
*   **Reachability**: Ensure your operators can theoretically reach *any* valid state from the start.

### 3. Score Function Design
The problem's query is not essentially the best guide for the search.

*   **Smoothing**:
    *   Problem: "Maximize minimum satisfaction".
    *   Issue: `min(10, 10, 10)` same as `min(10, 1000, 1000)`. Gradient is zero.
    *   Fix: `Score = min_val + 0.0001 * sum_val` or `min_val + epsilon * second_min`.
*   **Eliminate Plateaus**: If `delta == 0` often, the search ends up in a random walk. Add tie-breaking terms.
*   **soft Constraints (Penalty Method)**:
    *   Instead of `if (!valid) continue;`, use `Score = raw_score + penalty * violation_magnitude`.
    *   Start with small penalty, increase it over time to force validity at the end.

### 4. Temperature Scheduling
*   **Exponential Schedule**: Standard `temp = start * pow(end/start, t)`.
*   **Start Temp**: Should allow ~50-80% of **bad** moves.
    *   Formula: $T_{start} \approx \Delta_{avg} / \ln(0.5)$ where $\Delta_{avg}$ is the average bad move delta.
*   **End Temp**: Should allow almost 0% of bad moves.
    *   Set $T_{end} \ll \Delta_{min}$ (smallest possible positive delta).

### 5. Meta-Strategies
*   **Multi-Start**: If the landscape is jagged (many deep local optima), running 10 short SAs is better than 1 long SA.
*   **Coordinate Descent**: If state variables are independent, anneal one variable (or subset) at a time while fixing others.
*   **Hill Climbing**: Sometimes `Temp = 0` (Greedy ascent) is faster and effectively as good. Always benchmark against it.

### 6. Debugging
*   **Delta Verification**: Run `assert(current_score == calc_full_score())` every 10k iterations. Mismatches are the #1 bug.
*   **Determinism**: Remove time-dependency for debugging. Use fixed iteration counts and fixed RNG seeds to reproduce bugs.