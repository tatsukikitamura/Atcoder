---
description: Systematic process for improving AtCoder Heuristic Contest solutions
---

Follow this cycle to iteratively improve the score during a Heuristic Contest.

1. **Baseline Implementation**
   - Goal: Get a valid solution for all test cases.
   - Approach: Implement a simple Greedy algorithm or Random construction.
   - Verify: Run `make fast-test` to ensure positive scores.

2. **Profiling & Analysis**
   - Goal: Find performance bottlenecks.
   - Action: Use `AutoTimer` (if available in template) or `std::chrono` to measure `Solve()` vs `IO` vs `Logic`.
   - Action: Check `make vis` output for seeds with poor scores.

3. **Algorithm Evolution**
   - **Phase 1 (Hill Climbing)**: Implement `Score()` and `Neighbor()`. Iteratively improve the state.
   - **Phase 2 (Simulated Annealing)**: Add temperature cooling to escape local optima.
     - Key Parameters: `StartTemp`, `EndTemp`, `TimeLimit`.
   - **Phase 3 (Beam Search)**: If state transitions are sequential, try Beam Search or Chokudai Search to explore wider.

4. **Parameter Optimization (Optuna)**
   - Goal: Tune magic numbers (probabilities, weights, temperatures).
   - Action: Expose parameters in `main.cpp` using `argv` or global constants.
   - Action: Update `scripts/optimize_params.py` with the parameter range.
   - Run: `python scripts/optimize_params.py` (ensure `venv` is active).

5. **Verification & Testing**
   - **Quick Check**: `make fast-test` (Seeds 0-9).
   - **Regression Test**: `bash scripts/compare.sh` (Compare current `build/main` with `build/master` or previous best).
   - **Visual Check**: `make vis` -> open `tools/vis.html`.

6. **Code Optimization (Low-level)**
   > **Note**: Apply these ONLY after profiling confirms a bottleneck, or as standard practice where noted.
   > **Primary References**: [Qiita 1](https://qiita.com/kotauchisunsun/items/84e01c6fb621fcc1a647) | [Qiita 2](https://qiita.com/ageprocpp/items/7bda728d109c953ece3c)

   ### Standard Practice (Apply Always)
   - **I/O Speed**:
     - **MUST**: Add `std::ios::sync_with_stdio(false); std::cin.tie(nullptr);` to `main()`.
     - **MUST**: Use `\n` instead of `std::endl`.
   - **Memory**:
     - **MUST**: Use `emplace_back` instead of `push_back` for vectors.
     - **MUST**: Call `reserve()` on vectors if the size is known or predictable.
   - **Compiler pragmas (for AtCoder/Codeforces)**:
     ```cpp
     #pragma GCC target("avx2")
     #pragma GCC optimize("O3")
     #pragma GCC optimize("unroll-loops")
     ```

   ### Bottleneck Optimizations (Apply if needed)
   - **Computation**:
     - **IF** performing heavy modular arithmetic: Use `constexpr` for modulus and precomputed tables.
     - **IF** frequent division: Refactor to multiplication (e.g., `x / 2` -> `x * 0.5` for floats, or bit shifts for integers).
     - **IF** frequent zero-init: Use `memset(arr, 0, sizeof(arr))` over `std::fill`.
     - **IF** recursion depth is deep: Rewrite DFS as iterative to avoid stack overhead.
   - **Memory & Cache**:
     - **IF** 2D array access is slow: Ensure row-major access (`arr[i][j]` inside loop `i` then `j`).
     - **IF** dynamic allocation is high: Use a pre-allocated memory pool (static array or large vector).
     - **IF** memory usage > 1GB: Reduce memory to prevent cache trashing and TLE.
   - **Local Testing**:
     - Use flags: `-O3 -mtune=native -march=native` to simulate contest environment execution.

