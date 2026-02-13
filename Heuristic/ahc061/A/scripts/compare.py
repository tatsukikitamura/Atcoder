#!/usr/bin/env python3
"""
Compare multiple solvers on all test cases.
"""

import subprocess
import os
import re
import sys
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_DIR = "tools/in"
TOOLS_DIR = "tools"
JOBS = 4


def run_single_case(args):
    """Run solver on a single test case via tester and return (name, score)."""
    solver, input_file = args
    name = input_file.stem

    try:
        with open(input_file, 'r') as f_in:
            result = subprocess.run(
                ["cargo", "run", "-q", "-r", "--bin", "tester", solver],
                stdin=f_in,
                capture_output=True,
                text=True,
                cwd=TOOLS_DIR,
                timeout=30
            )

        match = re.search(r"Score = (\d+)", result.stderr)
        if match:
            return (name, int(match.group(1)))
        else:
            return (name, 0)

    except subprocess.TimeoutExpired:
        return (name, -1)
    except Exception as e:
        print(f"Error on {name}: {e}", file=sys.stderr)
        return (name, 0)


def test_solver(solver_path, input_files):
    """Test a solver on all input files and return dict of {name: score}."""
    solver = os.path.abspath(solver_path)
    if not os.path.exists(solver):
        print(f"Error: Solver not found at {solver_path}")
        return {}

    tasks = [(solver, f) for f in input_files]
    scores = {}
    with ProcessPoolExecutor(max_workers=JOBS) as executor:
        futures = {executor.submit(run_single_case, t): t for t in tasks}
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            name, score = future.result()
            scores[name] = score
            done += 1
            print(f"\r  Progress: {done}/{total}", end="", flush=True)
    print()
    return scores


def main():
    solvers = {
        "new.cpp": "build/new.exe",
        "8968805.cpp": "build/sub8968805.exe",
        "9261554.cpp": "build/sub9261554.exe",
    }

    input_files = sorted(Path(INPUT_DIR).glob("*.txt"))
    if not input_files:
        print(f"Error: No test cases found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Test cases: {len(input_files)}")
    print(f"Solvers: {', '.join(solvers.keys())}")
    print("=" * 70)

    all_results = {}
    for name, path in solvers.items():
        print(f"\nTesting: {name}")
        all_results[name] = test_solver(path, input_files)

    # Collect all test case names
    case_names = sorted(input_files[0].parent.parent.parent.joinpath("tools", "in").glob("*.txt"))
    case_names = sorted(set().union(*[r.keys() for r in all_results.values()]))

    # Print per-case comparison
    print("\n" + "=" * 70)
    print(f"{'Case':<10}", end="")
    for solver_name in solvers:
        print(f"  {solver_name:>14}", end="")
    print(f"  {'Best':>14}")
    print("-" * 70)

    solver_names = list(solvers.keys())
    win_counts = {name: 0 for name in solver_names}

    for case in case_names:
        print(f"{case:<10}", end="")
        case_scores = {}
        for solver_name in solver_names:
            score = all_results[solver_name].get(case, 0)
            case_scores[solver_name] = score
            print(f"  {score:>14}", end="")

        # Find the best (highest score)
        best_score = max(case_scores.values())
        best_solvers = [n for n, s in case_scores.items() if s == best_score]
        for bs in best_solvers:
            win_counts[bs] += 1
        print(f"  {'*' if len(best_solvers) > 1 else best_solvers[0]:>14}")

    # Summary
    print("=" * 70)
    print("\n--- SUMMARY ---")
    print(f"{'Solver':<20} {'Total Score':>12} {'Average':>10} {'Min':>8} {'Max':>8} {'Wins':>6} {'Timeouts':>9}")
    print("-" * 75)

    for solver_name in solver_names:
        scores_list = [all_results[solver_name].get(c, 0) for c in case_names]
        valid = [s for s in scores_list if s > 0]
        total = sum(valid)
        avg = total / len(case_names) if case_names else 0
        min_s = min(valid) if valid else 0
        max_s = max(valid) if valid else 0
        timeouts = sum(1 for s in scores_list if s == -1)
        wins = win_counts[solver_name]
        print(f"{solver_name:<20} {total:>12} {avg:>10.0f} {min_s:>8} {max_s:>8} {wins:>6} {timeouts:>9}")

    # Find overall best
    totals = {}
    for solver_name in solver_names:
        scores_list = [all_results[solver_name].get(c, 0) for c in case_names]
        totals[solver_name] = sum(s for s in scores_list if s > 0)

    best_solver = max(totals, key=totals.get)
    print(f"\n*** BEST SOLVER: {best_solver} (Total Score: {totals[best_solver]}) ***")


if __name__ == "__main__":
    main()
