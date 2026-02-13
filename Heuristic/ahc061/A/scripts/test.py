#!/usr/bin/env python3
"""
Test Runner - Run solver on all test cases via interactive tester.

Usage:
    python scripts/test.py           # Run all tests
    python scripts/test.py -n 10     # Run first 10 tests
    python scripts/test.py -j 4      # Use 4 parallel workers
"""

import subprocess
import os
import re
import sys
import argparse
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
SOLVER_PATH = "build/main"
INPUT_DIR = "tools/in"
TOOLS_DIR = "tools"


def get_solver_path():
    """Get the correct absolute solver path, checking for .exe on Windows."""
    base = os.path.abspath(SOLVER_PATH)
    if os.path.exists(base):
        return base
    if platform.system() == "Windows" and os.path.exists(base + ".exe"):
        return base + ".exe"
    return base


def run_single_case(input_file: Path) -> tuple:
    """Run solver on a single test case via tester and return (name, score)."""
    name = input_file.stem
    solver = get_solver_path()

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

        # Score is printed to stderr: "Score = N"
        match = re.search(r"Score = (\d+)", result.stderr)
        if match:
            return (name, int(match.group(1)))
        else:
            # Return error info for debugging
            err_msg = result.stderr.strip()[:200] if result.stderr else "no output"
            return (name, 0, err_msg)

    except subprocess.TimeoutExpired:
        return (name, -1)
    except Exception as e:
        print(f"Error on {name}: {e}", file=sys.stderr)
        return (name, 0)


def main():
    parser = argparse.ArgumentParser(description="Run tests and calculate score")
    parser.add_argument("-n", "--num", type=int, help="Number of test cases to run")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    # Check solver exists
    solver = get_solver_path()
    if not os.path.exists(solver):
        print(f"Error: Solver not found at {SOLVER_PATH}. Run 'make' first.")
        sys.exit(1)

    # Get test cases
    input_files = sorted(Path(INPUT_DIR).glob("*.txt"))
    if args.num:
        input_files = input_files[:args.num]

    if not input_files:
        print(f"Error: No test cases found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Running {len(input_files)} test cases with {args.jobs} workers...")
    print("=" * 50)

    # Run tests in parallel
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_single_case, f): f for f in input_files}
        for future in as_completed(futures):
            result = future.result()
            name, score = result[0], result[1]
            results.append((name, score))

            if score == -1:
                status = "TIMEOUT"
            elif score == 0 and len(result) > 2:
                status = f"ERROR: {result[2][:80]}"
                errors.append((name, result[2]))
            else:
                status = str(score)
            print(f"  {name}: {status}")

    # Summary
    results.sort(key=lambda x: x[0])
    scores = [s for _, s in results if s > 0]
    total_score = sum(scores)
    timeouts = sum(1 for _, s in results if s == -1)

    print("=" * 50)
    print(f"Total Score : {total_score}")
    if scores:
        print(f"Average     : {total_score / len(results):.0f}")
        print(f"Min         : {min(scores)}")
        print(f"Max         : {max(scores)}")
    if timeouts:
        print(f"Timeouts    : {timeouts}")
    if errors:
        print(f"Errors      : {len(errors)}")
    print(f"Cases       : {len(results)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
