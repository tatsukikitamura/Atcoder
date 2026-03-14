#!/usr/bin/env python3
"""
Test Runner - Run solver on all test cases and calculate total score.

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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DEFAULT_SOLVER_PATH = "./build/main.exe"
INPUT_DIR = "tools/in"
TMP_DIR = "tmp"
TOOLS_DIR = "tools"


def run_single_case(input_file: Path, solver_path: str, solver_args: list) -> tuple[str, int]:
    """Run solver on a single test case and return (name, score)."""
    name = input_file.stem
    output_file = Path(TMP_DIR) / f"out_{name}.txt"
    
    cmd = [solver_path] + solver_args
    
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            subprocess.run(
                cmd,
                stdin=f_in,
                stdout=f_out,
                stderr=subprocess.DEVNULL,
                timeout=10
            )
        
        # Get score from visualizer
        result = subprocess.run(
            ["cargo", "run", "-q", "-r", "--bin", "vis", 
             f"../{input_file}", f"../{output_file}"],
            capture_output=True,
            text=True,
            cwd=TOOLS_DIR
        )
        
        match = re.search(r"Score = (\d+)", result.stdout)
        if match:
            return (name, int(match.group(1)))
    except subprocess.TimeoutExpired:
        return (name, -1)  # Timeout
    except Exception as e:
        print(f"Error on {name}: {e}", file=sys.stderr)
    
    return (name, 0)


def main():
    parser = argparse.ArgumentParser(description="Run tests and calculate score")
    parser.add_argument("-n", "--num", type=int, help="Number of test cases to run")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--solver", default=DEFAULT_SOLVER_PATH, help=f"Solver path (default: {DEFAULT_SOLVER_PATH})")
    args, unknown_args = parser.parse_known_args()
    
    # Setup
    os.makedirs(TMP_DIR, exist_ok=True)
    
    if not os.path.exists(args.solver):
        print(f"Error: Solver not found: {args.solver}")
        sys.exit(1)
    
    # Get test cases
    input_files = sorted(Path(INPUT_DIR).glob("*.txt"))
    if args.num:
        input_files = input_files[:args.num]
    
    if not input_files:
        print(f"Error: No test cases found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"Running {len(input_files)} test cases with {args.jobs} workers...")
    print(f"Solver: {args.solver}")
    if unknown_args:
        print(f"Solver args: {' '.join(unknown_args)}")
    print("=" * 40)
    
    # Run tests in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_single_case, f, args.solver, unknown_args): f for f in input_files}
        for future in as_completed(futures):
            name, score = future.result()
            results.append((name, score))
            status = "TIMEOUT" if score == -1 else str(score)
            print(f"Case {name}: {status}")
    
    # Summary
    results.sort(key=lambda x: x[0])
    total_score = sum(max(0, s) for _, s in results)
    timeouts = sum(1 for _, s in results if s == -1)
    
    print("=" * 40)
    print(f"Total Score: {total_score}")
    if timeouts:
        print(f"Timeouts: {timeouts}")
    print("=" * 40)


if __name__ == "__main__":
    main()
