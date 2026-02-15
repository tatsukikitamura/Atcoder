#!/usr/bin/env python3
"""
AHC061 Test Case Analyzer

Runs the solver on specified test cases and reports:
- Average, Min, Max scores
- Top 5 best cases (Seed + Score)
- Bottom 5 worst cases (Seed + Score)
"""

import subprocess
import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Configuration =====
BASE_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = BASE_DIR / "tools"
INPUT_DIR = TOOLS_DIR / "in"

# OS-aware solver path
if sys.platform == "win32":
    SOLVER_PATH = BASE_DIR / "build" / "main.exe"
else:
    SOLVER_PATH = BASE_DIR / "build" / "main"

def run_single_case(args):
    """Runs a single test case and returns (filename, score)."""
    input_file = args
    if not input_file.exists():
        return input_file.name, 0
        
    try:
        with open(input_file, 'r') as f_in:
            result = subprocess.run(
                ["cargo", "run", "-q", "-r", "--bin", "tester", str(SOLVER_PATH)],
                stdin=f_in,
                capture_output=True,
                text=True,
                cwd=str(TOOLS_DIR),
                timeout=10, 
            )
        
        stderr_str = result.stderr or ""
        match = re.search(r"Score = (\d+)", stderr_str)
        if match:
            return input_file.name, int(match.group(1))
        
        # Debug: Print stderr if score not found
        print(f"[DEBUG] {input_file.name} output: {stderr_str[:200]}...")
        # Check if we have a valid score in stdout? Some testers output to stdout.
        stdout_str = result.stdout or ""
        match_out = re.search(r"Score = (\d+)", stdout_str)
        if match_out:
            return input_file.name, int(match_out.group(1))

        return input_file.name, 0
    except subprocess.TimeoutExpired:
        print(f"Timeout: {input_file.name}")
        return input_file.name, 0
    except Exception as e:
        print(f"Error {input_file.name}: {e}")
        return input_file.name, 0

def main():
    parser = argparse.ArgumentParser(description="Analyze AHC061 Test Cases")
    parser.add_argument("-n", "--num", type=int, default=50, help="Number of cases to check")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel jobs")
    args = parser.parse_args()

    # Re-build ensuring correct path
    if not SOLVER_PATH.exists():
        print(f"Solver not found: {SOLVER_PATH}")
        sys.exit(1)

    input_files = sorted(INPUT_DIR.glob("*.txt"))[:args.num]
    
    # Run analysis
    results: List[Tuple[str, int]] = []
    print(f"Analyzing {len(input_files)} cases with {args.jobs} jobs...")
    
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_single_case, f): f for f in input_files}
        for future in as_completed(futures):
            fname, score = future.result()
            results.append((fname, score))
            if len(results) % 10 == 0:
                print(f"Processed {len(results)}/{len(input_files)}...", end='\r')

    print("\n")
    results.sort(key=lambda x: x[1])

    scores = [r[1] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    
    print(f"{'='*40}")
    print(f"  ANALYSIS RESULTS")
    print(f"{'='*40}")
    print(f"  Average : {avg:,.0f}")
    print(f"  Min     : {min(scores) if scores else 0}")
    print(f"  Max     : {max(scores) if scores else 0}")
    
    print(f"\n{'-'*40}")
    print(f"  WORST 5 CASES")
    print(f"{'-'*40}")
    for r in results[:10]:
        print(f"  {r[0]:<10}: {r[1]:>10,}")

    print(f"\n{'-'*40}")
    print(f"  BEST 5 CASES")
    print(f"{'-'*40}")
    for r in results[-10:]:
        print(f"  {r[0]:<10}: {r[1]:>10,}")

if __name__ == "__main__":
    main()
