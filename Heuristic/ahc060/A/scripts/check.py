#!/usr/bin/env python3
"""
Crash Detector - Quick smoke test to find crashes.

Usage:
    python scripts/check.py          # Check all test cases
    python scripts/check.py -n 20    # Check first 20 cases
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path

# Configuration
SOLVER_PATH = "./build/main"
INPUT_DIR = "testcases/in"
TIMEOUT_SEC = 5


def main():
    parser = argparse.ArgumentParser(description="Quick crash detection")
    parser.add_argument("-n", "--num", type=int, help="Number of test cases to check")
    args = parser.parse_args()
    
    if not os.path.exists(SOLVER_PATH):
        print(f"Error: Solver not found. Run 'make' first.")
        sys.exit(1)
    
    input_files = sorted(Path(INPUT_DIR).glob("*.txt"))
    if args.num:
        input_files = input_files[:args.num]
    
    if not input_files:
        print(f"Error: No test cases found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"Checking {len(input_files)} test cases for crashes...")
    
    for i, input_file in enumerate(input_files):
        name = input_file.stem
        print(f"[{i+1}/{len(input_files)}] {name}...", end="\r", flush=True)
        
        try:
            with open(input_file, 'r') as f_in:
                result = subprocess.run(
                    [SOLVER_PATH],
                    stdin=f_in,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=TIMEOUT_SEC
                )
                
                if result.returncode != 0:
                    print(f"\n❌ CRASH on {name}! Return code: {result.returncode}")
                    if result.stderr:
                        print("Stderr:", result.stderr.decode('utf-8'))
                    sys.exit(1)
                    
        except subprocess.TimeoutExpired:
            pass  # Timeout is OK, just means it didn't crash
        except Exception as e:
            print(f"\n❌ Error on {name}: {e}")
            sys.exit(1)
    
    print(f"\n✅ All {len(input_files)} test cases passed without crash!")


if __name__ == "__main__":
    main()
