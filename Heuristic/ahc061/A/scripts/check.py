#!/usr/bin/env python3
"""
Crash Detector - Quick smoke test to find crashes via interactive tester.

Usage:
    python scripts/check.py          # Check all test cases
    python scripts/check.py -n 20    # Check first 20 cases
"""

import subprocess
import os
import re
import sys
import argparse
import platform
from pathlib import Path

# Configuration
SOLVER_PATH = "build/main"
INPUT_DIR = "tools/in"
TOOLS_DIR = "tools"
TIMEOUT_SEC = 30



def build_tester():
    """Build the tester binary."""
    print("Building tester...", end="", flush=True)
    try:
        subprocess.run(
            ["cargo", "build", "-r", "--bin", "tester"],
            cwd=TOOLS_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(" Done.")
    except subprocess.CalledProcessError:
        print("\nError: Failed to build tester.")
        sys.exit(1)


def get_tester_path():
    """Get path to tester binary."""
    ext = ".exe" if platform.system() == "Windows" else ""
    return os.path.abspath(os.path.join(TOOLS_DIR, "target", "release", "tester" + ext))


def get_solver_path():
    """Get the correct absolute solver path, checking for .exe on Windows."""
    base = os.path.abspath(SOLVER_PATH)
    if os.path.exists(base):
        return base
    if platform.system() == "Windows" and os.path.exists(base + ".exe"):
        return base + ".exe"
    return base


def main():
    parser = argparse.ArgumentParser(description="Quick crash detection")
    parser.add_argument("-n", "--num", type=int, help="Number of test cases to check")
    args = parser.parse_args()


    solver = get_solver_path()
    if not os.path.exists(solver):
        print(f"Error: Solver not found. Run 'make' first.")
        sys.exit(1)

    # Build tester
    build_tester()
    tester = get_tester_path()

    input_files = sorted(Path(INPUT_DIR).glob("*.txt"))
    if args.num:
        input_files = input_files[:args.num]

    if not input_files:
        print(f"Error: No test cases found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Checking {len(input_files)} test cases for crashes...")

    crashes = 0
    for i, input_file in enumerate(input_files):
        name = input_file.stem
        print(f"  [{i+1}/{len(input_files)}] {name}...", end="\r", flush=True)

        try:
            with open(input_file, 'r') as f_in:
                result = subprocess.run(
                    [tester, solver],
                    stdin=f_in,
                    capture_output=True,
                    text=True,
                    cwd=TOOLS_DIR,
                    timeout=TIMEOUT_SEC
                )

            # Check for errors: Score = 0 with error messages
            if result.returncode != 0:
                print(f"\n  CRASH on {name}! Return code: {result.returncode}")
                if result.stderr:
                    print(f"  Stderr: {result.stderr.strip()[:200]}")
                crashes += 1
            elif "Score = 0" in result.stderr:
                # Score 0 might indicate an error
                err_lines = [l for l in result.stderr.strip().split('\n') if 'Score' not in l]
                if err_lines:
                    print(f"\n  ERROR on {name}: {err_lines[0][:200]}")
                    crashes += 1

        except subprocess.TimeoutExpired:
            pass  # Timeout is OK for crash detection
        except Exception as e:
            print(f"\n  Error on {name}: {e}")
            crashes += 1

    if crashes == 0:
        print(f"\n  All {len(input_files)} test cases passed without crash!")
    else:
        print(f"\n  {crashes} crashes detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
