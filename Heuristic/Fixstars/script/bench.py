#!/usr/bin/env python3
"""Fixstars Contest - Benchmark Runner
Runs all test cases N times and reports average/min/max elapsed times.

Usage:
  python3 bench.py [options] [filter...]

Options:
  --no-build         Skip build step
  --arch ARCH        Build architecture (rocketlake, icelake, native)
  -n N               Number of repetitions (default: 100)
  -j N               Run test cases in parallel within each iteration (default: 1)
  filter             Only run test cases whose names contain this string

Examples:
  python3 bench.py                        # build + run all 30 cases x100
  python3 bench.py -n 20                  # 20 repetitions
  python3 bench.py --no-build -n 50       # skip build, 50 reps
  python3 bench.py small -n 10            # only small cases, 10 reps
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
RED    = '\033[0;31m'
GREEN  = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN   = '\033[0;36m'
BOLD   = '\033[1m'
NC     = '\033[0m'

def c(text: str, *codes: str) -> str:
    return ''.join(codes) + str(text) + NC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def natural_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', s)]


def extract_field(text: str, field: str) -> Optional[str]:
    for line in text.splitlines():
        if line.startswith(field + ' =') or line.startswith(field + '='):
            parts = line.split('=', 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(proj: Path, arch: str) -> bool:
    print(c('Building solver...', BOLD))
    cmd = [str(proj / 'build.sh')]
    if arch:
        cmd.append(arch)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(proj))
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    for line in lines[-5:]:
        print('  ' + line)
    if proc.returncode != 0:
        print(c('Build FAILED.', RED, BOLD))
        return False
    print(c('Build complete.', GREEN))
    print()
    return True


# ---------------------------------------------------------------------------
# Single test run — returns elapsed in µs, or None on failure
# ---------------------------------------------------------------------------
def run_once(proj: Path, tc: str) -> Optional[float]:
    input_file  = proj / 'data' / f'in-{tc}.txt'
    output_file = proj / 'data' / f'out-{tc}.txt'

    if output_file.exists():
        output_file.unlink()

    solver = proj / 'build' / 'run-solver'
    subprocess.run(
        [str(solver), str(input_file)],
        capture_output=True,
        cwd=str(proj),
    )

    if not output_file.exists():
        return None

    out_text = output_file.read_text()
    raw = extract_field(out_text, 'elapsed_nsec')
    if raw is None:
        return None
    try:
        return int(raw) / 1e3  # ns → µs
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Fixstars Contest - Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--no-build', action='store_true', help='Skip build step')
    parser.add_argument('--arch', default='', help='Build architecture')
    parser.add_argument('-n', type=int, default=100, metavar='N',
                        help='Number of repetitions (default: 100)')
    parser.add_argument('-j', type=int, default=1, metavar='J',
                        help='Parallel workers per iteration (default: 1)')
    parser.add_argument('filter', nargs='*', help='Filter test cases by name')
    args = parser.parse_args()

    proj = project_dir()

    print(c('============================================', BOLD, CYAN))
    print(c('  Fixstars Contest - Benchmark Runner', BOLD, CYAN))
    print(c('============================================', BOLD, CYAN))
    print()

    if not args.no_build:
        if not build(proj, args.arch):
            sys.exit(1)

    # Discover test cases (golden only by default)
    data_dir = proj / 'data'
    test_cases = []
    for f in data_dir.glob('in-*.txt'):
        tc = f.stem[3:]
        test_cases.append(tc)
    test_cases.sort(key=natural_key)

    if args.filter:
        test_cases = [tc for tc in test_cases if any(p in tc for p in args.filter)]
    else:
        test_cases = [tc for tc in test_cases if (data_dir / f'golden-{tc}.txt').exists()]

    n_cases = len(test_cases)
    n_reps  = args.n
    parallel = max(1, args.j)

    print(f"  Test cases : {n_cases}")
    print(f"  Repetitions: {n_reps}")
    print(f"  Workers    : {parallel}")
    print()

    # times[tc] = list of elapsed µs
    times: dict[str, list[float]] = {tc: [] for tc in test_cases}

    def run_iteration(rep: int):
        """Run all cases once; return list of (tc, elapsed_or_None)."""
        results = []
        if parallel == 1:
            for tc in test_cases:
                results.append((tc, run_once(proj, tc)))
        else:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futs = {pool.submit(run_once, proj, tc): tc for tc in test_cases}
                for fut in as_completed(futs):
                    tc = futs[fut]
                    results.append((tc, fut.result()))
        return results

    # Run all repetitions
    for rep in range(1, n_reps + 1):
        print(f"\r  Progress: {rep}/{n_reps}", end='', flush=True)
        for tc, elapsed in run_iteration(rep):
            if elapsed is not None:
                times[tc].append(elapsed)

    print(f"\r  Progress: {n_reps}/{n_reps}  Done.          ")
    print()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    col_tc  = 24
    col_num =  8
    col_avg = 12
    col_min = 12
    col_max = 12

    hdr = (f"  {c(f'{'Test Case':<{col_tc}}', BOLD)}"
           f"  {c(f'{'Runs':>{col_num}}', BOLD)}"
           f"  {c(f'{'Avg (µs)':>{col_avg}}', BOLD)}"
           f"  {c(f'{'Min (µs)':>{col_min}}', BOLD)}"
           f"  {c(f'{'Max (µs)':>{col_max}}', BOLD)}")
    sep = (f"  {'─'*col_tc}  {'─'*col_num}  {'─'*col_avg}  {'─'*col_min}  {'─'*col_max}")

    print(c('============================================', BOLD, CYAN))
    print(c('  Per-case Statistics', BOLD, CYAN))
    print(c('============================================', BOLD, CYAN))
    print()
    print(hdr)
    print(sep)

    total_avg_sum = 0.0
    total_min_sum = 0.0
    total_max_sum = 0.0

    for tc in test_cases:
        data = times[tc]
        if not data:
            print(f"  {tc:<{col_tc}}  {'N/A':>{col_num}}  {'N/A':>{col_avg}}  {'N/A':>{col_min}}  {'N/A':>{col_max}}")
            continue
        avg = sum(data) / len(data)
        mn  = min(data)
        mx  = max(data)
        total_avg_sum += avg
        total_min_sum += mn
        total_max_sum += mx
        print(f"  {tc:<{col_tc}}  {len(data):>{col_num}}"
              f"  {avg:>{col_avg}.1f}  {mn:>{col_min}.1f}  {mx:>{col_max}.1f}")

    print(sep)
    print(f"  {c(f'{'TOTAL':<{col_tc}}', BOLD)}"
          f"  {' ':>{col_num}}"
          f"  {c(f'{total_avg_sum:>{col_avg}.1f}', CYAN)}"
          f"  {c(f'{total_min_sum:>{col_min}.1f}', GREEN)}"
          f"  {c(f'{total_max_sum:>{col_max}.1f}', RED)}")
    print()

    # Per-run total times (sum across cases)
    run_totals = []
    for rep_idx in range(n_reps):
        t = 0.0
        for tc in test_cases:
            if rep_idx < len(times[tc]):
                t += times[tc][rep_idx]
        run_totals.append(t)

    if run_totals:
        avg_total = sum(run_totals) / len(run_totals)
        min_total = min(run_totals)
        max_total = max(run_totals)
        print(f"  {c('Per-run total (sum of all cases):', BOLD)}")
        print(f"    Avg: {c(f'{avg_total:.1f}µs', CYAN)}"
              f"   Min: {c(f'{min_total:.1f}µs', GREEN)}"
              f"   Max: {c(f'{max_total:.1f}µs', RED)}")
    print()


if __name__ == '__main__':
    main()
