#!/usr/bin/env python3
"""Fixstars Contest - Test Runner
Builds the solver, runs all test cases, and compares with golden outputs.

Usage:
  python3 tests.py [options] [filter...]

Options:
  --no-build         Skip build step
  --arch ARCH        Build architecture (rocketlake, icelake, native)
  -j N               Run N tests in parallel (default: 1)
  filter             Only run test cases whose names contain this string
                     (can specify multiple)

Examples:
  python3 tests.py                        # build + run all
  python3 tests.py --no-build             # skip build, run all
  python3 tests.py small                  # run only "small" cases
  python3 tests.py custom-large          # run only custom-large cases
  python3 tests.py -j 4 --no-build       # parallel run, skip build
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
    """Wrap text with ANSI color codes."""
    return ''.join(codes) + str(text) + NC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def natural_key(s: str):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', s)]


def extract_field(text: str, field: str) -> Optional[str]:
    for line in text.splitlines():
        if line.startswith(field + ' =') or line.startswith(field + '='):
            parts = line.split('=', 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def extract_members(text: str) -> Optional[str]:
    members = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == 'PartyMembers = {':
            in_block = True
            continue
        if in_block:
            if stripped == '}':
                break
            members.extend(stripped.split())
    return ' '.join(members) if members else None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(proj: Path, arch: str, step_label: str) -> bool:
    print(c(f'{step_label} Building solver...', BOLD))
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
# Single test execution
# ---------------------------------------------------------------------------
def run_test(proj: Path, tc: str) -> dict:
    input_file  = proj / 'data' / f'in-{tc}.txt'
    output_file = proj / 'data' / f'out-{tc}.txt'
    golden_file = proj / 'data' / f'golden-{tc}.txt'

    result = {
        'tc':         tc,
        'status':     'SKIP',
        'out_score':  'N/A',
        'gold_score': 'N/A',
        'elapsed':    'N/A',
        'elapsed_num': 0.0,
        'detail':     '',
    }

    if not input_file.exists():
        result['detail'] = 'Input file not found'
        return result

    # Remove stale output
    if output_file.exists():
        output_file.unlink()

    solver = proj / 'build' / 'run-solver'
    subprocess.run(
        [str(solver), str(input_file)],
        capture_output=True,
        cwd=str(proj),
    )

    if not output_file.exists():
        result['status'] = 'FAIL'
        result['detail'] = 'Output file not generated'
        return result

    out_text = output_file.read_text()
    out_psum    = extract_field(out_text, 'ParameterSum')
    out_psize   = extract_field(out_text, 'PartySize')
    out_members = extract_members(out_text)
    out_elapsed = extract_field(out_text, 'elapsed_nsec')

    result['out_score'] = out_psum or 'N/A'
    if out_elapsed is not None:
        try:
            val = int(out_elapsed) / 1e3 # Convert nanoseconds to microseconds
            result['elapsed'] = f'{val:.1f}µs'
            result['elapsed_num'] = val
        except ValueError:
            result['elapsed'] = 'N/A'

    has_golden = golden_file.exists()
    if has_golden:
        gold_text    = golden_file.read_text()
        gold_psum    = extract_field(gold_text, 'ParameterSum')
        gold_psize   = extract_field(gold_text, 'PartySize')
        gold_members = extract_members(gold_text)
        result['gold_score'] = gold_psum or 'N/A'

        if out_psum != gold_psum:
            result['status'] = 'FAIL'
            result['detail'] = f'ParameterSum mismatch: got {out_psum}, expected {gold_psum}'
        elif out_psize != gold_psize:
            result['status'] = 'FAIL'
            result['detail'] = f'PartySize mismatch: got {out_psize}, expected {gold_psize}'
        elif out_members != gold_members:
            result['status'] = 'FAIL'
            result['detail'] = 'Members mismatch'
        else:
            result['status'] = 'PASS'
    else:
        result['status'] = 'PASS'
        result['detail'] = f'No golden file (T={out_psum})'

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Fixstars Contest - Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--no-build', action='store_true', help='Skip build step')
    parser.add_argument('--arch', default='', help='Build architecture')
    parser.add_argument('-j', type=int, default=1, metavar='N', help='Parallel jobs (default: 1)')
    parser.add_argument('filter', nargs='*', help='Run only test cases matching these strings')
    args = parser.parse_args()

    proj = project_dir()

    print(c('============================================', BOLD, CYAN))
    print(c('  Fixstars Contest - Test Runner', BOLD, CYAN))
    print(c('============================================', BOLD, CYAN))
    print()

    # --- Build ---
    if not args.no_build:
        ok = build(proj, args.arch, '[1/2]')
        if not ok:
            sys.exit(1)
        run_step_label = '[2/2]'
    else:
        run_step_label = '[1/1]'

    # --- Discover test cases ---
    data_dir = proj / 'data'
    
    # Remove all stale outputs upfront
    for f in data_dir.glob('out-*.txt'):
        f.unlink()

    test_cases = []
    for f in data_dir.glob('in-*.txt'):
        tc = f.stem[3:]   # strip leading "in-"
        test_cases.append(tc)
    test_cases.sort(key=natural_key)

    if args.filter:
        test_cases = [tc for tc in test_cases if any(pat in tc for pat in args.filter)]
    else:
        # デフォルトはゴールデンファイルがある公式ケースのみ
        test_cases = [tc for tc in test_cases if (proj / 'data' / f'golden-{tc}.txt').exists()]

    total = len(test_cases)
    parallel = max(1, args.j)
    parallel_note = f'  (parallel: {parallel})' if parallel > 1 else ''
    print(f"{c(run_step_label + ' Running tests...', BOLD)} ({total} test cases{parallel_note})")
    print()

    # --- Run tests ---
    results   = [None] * total   # preserve order
    completed = 0
    lock_print = __import__('threading').Lock()

    def _run(idx: int, tc: str) -> tuple[int, dict]:
        res = run_test(proj, tc)
        return idx, res

    def _print_result(idx: int, res: dict):
        nonlocal completed
        completed += 1
        progress = f'[{completed}/{total}]'
        tc = res['tc']
        status = res['status']
        if status == 'PASS':
            line = (f"  {c(progress, CYAN)}  {c('✓ PASS', GREEN)}"
                    f"  {tc:<24}  T={res['out_score']:<12}  {res['elapsed']}")
            if res['detail']:
                line += f"  {c(res['detail'], YELLOW)}"
        elif status == 'FAIL':
            line = (f"  {c(progress, CYAN)}  {c('✗ FAIL', RED)}"
                    f"  {tc:<24}  {res['detail']}  {res['elapsed']}")
        else:
            line = (f"  {c(progress, CYAN)}  {c('SKIP', YELLOW)}"
                    f"  {tc:<24}  {res['detail']}")
        print(line, flush=True)

    if parallel == 1:
        for idx, tc in enumerate(test_cases):
            progress = f'[{idx + 1}/{total}]'
            print(f"  {c(progress, CYAN)}  {c('Running', BOLD)} {tc}...", end='\r', flush=True)
            _, res = _run(idx, tc)
            results[idx] = res
            _print_result(idx, res)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_run, idx, tc): idx for idx, tc in enumerate(test_cases)}
            for fut in as_completed(futures):
                idx, res = fut.result()
                results[idx] = res
                with lock_print:
                    _print_result(idx, res)

    # --- Summary ---
    pass_count = sum(1 for r in results if r and r['status'] == 'PASS')
    fail_count = sum(1 for r in results if r and r['status'] == 'FAIL')
    skip_count = sum(1 for r in results if r and r['status'] == 'SKIP')
    total_time = sum(r['elapsed_num'] for r in results if r and 'elapsed_num' in r)

    # Contest score: sum of (my_time / best_time) per test case.
    # Best reference times (matching the contest server fastest known):
    #   small (N=24): 100 µs
    #   large (N=64): 150 µs
    BEST_TIME_SMALL_US = 100.0
    BEST_TIME_LARGE_US = 150.0

    def best_time_for(tc: str) -> float:
        return BEST_TIME_LARGE_US if 'large' in tc else BEST_TIME_SMALL_US

    print()
    print(c('============================================', BOLD, CYAN))
    print(c('  Summary', BOLD, CYAN))
    print(c('============================================', BOLD, CYAN))
    print()

    # Table header
    col_tc     = 24
    col_status =  8
    col_score  = 14
    col_time   = 10
    col_cscore =  8
    header = (f"  {c(f'{'Test Case':<{col_tc}}', BOLD)}"
              f"  {c(f'{'Status':<{col_status}}', BOLD)}"
              f"  {c(f'{'Output Score':<{col_score}}', BOLD)}"
              f"  {c(f'{'Golden Score':<{col_score}}', BOLD)}"
              f"  {c(f'{'Time':<{col_time}}', BOLD)}"
              f"  {c(f'{'CScore':<{col_cscore}}', BOLD)}")
    sep = (f"  {'─' * col_tc}  {'─' * col_status}  {'─' * col_score}"
           f"  {'─' * col_score}  {'─' * col_time}  {'─' * col_cscore}")
    print(header)
    print(sep)

    contest_score_total = 0.0
    contest_score_count = 0

    for res in results:
        if res is None:
            continue
        if res['status'] == 'PASS':
            status_colored = c(f"{'PASS':<{col_status}}", GREEN)
        elif res['status'] == 'FAIL':
            status_colored = c(f"{'FAIL':<{col_status}}", RED)
        else:
            status_colored = c(f"{'SKIP':<{col_status}}", YELLOW)

        # Contest score for this case: best_time / my_time (higher = better, 1.0 = matches fastest)
        t = res.get('elapsed_num', 0.0)
        if t > 0:
            cs = best_time_for(res['tc']) / t
            contest_score_total += cs
            contest_score_count += 1
            if cs >= 1.0:
                cs_str = c(f'{cs:.3f}', GREEN)
            elif cs >= 0.5:
                cs_str = c(f'{cs:.3f}', YELLOW)
            else:
                cs_str = c(f'{cs:.3f}', RED)
        else:
            cs_str = 'N/A'

        print(f"  {res['tc']:<{col_tc}}  {status_colored}  {res['out_score']:<{col_score}}"
              f"  {res['gold_score']:<{col_score}}  {res['elapsed']:<{col_time}}  {cs_str}")

    print()
    parts = [c(f'PASS: {pass_count}', GREEN), c(f'FAIL: {fail_count}', RED)]
    if skip_count:
        parts.append(c(f'SKIP: {skip_count}', YELLOW))
    parts.append(f'Total: {total}')
    parts.append(c(f'Total Time: {total_time:.1f}µs', CYAN))
    print('  ' + '  '.join(parts))
    print()

    # Contest score summary
    if contest_score_count > 0:
        avg_cs = contest_score_total / contest_score_count
        if contest_score_total >= contest_score_count:
            total_color = GREEN   # all cases at or above best reference
        elif contest_score_total >= contest_score_count * 0.5:
            total_color = YELLOW
        else:
            total_color = RED
        print(c('  Contest Score (higher = better, 1.0 = matches fastest known)', BOLD))
        print(f'  Reference: small={BEST_TIME_SMALL_US:.0f}µs  large={BEST_TIME_LARGE_US:.0f}µs')
        print(f'  {"Total score:":<16} {c(f"{contest_score_total:.4f}", total_color)}  '
              f'(avg per case: {c(f"{avg_cs:.4f}", total_color)},  cases: {contest_score_count})')
        print()

    if fail_count == 0:
        print(c('All tests passed!', BOLD, GREEN))
        sys.exit(0)
    else:
        print(c('Some tests failed.', BOLD, RED))
        sys.exit(1)


if __name__ == '__main__':
    main()
