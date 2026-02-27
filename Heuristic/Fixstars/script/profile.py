#!/usr/bin/env python3
"""Profile solve() internals using Valgrind/callgrind.

Builds a no-inline debug binary, runs callgrind on selected test cases,
then shows which functions / source lines in solve.cpp consume the most
instruction counts (Ir = proxy for CPU time).

Usage:
  python3 profile.py [options] [case...]

Options:
  --no-build         Skip the profiling build step
  --cases N          Number of hardest cases to auto-select (default: 3)
  --top N            Show top-N functions in the ranking (default: 20)
  --src              Also show annotated source lines for the top function

Examples:
  python3 profile.py                        # auto pick 3 hardest cases
  python3 profile.py large-104              # specific case
  python3 profile.py large-104 large-16     # multiple cases
  python3 profile.py --cases 5 --top 30    # 5 cases, 30-function table
  python3 profile.py large-104 --src        # with hot-line source view
"""

import argparse
import re
import subprocess
import sys
import tempfile
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
DIM    = '\033[2m'
NC     = '\033[0m'

def c(text, *codes):
    return ''.join(codes) + str(text) + NC


# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------
def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def natural_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', s)]


# ---------------------------------------------------------------------------
# Step 1: Build a no-inline debug binary in build_profile/
# ---------------------------------------------------------------------------
BUILD_DIR = 'build_profile'

def build_profile(proj: Path) -> bool:
    bd = proj / BUILD_DIR
    print(c('Building profiling binary (no-inline, -O2, -g) ...', BOLD))
    cmake_cmd = [
        'cmake', '-B', str(bd),
        '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
        '-DCMAKE_CXX_FLAGS=-O2 -fno-inline -g -fopenmp',
        '-DTARGET_ARCH=native',
    ]
    r = subprocess.run(cmake_cmd, capture_output=True, text=True, cwd=str(proj))
    if r.returncode != 0:
        print(c('cmake configure failed:', RED))
        print(r.stderr[-2000:])
        return False

    build_cmd = ['cmake', '--build', str(bd), '--target', 'run-solver', '-j4']
    r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=str(proj))
    if r.returncode != 0:
        print(c('Build FAILED:', RED))
        print(r.stderr[-2000:])
        return False

    print(c('Profiling build ready.', GREEN))
    return True


# ---------------------------------------------------------------------------
# Step 2: Pick test cases to profile
# ---------------------------------------------------------------------------
def pick_hardest(proj: Path, n: int) -> list[str]:
    """Return the n hardest cases by elapsed_nsec from existing out-*.txt."""
    data = proj / 'data'
    scored = []
    for f in data.glob('out-*.txt'):
        tc = f.stem[4:]         # strip "out-"
        in_file = data / f'in-{tc}.txt'
        if not in_file.exists():
            continue
        try:
            txt = f.read_text()
            m = re.search(r'elapsed_nsec\s*=\s*(\d+)', txt)
            if m:
                scored.append((int(m.group(1)), tc))
        except Exception:
            pass
    scored.sort(reverse=True)
    return [tc for _, tc in scored[:n]]


# ---------------------------------------------------------------------------
# Step 3: Run callgrind for one test case
# ---------------------------------------------------------------------------
def run_callgrind(proj: Path, tc: str, cg_file: Path) -> bool:
    solver = proj / BUILD_DIR / 'run-solver'
    in_file = proj / 'data' / f'in-{tc}.txt'
    if not solver.exists():
        print(c(f'Solver not found: {solver}', RED))
        return False
    if not in_file.exists():
        print(c(f'Input not found: {in_file}', RED))
        return False

    cmd = [
        'valgrind',
        '--tool=callgrind',
        '--instr-atstart=yes',
        f'--callgrind-out-file={cg_file}',
        str(solver),
        str(in_file),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(proj))
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Step 4: Parse callgrind_annotate output
# ---------------------------------------------------------------------------
SOLVE_CPP = 'solve.cpp'

# matches lines like:
#   3,456,789 (12.34%)  ???:funcname [binary]
#   3,456,789 (12.34%)  src/file.cpp:funcname [binary]
FN_RE = re.compile(
    r'^\s*([\d,]+)\s+\(\s*([\d.]+)%\)\s+'   # count + percent
    r'([^:]+):(.+?)'                          # file:function
    r'(?:\s+\[.*\])?$'
)
# matches annotated source lines:
#   3,456,789  int v_isub(...)
SRC_LINE_RE = re.compile(r'^\s*([\d,]+)\s+(.*)')


def parse_callgrind(cg_file: Path, proj: Path):
    """
    Returns:
      functions: list of (ir_count, pct, filename, funcname)
      annotated_src: list of (ir_count, lineno, source_text) for solve.cpp only
    """
    r = subprocess.run(
        ['callgrind_annotate', '--auto=yes', str(cg_file)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout:
        return [], []

    lines = r.stdout.splitlines()
    functions = []
    annotated_src = []

    # --- parse function table ---
    # Detect the header "Ir   file:function" then read until next "---" separator
    in_fn_table = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'^Ir\s+file:function', stripped):
            in_fn_table = True
            continue
        if not in_fn_table:
            continue
        if stripped.startswith('---') or stripped == '':
            if stripped.startswith('---'):
                in_fn_table = False
            continue
        # Format: "  3,456 (12.34%)  path/file:funcname [binary]"
        m = re.match(r'^\s*([\d,]+)\s+\(\s*([\d.]+)%\)\s+(.+?):(.+?)(?:\s+\[.+\])?$', line)
        if m:
            count = int(m.group(1).replace(',', ''))
            pct   = float(m.group(2))
            fname = m.group(3).strip()
            func  = m.group(4).strip()
            functions.append((count, pct, fname, func))

    # --- parse annotated source for solve.cpp ---
    # callgrind_annotate marks source sections with a line ending in ":"
    # that contains the filename, then lists annotated lines.
    in_src = False
    lineno = 0
    for line in lines:
        # Detect start of a source file annotation block
        # e.g. "/home/.../src/solve.cpp:"
        if line.rstrip().endswith(':') and SOLVE_CPP in line and '(' not in line:
            in_src = True
            lineno = 0
            continue
        if not in_src:
            continue
        # End marker
        if line.startswith('---'):
            in_src = False
            continue
        lineno += 1
        # Annotated line: "  3,456  source code here"  or "        . source code"
        m = re.match(r'^\s*([\d,]+)\s+(.*)', line)
        if m and m.group(1).replace(',', '').isdigit():
            count = int(m.group(1).replace(',', ''))
            src   = m.group(2)
        else:
            count = 0
            src   = line
        annotated_src.append((count, lineno, src.rstrip()))

    return functions, annotated_src


# ---------------------------------------------------------------------------
# Step 5: Display results
# ---------------------------------------------------------------------------
def display_functions(functions: list, top: int, title: str):
    if not functions:
        print(c('  (no function data found)', DIM))
        return

    total_ir = sum(f[0] for f in functions) or 1

    # filter to solve.cpp or our binary
    solve_fns = [(ir, pct, fn, func) for ir, pct, fn, func in functions
                 if SOLVE_CPP in fn]
    other_fns = [(ir, pct, fn, func) for ir, pct, fn, func in functions
                 if SOLVE_CPP not in fn]

    print(c(f'\n  {title}', BOLD, CYAN))
    print(c('  ─' * 38, CYAN))

    col_pct  = 8
    col_self = 14
    col_func = 46

    hdr = (f"  {c(f'{'%':>{col_pct}}', BOLD)}"
           f"  {c(f'{'Ir count':>{col_self}}', BOLD)}"
           f"  {c(f'{'Function':<{col_func}}', BOLD)}")
    print(hdr)
    print(f"  {'─'*col_pct}  {'─'*col_self}  {'─'*col_func}")

    shown = 0
    for ir, pct, fn, func in sorted(functions, reverse=True, key=lambda x: x[0])[:top]:
        in_solve = SOLVE_CPP in fn
        color   = YELLOW if in_solve else DIM
        fn_short = Path(fn).name if '/' in fn else fn
        label = f'{fn_short}:{func}'[:col_func]
        print(f"  {c(f'{pct:>{col_pct}.2f}%', color)}"
              f"  {c(f'{ir:>{col_self},}', color)}"
              f"  {c(f'{label:<{col_func}}', color)}")
        shown += 1


def display_hotlines(annotated_src: list, top_n: int = 20):
    """Show the hottest source lines from solve.cpp."""
    if not annotated_src:
        print(c('  (no source annotation data)', DIM))
        return

    # Sort by instruction count, filter out zero lines
    hot = [(ir, lno, src) for ir, lno, src in annotated_src if ir > 0]
    hot.sort(reverse=True)

    print(c(f'\n  Hot lines in {SOLVE_CPP} (top {top_n})', BOLD, CYAN))
    print(c('  ─' * 38, CYAN))

    col_pct  = 8
    col_lno  = 6
    col_src  = 60

    total = sum(ir for ir, _, _ in hot) or 1

    print(f"  {c(f'{'%':>{col_pct}}', BOLD)}"
          f"  {c(f'{'Line':>{col_lno}}', BOLD)}"
          f"  {c(f'{'Source':<{col_src}}', BOLD)}")
    print(f"  {'─'*col_pct}  {'─'*col_lno}  {'─'*col_src}")

    for ir, lno, src in hot[:top_n]:
        pct = ir / total * 100
        src_trunc = src.strip()[:col_src]
        intensity = RED if pct > 10 else (YELLOW if pct > 3 else NC)
        print(f"  {c(f'{pct:>{col_pct}.2f}%', intensity)}"
              f"  {c(f'{lno:>{col_lno}}', DIM)}"
              f"  {src_trunc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Profile solve() with callgrind',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('cases', nargs='*',
                        help='Test case names, e.g. large-104 small-1')
    parser.add_argument('--no-build', action='store_true',
                        help='Skip profiling build')
    parser.add_argument('--cases', type=int, default=3, dest='auto_cases',
                        metavar='N', help='Auto-select N hardest cases (default: 3)')
    parser.add_argument('--top', type=int, default=20,
                        help='Show top-N functions (default: 20)')
    parser.add_argument('--src', action='store_true',
                        help='Show hot source lines for each case')
    args = parser.parse_args()

    proj = project_dir()

    print(c('=' * 50, BOLD, CYAN))
    print(c('  Fixstars - Internal Profiler (callgrind)', BOLD, CYAN))
    print(c('=' * 50, BOLD, CYAN))
    print()

    # Build
    if not args.no_build:
        if not build_profile(proj):
            sys.exit(1)
        print()

    # Pick cases
    cases = args.cases or pick_hardest(proj, args.auto_cases)
    if not cases:
        print(c('No test cases found. Run bench.py first to generate out-*.txt.', RED))
        sys.exit(1)

    print(f"  Profiling {len(cases)} case(s): {', '.join(cases)}")
    print()

    # Accumulate across cases
    combined_functions: dict[tuple, int] = {}

    for tc in cases:
        print(c(f'  [{tc}] running callgrind...', BOLD), end='', flush=True)
        with tempfile.NamedTemporaryFile(suffix='.cg', delete=False) as tf:
            cg_path = Path(tf.name)

        ok = run_callgrind(proj, tc, cg_path)
        if not ok:
            print(c(' FAILED', RED))
            continue
        print(c(' done', GREEN))

        functions, annotated_src = parse_callgrind(cg_path, proj)
        cg_path.unlink(missing_ok=True)

        # Merge function data
        for ir, pct, fn, func in functions:
            key = (fn, func)
            combined_functions[key] = combined_functions.get(key, 0) + ir

        if args.src:
            display_hotlines(annotated_src, top_n=args.top)

    # Combined function table
    if combined_functions:
        total_ir = sum(combined_functions.values()) or 1
        fn_list = [
            (ir, ir / total_ir * 100, fn, func)
            for (fn, func), ir in combined_functions.items()
        ]
        fn_list.sort(reverse=True)
        display_functions(fn_list, args.top,
                          f'Function ranking (combined, {len(cases)} case(s))')
    print()


if __name__ == '__main__':
    main()
