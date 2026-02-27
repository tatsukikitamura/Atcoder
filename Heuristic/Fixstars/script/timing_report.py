#!/usr/bin/env python3
"""Run bench-solver (SOLVE_TIMING=ON) on every test case and write docs/timing_analysis.md.

Usage:
  python3 script/timing_report.py [--iters-small N] [--iters-large N]

Defaults: 10 iterations for small cases, 5 for large cases.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ     = Path(__file__).resolve().parent.parent
SOLVER   = PROJ / 'build' / 'bench' / 'bench-solver'
DATA_DIR = PROJ / 'data'
OUT_MD   = PROJ / 'docs' / 'timing_analysis.md'

# ---------------------------------------------------------------------------
# Natural sort
# ---------------------------------------------------------------------------
def natural_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', s)]


# ---------------------------------------------------------------------------
# Read N from input file (first line)
# ---------------------------------------------------------------------------
def read_N(path: Path) -> int:
    with open(path) as f:
        return int(f.readline().strip())


# ---------------------------------------------------------------------------
# Run one case, return (stdout, stderr) strings or (None, None) on timeout
# ---------------------------------------------------------------------------
def run_case(infile: Path, iters: int, timeout: int) -> tuple[str | None, str | None]:
    cmd = [str(SOLVER), str(infile), str(iters)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(PROJ))
        return r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return None, None


# ---------------------------------------------------------------------------
# Parse stdout:  "[path]  n=10  min=..us  median=..us  avg=..us  max=..us"
#                "  result: K=XX  T=XXXX"
# ---------------------------------------------------------------------------
def parse_stdout(stdout: str) -> dict:
    d = {}
    if stdout is None:
        return d
    m = re.search(r'n=(\d+)\s+min=([\d.]+)us\s+median=([\d.]+)us\s+avg=([\d.]+)us\s+max=([\d.]+)us', stdout)
    if m:
        d['iters']      = int(m.group(1))
        d['wall_min']   = float(m.group(2))
        d['wall_med']   = float(m.group(3))
        d['wall_avg']   = float(m.group(4))
        d['wall_max']   = float(m.group(5))
    m2 = re.search(r'result:\s+K=(\d+)\s+T=(\d+)', stdout)
    if m2:
        d['K'] = int(m2.group(1))
        d['T'] = int(m2.group(2))
    return d


# ---------------------------------------------------------------------------
# Parse stderr: section timing + distributions
# ---------------------------------------------------------------------------
SEC_RE   = re.compile(r'^\s+(init|prep \(FPT\)|search|output)\s+:\s+([\d.]+) us\s+\(\s*([\d.]+)%\)')
TOTAL_RE = re.compile(r'total\s+:\s+([\d.]+) us')
NTASK_RE = re.compile(r'\[(\d+) tasks generated\]')
DIST_RE  = re.compile(
    r'n=(\d+)\s+min=\s*([\d.]+)us\s+p25=\s*([\d.]+)us\s+med=\s*([\d.]+)us'
    r'\s+p75=\s*([\d.]+)us\s+max=\s*([\d.]+)us\s+total=\s*([\d.]+)us'
)
SKIP_RE  = re.compile(r'\((\d+) skipped')

def parse_stderr(stderr: str) -> dict:
    d = {}
    if stderr is None:
        return d
    for line in stderr.splitlines():
        m = SEC_RE.match(line)
        if m:
            key = m.group(1).replace(' ', '_').replace('(', '').replace(')', '')
            d[f'{key}_us']  = float(m.group(2))
            d[f'{key}_pct'] = float(m.group(3))
            m2 = NTASK_RE.search(line)
            if m2:
                d['n_tasks'] = int(m2.group(1))
        m = TOTAL_RE.search(line)
        if m:
            d['total_us'] = float(m.group(1))
        m = DIST_RE.search(line)
        if m:
            if 'fpt' in line.lower() or 'task' in line.lower():
                prefix = 'fpt'
            else:
                prefix = 'bnb'
            d[f'{prefix}_n']     = int(m.group(1))
            d[f'{prefix}_min']   = float(m.group(2))
            d[f'{prefix}_p25']   = float(m.group(3))
            d[f'{prefix}_med']   = float(m.group(4))
            d[f'{prefix}_p75']   = float(m.group(5))
            d[f'{prefix}_max']   = float(m.group(6))
            d[f'{prefix}_total'] = float(m.group(7))
        m = SKIP_RE.search(line)
        if m and 'bnb_skip' not in d:
            d['bnb_skip'] = int(m.group(1))
    return d


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
def fmt(v, decimals=2):
    if v is None:
        return '—'
    return f'{v:.{decimals}f}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters-small', type=int, default=10)
    parser.add_argument('--iters-large', type=int, default=5)
    parser.add_argument('--timeout',     type=int, default=120,
                        help='Per-case timeout in seconds (default: 120)')
    args = parser.parse_args()

    if not SOLVER.exists():
        print(f'ERROR: {SOLVER} not found. Build with -DSOLVE_TIMING=ON first.', file=sys.stderr)
        sys.exit(1)

    # Discover cases
    small_cases = sorted(
        [f.stem[3:] for f in DATA_DIR.glob('in-small-*.txt')],
        key=natural_key
    )
    large_cases = sorted(
        [f.stem[3:] for f in DATA_DIR.glob('in-large-*.txt')],
        key=natural_key
    )
    all_cases = [('small', tc) for tc in small_cases] + \
                [('large', tc) for tc in large_cases]

    total = len(all_cases)
    print(f'Running {total} cases  (small×{len(small_cases)}, large×{len(large_cases)})')

    results = []
    for idx, (kind, tc) in enumerate(all_cases, 1):
        infile = DATA_DIR / f'in-{tc}.txt'
        N      = read_N(infile)
        iters  = args.iters_small if kind == 'small' else args.iters_large
        print(f'  [{idx:>3}/{total}] {tc:20s}  N={N:<3}  ', end='', flush=True)
        stdout, stderr = run_case(infile, iters, args.timeout)
        if stdout is None:
            print('TIMEOUT')
            results.append({'tc': tc, 'kind': kind, 'N': N, 'timeout': True})
            continue
        d = {'tc': tc, 'kind': kind, 'N': N, 'timeout': False}
        d.update(parse_stdout(stdout))
        d.update(parse_stderr(stderr))
        results.append(d)
        wall = d.get('wall_med', 0)
        total_us = d.get('total_us', 0)
        print(f'wall_med={wall:9.2f}us  timing_total={total_us:9.2f}us  '
              f"K={d.get('K','?'):>3}  T={d.get('T','?')}")

    # -----------------------------------------------------------------------
    # Write Markdown
    # -----------------------------------------------------------------------
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, 'w') as f:
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        f.write(f'# Timing Analysis\n\n')
        f.write(f'Generated: {now}  \n')
        f.write(f'Binary: `build/bench/bench-solver` (SOLVE_TIMING=ON, -O2, -march=native)  \n')
        f.write(f'Iterations: small={args.iters_small}, large={args.iters_large}  \n\n')

        # ---- Summary table ----
        f.write('## Summary\n\n')
        f.write('| Case | N | K | T | wall med (us) | wall min (us) | wall max (us) | '
                'init (us) | prep (us) | search (us) | search% | output (us) | timing total (us) |\n')
        f.write('|------|---|---|---|--------------|--------------|--------------|'
                '----------|-----------|-------------|---------|-------------|-------------------|\n')
        for d in results:
            tc = d['tc']
            if d.get('timeout'):
                f.write(f"| {tc} | {d['N']} | — | — | TIMEOUT | — | — | — | — | — | — | — | — |\n")
                continue
            f.write(
                f"| {tc} | {d['N']} | {d.get('K','—')} | {d.get('T','—')} "
                f"| {fmt(d.get('wall_med'))} | {fmt(d.get('wall_min'))} | {fmt(d.get('wall_max'))} "
                f"| {fmt(d.get('init_us'))} | {fmt(d.get('prep_FPT_us', d.get('prep__FPT__us', 0.0)))} "
                f"| {fmt(d.get('search_us'))} | {fmt(d.get('search_pct'))} "
                f"| {fmt(d.get('output_us'))} | {fmt(d.get('total_us'))} |\n"
            )

        # ---- Small cases detail ----
        f.write('\n## Small Cases — FPT Path Detail\n\n')
        f.write('| Case | N | tasks | fpt n | fpt min | fpt p25 | fpt med | fpt p75 | fpt max | fpt total (us) |\n')
        f.write('|------|---|-------|-------|---------|---------|---------|---------|---------|---------------|\n')
        for d in results:
            if d.get('timeout') or d['kind'] != 'small':
                continue
            f.write(
                f"| {d['tc']} | {d['N']} | {d.get('n_tasks','—')} "
                f"| {d.get('fpt_n','—')} | {fmt(d.get('fpt_min'))} | {fmt(d.get('fpt_p25'))} "
                f"| {fmt(d.get('fpt_med'))} | {fmt(d.get('fpt_p75'))} | {fmt(d.get('fpt_max'))} "
                f"| {fmt(d.get('fpt_total'))} |\n"
            )

        # ---- Large cases detail ----
        f.write('\n## Large Cases — BnB Path Detail\n\n')
        f.write('| Case | N | skipped verts | bnb n | bnb min | bnb p25 | bnb med | bnb p75 | bnb max | bnb total (us) |\n')
        f.write('|------|---|---------------|-------|---------|---------|---------|---------|---------|----------------|\n')
        for d in results:
            if d.get('timeout') or d['kind'] != 'large':
                continue
            f.write(
                f"| {d['tc']} | {d['N']} | {d.get('bnb_skip','—')} "
                f"| {d.get('bnb_n','—')} | {fmt(d.get('bnb_min'))} | {fmt(d.get('bnb_p25'))} "
                f"| {fmt(d.get('bnb_med'))} | {fmt(d.get('bnb_p75'))} | {fmt(d.get('bnb_max'))} "
                f"| {fmt(d.get('bnb_total'))} |\n"
            )

        # ---- Top 10 heaviest ----
        f.write('\n## Top 10 Heaviest Cases (by wall_med)\n\n')
        ranked = [d for d in results if not d.get('timeout') and 'wall_med' in d]
        ranked.sort(key=lambda d: d.get('wall_med', 0), reverse=True)
        f.write('| Rank | Case | N | K | T | wall med (us) | search% | max task/vert (us) |\n')
        f.write('|------|------|---|---|---|--------------|---------|--------------------|\n')
        for rank, d in enumerate(ranked[:10], 1):
            max_detail = d.get('fpt_max') or d.get('bnb_max')
            f.write(
                f"| {rank} | {d['tc']} | {d['N']} | {d.get('K','—')} | {d.get('T','—')} "
                f"| {fmt(d.get('wall_med'))} | {fmt(d.get('search_pct'))} "
                f"| {fmt(max_detail)} |\n"
            )

        f.write('\n---\n*SOLVE_TIMING adds one extra warm solve() call at the end of each bench run.*\n')

    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
