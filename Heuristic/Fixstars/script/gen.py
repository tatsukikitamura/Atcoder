#!/usr/bin/env python3
"""Test-case generator for the Fixstars contest solver.

Usage:
  python3 gen.py all               # generate 100 small + 200 large cases (cleans old data)
  python3 gen.py small [seed]      # print a single small test case to stdout
  python3 gen.py large [seed]      # print a single large test case to stdout

'all' mode:
  - Writes in-small-{1..100}.txt and in-large-{1..200}.txt to data/
  - Runs the solver on each to produce golden-*.txt
  - Solver runs are parallelised with 4 workers (solver itself uses 8 OMP threads,
    so 4 concurrent = ~32 logical threads — safe on most dev machines).
  - Removes stale in-*/out-*/golden-* files that belong to old numbering before writing.
"""

import sys
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Test-case generation
# ---------------------------------------------------------------------------

def generate_testcase(is_large: bool, seed: int | None = None) -> str:
    if seed is not None:
        random.seed(seed)

    N = 64 if is_large else 24

    # 1. Graph with edge probability p=0.95
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < 0.95:
                A[i][j] = 1
                A[j][i] = 1

    # 2. Per-vertex capability vectors  [1, 100000]
    V = [[random.randint(1, 100_000) for _ in range(128)] for _ in range(N)]

    # 3. BOSS requirement vector — guaranteed satisfiable by construction:
    #    find a maximal clique and set R[k] <= sum of clique members for each k.
    nodes = list(range(N))
    random.shuffle(nodes)
    clique: list[int] = []
    for u in nodes:
        if all(A[u][v] for v in clique):
            clique.append(u)

    clique_sums = [sum(V[u][k] for u in clique) for k in range(128)]
    R = [random.randint(0, clique_sums[k]) for k in range(128)]

    lines: list[str] = [str(N)]
    for row in A:
        lines.append(" ".join(map(str, row)))
    for row in V:
        lines.append(" ".join(map(str, row)))
    lines.append(" ".join(map(str, R)))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Bulk generation helpers
# ---------------------------------------------------------------------------

def _run_solver_for_golden(
    solver: Path,
    proj_dir: Path,
    in_path: Path,
    golden_path: Path,
) -> str:
    """Run solver on *in_path* and rename the output to *golden_path*.
    Returns a human-readable status string."""
    tc = in_path.stem[3:]  # strip leading "in-"
    out_path = in_path.parent / f"out-{tc}.txt"
    if out_path.exists():
        out_path.unlink()

    result = subprocess.run(
        [str(solver), str(in_path)],
        cwd=str(proj_dir),
        capture_output=True,
    )
    if result.returncode != 0:
        return f"  [FAIL] {tc} (solver exit {result.returncode})"

    if out_path.exists():
        out_path.replace(golden_path)
        return f"  [OK]   {tc}"
    return f"  [WARN] {tc} — no output file produced"


def _clean_stale_files(data_dir: Path, kind: str, keep_count: int) -> None:
    """Remove in-/out-/golden- files for *kind* with index > keep_count."""
    for prefix in ("in", "out", "golden"):
        for f in data_dir.glob(f"{prefix}-{kind}-*.txt"):
            stem = f.stem  # e.g. "in-large-25"
            parts = stem.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx > keep_count:
                    f.unlink()


def generate_all(
    n_small: int = 100,
    n_large: int = 200,
    solver_workers: int = 4,
    seed_base_small: int = 1000,
    seed_base_large: int = 2000,
) -> None:
    proj_dir = Path(__file__).resolve().parent.parent
    data_dir = proj_dir / "data"
    data_dir.mkdir(exist_ok=True)

    solver_path = proj_dir / "build" / "run-solver"
    if not solver_path.exists():
        print("Solver not found — running build.sh ...")
        subprocess.run([str(proj_dir / "build.sh")], cwd=str(proj_dir), check=True)

    # Remove stale files from previous (smaller) runs
    _clean_stale_files(data_dir, "small", n_small)
    _clean_stale_files(data_dir, "large", n_large)

    # Build task list: (is_large, index, seed)
    tasks: list[tuple[bool, int, int]] = []
    for i in range(1, n_small + 1):
        tasks.append((False, i, seed_base_small + i))
    for i in range(1, n_large + 1):
        tasks.append((True, i, seed_base_large + i))

    print(f"Generating {n_small} small + {n_large} large = {len(tasks)} test cases "
          f"(solver parallelism: {solver_workers}) ...")

    # Step 1 — write all input files (fast, sequential)
    solver_jobs: list[tuple[Path, Path]] = []
    for is_large, i, seed in tasks:
        kind = "large" if is_large else "small"
        tc_name = f"{kind}-{i}"
        in_path = data_dir / f"in-{tc_name}.txt"
        golden_path = data_dir / f"golden-{tc_name}.txt"
        in_path.write_text(generate_testcase(is_large, seed))
        solver_jobs.append((in_path, golden_path))

    print(f"  Written {len(solver_jobs)} input files.")

    # Step 2 — run solver in parallel to produce golden files
    done = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=solver_workers) as pool:
        futs = {
            pool.submit(_run_solver_for_golden, solver_path, proj_dir, inp, gld): (inp, gld)
            for inp, gld in solver_jobs
        }
        for fut in as_completed(futs):
            msg = fut.result()
            done += 1
            print(f"\r  Solver progress: {done}/{len(solver_jobs)}", end="", flush=True)
            if "[FAIL]" in msg or "[WARN]" in msg:
                failed.append(msg)

    print(f"\r  Solver progress: {done}/{len(solver_jobs)}  Done.          ")
    if failed:
        print("\nProblems encountered:")
        for m in failed:
            print(m)
    print(f"\nDone! {len(solver_jobs) - len(failed)}/{len(solver_jobs)} golden files written to '{data_dir}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen.py all")
        print("       python3 gen.py [small|large] [seed]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "all":
        generate_all()
    elif mode in ("small", "large"):
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print(generate_testcase(mode == "large", seed), end="")
    else:
        print(f"Error: mode must be 'all', 'small', or 'large' (got '{mode}')")
        sys.exit(1)
