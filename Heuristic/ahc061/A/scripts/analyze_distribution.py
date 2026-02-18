"""
Score distribution analysis by M, U, and starting position density.
Usage: python scripts/analyze_distribution.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOLUTION_BIN = str(PROJECT_ROOT / "build" / "main.exe")
TESTER_BIN = str(PROJECT_ROOT / "tools" / "target" / "release" / "tester.exe")
TESTCASE_DIR = PROJECT_ROOT / "tools" / "in"


def parse_testcase(path):
    """Parse test case input: returns (N, M, T, U, V_grid, positions)"""
    with open(path) as f:
        lines = f.read().strip().split('\n')
    N, M, T, U = map(int, lines[0].split())
    V = []
    for i in range(1, N + 1):
        V.append(list(map(int, lines[i].split())))
    positions = []
    for i in range(N + 1, N + 1 + M):
        x, y = map(int, lines[i].split())
        positions.append((x, y))
    return N, M, T, U, V, positions


def compute_density(V, pos, radius=2):
    """Compute sum of V values within Manhattan distance `radius` from pos"""
    x0, y0 = pos
    N = len(V)
    total = 0
    count = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            nx, ny = x0 + dx, y0 + dy
            if 0 <= nx < N and 0 <= ny < N:
                total += V[nx][ny]
                count += 1
    return total, count


def compute_nearby_enemy_distance(positions, M):
    """Average Manhattan distance from player 0 to all other players"""
    x0, y0 = positions[0]
    if M <= 1:
        return 0
    total = 0
    for p in range(1, M):
        xp, yp = positions[p]
        total += abs(x0 - xp) + abs(y0 - yp)
    return total / (M - 1)


def run_single(testcase_path):
    """Run a single test case and return score"""
    try:
        with open(testcase_path, "r") as tc_input:
            result = subprocess.run(
                [TESTER_BIN, SOLUTION_BIN],
                stdin=tc_input,
                capture_output=True, text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT)
            )
        for line in (result.stderr + "\n" + result.stdout).split("\n"):
            m = re.search(r'[Ss]core\s*[=:]\s*([\d.]+)', line)
            if m:
                return float(m.group(1))
        return 0.0
    except Exception:
        return 0.0


def main():
    # Parse all test cases
    cases = sorted(TESTCASE_DIR.glob("*.txt"))
    if not cases:
        print("[ERROR] No test cases found")
        sys.exit(1)

    max_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cases = cases[:max_cases]
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print(f"[INFO] Analyzing {len(cases)} cases with {jobs} workers...")
    print(f"[INFO] Building...")

    # Build
    subprocess.run(
        "g++ -std=c++23 -O2 -Wall -static -o build/main.exe src/main.cpp",
        shell=True, cwd=str(PROJECT_ROOT), capture_output=True
    )

    # Parse test case metadata
    metadata = {}
    for tc in cases:
        name = tc.stem
        N, M, T, U, V, positions = parse_testcase(tc)
        density_sum, density_cnt = compute_density(V, positions[0], radius=2)
        avg_enemy_dist = compute_nearby_enemy_distance(positions, M)

        # Edge distance: how far from board edge is player 0
        x0, y0 = positions[0]
        edge_dist = min(x0, y0, N - 1 - x0, N - 1 - y0)

        metadata[name] = {
            "M": M, "U": U, "T": T,
            "density": density_sum,
            "density_avg": density_sum / max(1, density_cnt),
            "avg_enemy_dist": avg_enemy_dist,
            "edge_dist": edge_dist,
            "start_pos": positions[0],
        }

    # Run all test cases
    print(f"[INFO] Running test cases...")
    scores = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_single, str(tc)): tc.stem for tc in cases}
        done = 0
        for future in as_completed(futures):
            name = futures[future]
            scores[name] = future.result()
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(cases)}] completed...")

    print(f"\n{'='*70}")

    # === Analysis by M ===
    print("\n📊 Score by M (player count):")
    print(f"  {'M':>3} | {'Count':>5} | {'Average':>10} | {'Min':>8} | {'Max':>8} | {'MinCase':>8}")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    by_m = defaultdict(list)
    for name in scores:
        by_m[metadata[name]["M"]].append((scores[name], name))
    for m in sorted(by_m):
        vals = by_m[m]
        s = [v[0] for v in vals]
        min_case = min(vals, key=lambda x: x[0])
        print(f"  {m:>3} | {len(s):>5} | {sum(s)/len(s):>10.0f} | {min(s):>8.0f} | {max(s):>8.0f} | {min_case[1]:>8}")

    # === Analysis by U ===
    print("\n📊 Score by U (max level):")
    print(f"  {'U':>3} | {'Count':>5} | {'Average':>10} | {'Min':>8} | {'Max':>8} | {'MinCase':>8}")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    by_u = defaultdict(list)
    for name in scores:
        by_u[metadata[name]["U"]].append((scores[name], name))
    for u in sorted(by_u):
        vals = by_u[u]
        s = [v[0] for v in vals]
        min_case = min(vals, key=lambda x: x[0])
        print(f"  {u:>3} | {len(s):>5} | {sum(s)/len(s):>10.0f} | {min(s):>8.0f} | {max(s):>8.0f} | {min_case[1]:>8}")

    # === Analysis by (M, U) ===
    print("\n📊 Score by (M, U):")
    print(f"  {'M':>3} {'U':>3} | {'Count':>5} | {'Average':>10} | {'Min':>8} | {'Max':>8} | {'MinCase':>8}")
    print(f"  {'-'*3} {'-'*3}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    by_mu = defaultdict(list)
    for name in scores:
        key = (metadata[name]["M"], metadata[name]["U"])
        by_mu[key].append((scores[name], name))
    for key in sorted(by_mu):
        vals = by_mu[key]
        s = [v[0] for v in vals]
        min_case = min(vals, key=lambda x: x[0])
        print(f"  {key[0]:>3} {key[1]:>3} | {len(s):>5} | {sum(s)/len(s):>10.0f} | {min(s):>8.0f} | {max(s):>8.0f} | {min_case[1]:>8}")

    # === Analysis by Density ===
    print("\n📊 Score by starting position density (V sum within r=2):")
    density_vals = [metadata[n]["density"] for n in scores]
    d_min, d_max = min(density_vals), max(density_vals)
    d_step = (d_max - d_min) / 4
    bins = []
    for i in range(4):
        lo = d_min + i * d_step
        hi = d_min + (i + 1) * d_step if i < 3 else d_max + 1
        bins.append((lo, hi))

    print(f"  {'Density Range':>18} | {'Count':>5} | {'Average':>10} | {'Min':>8} | {'Max':>8}")
    print(f"  {'-'*18}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for lo, hi in bins:
        group = [(scores[n], n) for n in scores if lo <= metadata[n]["density"] < hi]
        if not group:
            continue
        s = [v[0] for v in group]
        print(f"  {lo:>8.0f}-{hi:>8.0f} | {len(s):>5} | {sum(s)/len(s):>10.0f} | {min(s):>8.0f} | {max(s):>8.0f}")

    # === Analysis by Avg Enemy Distance ===
    print("\n📊 Score by average enemy distance:")
    dist_vals = [metadata[n]["avg_enemy_dist"] for n in scores]
    d_min, d_max = min(dist_vals), max(dist_vals)
    d_step = (d_max - d_min) / 4
    bins_d = []
    for i in range(4):
        lo = d_min + i * d_step
        hi = d_min + (i + 1) * d_step if i < 3 else d_max + 1
        bins_d.append((lo, hi))

    print(f"  {'Dist Range':>14} | {'Count':>5} | {'Average':>10} | {'Min':>8} | {'Max':>8}")
    print(f"  {'-'*14}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for lo, hi in bins_d:
        group = [(scores[n], n) for n in scores if lo <= metadata[n]["avg_enemy_dist"] < hi]
        if not group:
            continue
        s = [v[0] for v in group]
        print(f"  {lo:>6.1f}-{hi:>6.1f} | {len(s):>5} | {sum(s)/len(s):>10.0f} | {min(s):>8.0f} | {max(s):>8.0f}")

    # === Analysis by Edge Distance ===
    print("\n📊 Score by edge distance (0=corner, 4=center area):")
    print(f"  {'Edge':>4} | {'Count':>5} | {'Average':>10} | {'Min':>8} | {'Max':>8}")
    print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    by_edge = defaultdict(list)
    for name in scores:
        by_edge[metadata[name]["edge_dist"]].append((scores[name], name))
    for e in sorted(by_edge):
        vals = by_edge[e]
        s = [v[0] for v in vals]
        print(f"  {e:>4} | {len(s):>5} | {sum(s)/len(s):>10.0f} | {min(s):>8.0f} | {max(s):>8.0f}")

    # === Worst 10 cases ===
    print("\n🔻 Worst 10 cases:")
    print(f"  {'Case':>6} | {'Score':>8} | {'M':>2} {'U':>2} | {'Density':>8} | {'EnemyDist':>9} | {'Edge':>4} | {'Pos':>7}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*2}-{'-'*2}-+-{'-'*8}-+-{'-'*9}-+-{'-'*4}-+-{'-'*7}")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    for name, score in sorted_scores[:10]:
        md = metadata[name]
        print(f"  {name:>6} | {score:>8.0f} | {md['M']:>2} {md['U']:>2} | {md['density']:>8.0f} | {md['avg_enemy_dist']:>9.1f} | {md['edge_dist']:>4} | ({md['start_pos'][0]},{md['start_pos'][1]})")

    # === Best 10 cases ===
    print("\n🔺 Best 10 cases:")
    print(f"  {'Case':>6} | {'Score':>8} | {'M':>2} {'U':>2} | {'Density':>8} | {'EnemyDist':>9} | {'Edge':>4} | {'Pos':>7}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*2}-{'-'*2}-+-{'-'*8}-+-{'-'*9}-+-{'-'*4}-+-{'-'*7}")
    for name, score in sorted_scores[-10:]:
        md = metadata[name]
        print(f"  {name:>6} | {score:>8.0f} | {md['M']:>2} {md['U']:>2} | {md['density']:>8.0f} | {md['avg_enemy_dist']:>9.1f} | {md['edge_dist']:>4} | ({md['start_pos'][0]},{md['start_pos'][1]})")

    # === Overall stats ===
    all_scores = list(scores.values())
    print(f"\n{'='*70}")
    print(f"  Total: {sum(all_scores):.0f}  Average: {sum(all_scores)/len(all_scores):.0f}  "
          f"Min: {min(all_scores):.0f}  Max: {max(all_scores):.0f}  Cases: {len(all_scores)}")


if __name__ == "__main__":
    main()
