#!/usr/bin/env python3
import sys
import random
import os
import subprocess
from pathlib import Path

def generate_testcase(is_large, seed=None):
    if seed is not None:
        random.seed(seed)
        
    N = 64 if is_large else 24
    
    # 1. グラフ生成 (p = 0.95)
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < 0.95:
                A[i][j] = 1
                A[j][i] = 1
                
    # 2. 能力値生成 [1, 100000]
    V = [[random.randint(1, 100000) for _ in range(128)] for _ in range(N)]
    
    # 3. BOSS要求値生成
    # Guarantee solvability by picking a random clique and bounding R by its sum
    nodes = list(range(N))
    random.shuffle(nodes)
    clique = []
    for u in nodes:
        if all(A[u][v] == 1 for v in clique):
             clique.append(u)
    
    clique_sums = [sum(V[u][k] for u in clique) for k in range(128)]
    
    # Generate R such that at least this clique satisfies it
    R = [random.randint(0, clique_sums[k]) for k in range(128)]
    
    # 出力フォーマットに合わせて出力
    lines = []
    lines.append(str(N))
    for row in A:
        lines.append(" ".join(map(str, row)))
    for row in V:
        lines.append(" ".join(map(str, row)))
    lines.append(" ".join(map(str, R)))
    
    return "\n".join(lines) + "\n"

def generate_all():
    proj_dir = Path(__file__).resolve().parent.parent
    data_dir = proj_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    solver_path = proj_dir / 'build' / 'run-solver'
    if not solver_path.exists():
        print("Solver not found. Running build.sh...")
        subprocess.run([str(proj_dir / 'build.sh')], cwd=str(proj_dir), check=True)

    print("Generating 10 small testcases...")
    for i in range(1, 11):
        seed = 100 + i # arbitrary distinct seeds
        content = generate_testcase(is_large=False, seed=seed)
        tc_name = f"small-{i}"
        in_path = data_dir / f"in-{tc_name}.txt"
        golden_path = data_dir / f"golden-{tc_name}.txt"
        
        with open(in_path, 'w') as f:
            f.write(content)
            
        subprocess.run([str(solver_path), str(in_path)], cwd=str(proj_dir), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        out_path = data_dir / f"out-{tc_name}.txt"
        if out_path.exists():
            out_path.replace(golden_path)

    print("Generating 20 large testcases...")
    for i in range(1, 21):
        seed = 200 + i # arbitrary distinct seeds
        content = generate_testcase(is_large=True, seed=seed)
        tc_name = f"large-{i}"
        in_path = data_dir / f"in-{tc_name}.txt"
        golden_path = data_dir / f"golden-{tc_name}.txt"
        
        with open(in_path, 'w') as f:
            f.write(content)
            
        subprocess.run([str(solver_path), str(in_path)], cwd=str(proj_dir), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        out_path = data_dir / f"out-{tc_name}.txt"
        if out_path.exists():
            out_path.replace(golden_path)
            
    print("Done! Generated 30 testcases total in 'data' directory.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen.py all")
        print("       python3 gen.py [small|large] [seed (optional)]")
        sys.exit(1)
        
    mode = sys.argv[1].lower()
    if mode == "all":
        generate_all()
    else:
        if mode not in ("small", "large"):
            print("Error: mode must be 'all', 'small', or 'large'")
            sys.exit(1)
            
        is_large = (mode == "large")
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
        
        print(generate_testcase(is_large, seed), end="")
