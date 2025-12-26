import subprocess
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
import time

def compile_solvers():
    print("Compiling 1.cpp...")
    subprocess.run(["g++", "-O3", "-std=c++17", "1.cpp", "-o", "main1"], check=True)
    print("Compiling 2.cpp...")
    subprocess.run(["g++", "-O3", "-std=c++17", "2.cpp", "-o", "main2"], check=True)

def run_solver(solver_executable, input_file):
    try:
        start_time = time.time()
        result = subprocess.run([solver_executable], stdin=open(input_file, "r"), capture_output=True, text=True, timeout=5)
        elapsed = time.time() - start_time
        
        # Regex to find "Best: <score>"
        match = re.search(r"Best:\s*(\d+)", result.stderr)
        if match:
            return int(match.group(1))
        
        # Fallback: Check output lines? Visualizer score?
        # Assuming code prints to stderr
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

def main():
    compile_solvers()
    
    input_files = sorted(glob.glob("in/*.txt"))
    results = []
    
    print(f"{'Case':<10} {'1.cpp':<10} {'2.cpp':<10} {'Diff':<10} {'Winner':<10}")
    print("-" * 55)
    
    total_1 = 0
    total_2 = 0
    wins_1 = 0
    wins_2 = 0
    
    # Run sequentially or parallel? Parallel is faster but might affect timing if CPU bound.
    # Since solvers are O3 and time-based (1.97s), running parallel heavily impacts performance.
    # Best to run sequentially or with very low parallelism if cores allow.
    # User's machine has multiple cores?
    # Safer to run sequentially to get ACCURATE individual performance, 
    # but 30 cases * 2 solvers * 2s = 120s = 2 mins.
    # Let's run with max_workers=4?
    # Or just sequential for accuracy. Sequential is best for benchmarking.
    
    for input_file in input_files:
        basename = os.path.basename(input_file)
        
        score1 = run_solver("./main1", input_file)
        score2 = run_solver("./main2", input_file)
        
        if score1 is None: score1 = 999999
        if score2 is None: score2 = 999999
        
        diff = score1 - score2
        winner = "1.cpp" if score1 < score2 else ("2.cpp" if score2 < score1 else "Draw")
        
        if winner == "1.cpp": wins_1 += 1
        if winner == "2.cpp": wins_2 += 1
        
        total_1 += score1
        total_2 += score2
        
        print(f"{basename:<10} {score1:<10} {score2:<10} {diff:<10} {winner:<10}")
        
    print("-" * 55)
    print(f"{'Total':<10} {total_1:<10} {total_2:<10}")
    print(f"Wins: 1.cpp={wins_1}, 2.cpp={wins_2}")

if __name__ == "__main__":
    main()
