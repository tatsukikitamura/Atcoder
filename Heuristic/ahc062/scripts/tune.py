#!/usr/bin/env python3
"""
Optuna Tuning Script for AHC062
"""

import optuna
import subprocess
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
SOLVER_PATH = "./build/main.exe"
INPUT_DIR = "tools/in"
TMP_DIR = "tmp"
TOOLS_DIR = "tools"

def run_single_case(input_file: Path, params: dict) -> int:
    name = input_file.stem
    output_file = Path(TMP_DIR) / f"out_{name}.txt"
    
    # cmd args
    cmd = [SOLVER_PATH]
    for k, v in params.items():
        cmd.extend([f"--{k}", str(v)])
        
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            subprocess.run(
                cmd,
                stdin=f_in,
                stdout=f_out,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
        
        # Get score from visualizer
        result = subprocess.run(
            ["cargo", "run", "-q", "-r", "--bin", "vis", 
             f"../{input_file}", f"../{output_file}"],
            capture_output=True,
            text=True,
            cwd=TOOLS_DIR
        )
        
        match = re.search(r"Score = (\d+)", result.stdout)
        if match:
            return int(match.group(1))
    except Exception as e:
        pass
    
    return 0

def objective(trial):
    # Parameter Search Space
    t0 = trial.suggest_float("t0", 100.0, 10000.0, log=True)
    t1 = trial.suggest_float("t1", 0.01, 100.0, log=True)
    maxlen = trial.suggest_int("maxlen", 1000, 30000)
    
    # Neighborhood probabilities
    p2opt = trial.suggest_int("p2opt", 0, 100)
    pswap = trial.suggest_int("pswap", 0, 100 - p2opt)
    
    # Custom evaluation weights (-100 to 100)
    w1 = trial.suggest_float("w1", -100.0, 100.0)
    w2 = trial.suggest_float("w2", -100.0, 100.0)
    w3 = trial.suggest_float("w3", -100.0, 100.0)
    w4 = trial.suggest_float("w4", -100.0, 100.0)
    w5 = trial.suggest_float("w5", -100.0, 100.0)
    w6 = trial.suggest_float("w6", -100.0, 100.0)
    w7 = trial.suggest_float("w7", -100.0, 100.0)
    w8 = trial.suggest_float("w8", -100.0, 100.0)
    
    params = {
        "t0": round(t0, 2),
        "t1": round(t1, 2),
        "maxlen": maxlen,
        "p2opt": p2opt,
        "pswap": pswap,
        "w1": round(w1, 2),
        "w2": round(w2, 2),
        "w3": round(w3, 2),
        "w4": round(w4, 2),
        "w5": round(w5, 2),
        "w6": round(w6, 2),
        "w7": round(w7, 2),
        "w8": round(w8, 2),
        "limit": 2.85 # Fixed time limit for tuning
    }
    
    # 1 trial evaluates exactly 50 cases as requested
    input_files = sorted(Path(INPUT_DIR).glob("*.txt"))[:50]
    total_score = 0
    
    # Run tests in parallel with 8 CPU workers
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_single_case, f, params): f for f in input_files}
        for future in as_completed(futures):
            score = future.result()
            total_score += score
            
    # We want to MAXIMIZE the score
    return total_score

def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    
    if not os.path.exists(SOLVER_PATH):
        print(f"Error: Solver not found at {SOLVER_PATH}")
        sys.exit(1)
        
    # Use SQLite for Optuna-Dashboard support
    storage_path = "sqlite:///ahc062_tune.db"
    study = optuna.create_study(
        direction="maximize", 
        study_name="ahc062_multi_evals",
        storage=storage_path,
        load_if_exists=True
    )
    
    # Start from baseline: existing best param with 0 weights
    study.enqueue_trial({
        "t0": 4420.73,
        "t1": 0.16,
        "maxlen": 27783,
        "p2opt": 35,
        "pswap": 42,
        "w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0,
        "w5": 0.0, "w6": 0.0, "w7": 0.0, "w8": 0.0
    })
    
    print("="*60)
    print("Starting Optuna tuning (100 trials, 50 tests/trial, 8 workers)")
    print("Run `optuna-dashboard sqlite:///ahc062_tune.db` in another terminal to view results.")
    print("="*60)
    
    # 100 trials
    study.optimize(objective, n_trials=100)
    
    print("\n" + "="*50)
    print("Optimization Result")
    print("="*50)
    print("Best parameters:")
    print(study.best_params)
    print(f"Best score (50 cases): {study.best_value}")

if __name__ == "__main__":
    main()
