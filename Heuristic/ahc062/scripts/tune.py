#!/usr/bin/env python3
"""
Optuna Tuning Script for AHC062
"""

import optuna
import subprocess
import os
import re
import sys
import argparse
from datetime import datetime
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
    
    # Neighborhood probabilities (relative weights)
    p2opt = trial.suggest_int("p2opt", 10, 120)
    poropt = trial.suggest_int("poropt", 0, 120)
    plns = trial.suggest_int("plns", 0, 40)

    # LNS controls
    lns_min = trial.suggest_int("lns_min", 4, 32)
    lns_max = trial.suggest_int("lns_max", lns_min + 4, 200)
    lns_cands = trial.suggest_int("lns_cands", 4, 64)
    
    # Custom evaluation weights (-100 to 100)
    w1 = trial.suggest_float("w1", -100.0, 100.0)
    w2 = trial.suggest_float("w2", -100.0, 100.0)
    w3 = trial.suggest_float("w3", -100.0, 300.0)
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
        "poropt": poropt,
        "plns": plns,
        "lns-min": lns_min,
        "lns-max": lns_max,
        "lns-cands": lns_cands,
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

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuning for AHC062")
    parser.add_argument(
        "--db",
        type=str,
        default="",
        help="SQLite DB file path (example: ahc062_tune_lns.db). If omitted, a timestamped new DB is created.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ahc062_multi_evals",
        help="Optuna study name",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing study if present (default: create new study and fail if name already exists in DB)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(TMP_DIR, exist_ok=True)
    
    if not os.path.exists(SOLVER_PATH):
        print(f"Error: Solver not found at {SOLVER_PATH}")
        sys.exit(1)
        
    # Use SQLite for Optuna-Dashboard support
    if args.db:
        db_file = args.db
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_file = f"ahc062_tune_{ts}.db"
    storage_path = f"sqlite:///{db_file}"

    study = optuna.create_study(
        direction="maximize", 
        study_name=args.study_name,
        storage=storage_path,
        load_if_exists=args.resume
    )
    
    # Start from baseline: existing best param with 0 weights
    study.enqueue_trial({
        "t0": 5545.355848986292,
        "t1": 0.040546429892466644,
        "maxlen": 11276,
        "p2opt": 96,
        "poropt": 91,
        "plns": 34,
        "lns_min": 24,
        "lns_max": 75,
        "lns_cands": 32,
        "w1": -45.914168113550524,
        "w2": 68.95742290592759,
        "w3": 68.34295211938485,
        "w4": 75.33955658885995,
        "w5": -8.4887833674995,
        "w6": -93.42334825462912,
        "w7": -28.303654702102556,
        "w8": -55.350700494417225,
    })
    
    print("="*60)
    print(f"Starting Optuna tuning ({args.trials} trials, 50 tests/trial, 8 workers)")
    print(f"Storage: {storage_path}")
    print(f"Study: {args.study_name}")
    print(f"Run `optuna-dashboard {storage_path}` in another terminal to view results.")
    print("="*60)
    
    study.optimize(objective, n_trials=args.trials)
    
    print("\n" + "="*50)
    print("Optimization Result")
    print("="*50)
    print("Best parameters:")
    print(study.best_params)
    print(f"Best score (50 cases): {study.best_value}")

if __name__ == "__main__":
    main()
