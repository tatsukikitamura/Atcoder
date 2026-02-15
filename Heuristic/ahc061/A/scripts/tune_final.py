#!/usr/bin/env python3
"""
AHC061 Parameter Tuning with Optuna (Final Version)
Updates parameters of the C++ solver via environment variables.

Usage:
    pip install optuna
    python scripts/tune_final.py -n 50 -t 100 -j 4
"""

import optuna
import subprocess
import os
import re
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Configuration =====
BASE_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = BASE_DIR / "tools"
INPUT_DIR = TOOLS_DIR / "in"

if sys.platform == "win32":
    SOLVER_PATH = BASE_DIR / "build" / "main.exe"
else:
    SOLVER_PATH = BASE_DIR / "build" / "main"

# Parameter definitions based on current main.cpp:
# (optuna_name, env_name, default, low, high)
PARAM_DEFS = [
    ("W2_BASE",           "P_W2_BASE",           0.192222,  0.0,   1.0),
    ("W3_MULT",           "P_W3_MULT",           1.128094,  0.0,   3.0),
    ("W7_MAX",            "P_W7_MAX",            0.751249,  0.0,   3.0),
    ("RATIO_SCALE",       "P_RATIO_SCALE",       0.845678,  0.0,   5.0),
    ("W4_BASE",           "P_W4_BASE",           0.127005,  0.0,   0.5),
    ("W5_BASE",           "P_W5_BASE",           0.015127,  0.0,   0.1),
    ("REACH_DECAY",       "P_REACH_DECAY",       0.495684,  0.1,   2.0),
    ("QE_CAPTURE",        "P_QE_CAPTURE",        2.604286,  1.0,   10.0),
    ("QE_ATK_BONUS",      "P_QE_ATK_BONUS",      1.723014,  0.0,   10.0),
    ("QE_EMPTY_FUT",      "P_QE_EMPTY_FUT",      0.213099,  0.0,   0.5),
    ("COL_NEAR",          "P_COL_NEAR",          281.223607, 0.0,   600.0),
    ("COL_TARGET",        "P_COL_TARGET",        80.594577,  0.0,   300.0),
    ("W7_PHASE_START",    "P_W7_PHASE_START",    0.226949,  0.0,   0.8),
    ("RATIO_PHASE_START", "P_RATIO_PHASE_START", 0.155327,  0.0,   0.8),
    # M/U adaptation coefficients
    ("W2_BASE_U",         "P_W2_BASE_U",         -0.008985, -0.1,   0.1),
    ("W3_MULT_U",         "P_W3_MULT_U",         -0.033488, -0.2,   0.2),
    ("W7_MAX_M",          "P_W7_MAX_M",          -0.006833, -0.1,   0.1),
    ("W4_BASE_M",         "P_W4_BASE_M",         0.023649,  -0.1,   0.1),
    ("W7_PHASE_START_M",  "P_W7_PHASE_START_M",  -0.078900, -0.2,   0.2),
]

def build_solver():
    print("Building solver...")
    result = subprocess.run(["make"], cwd=str(BASE_DIR), capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

def run_single_case(args):
    input_file, env = args
    try:
        with open(input_file, 'r') as f_in:
            result = subprocess.run(
                ["cargo", "run", "-q", "-r", "--bin", "tester", str(SOLVER_PATH)],
                stdin=f_in, capture_output=True, text=True, cwd=str(TOOLS_DIR), timeout=30, env=env,
            )
        match = re.search(r"Score = (\d+)", result.stderr)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0

def evaluate_params(params_env, input_files, n_jobs):
    env = os.environ.copy()
    for k, v in params_env.items():
        env[k] = f"{v:.6f}"
    
    tasks = [(f, env) for f in input_files]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        scores = list(executor.map(run_single_case, tasks))
    return sum(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-cases", type=int, default=50)
    parser.add_argument("-t", "--trials", type=int, default=100)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    parser.add_argument("--db", type=str, default="optuna_final.db")
    args = parser.parse_args()

    if not SOLVER_PATH.exists():
        build_solver()

    input_files = sorted(INPUT_DIR.glob("*.txt"))[:args.num_cases]
    if not input_files:
        print(f"Error: No inputs in {INPUT_DIR}")
        sys.exit(1)

    print(f"Tuning {len(PARAM_DEFS)} parameters on {len(input_files)} cases...")

    # Optuna study の作成
    storage = f"sqlite:///{args.db}"
    # Optuna のログレベルを抑制（試行ごとのパラメータ表示を消す）
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize", storage=storage, study_name="ahc061_final", load_if_exists=True)
    
    # デフォルトパラメータをキューに追加
    study.enqueue_trial({name: default for name, _, default, _, _ in PARAM_DEFS})

    def objective(trial):
        params_env = {env_name: trial.suggest_float(optuna_name, low, high) 
                      for optuna_name, env_name, _, low, high in PARAM_DEFS}
        total = evaluate_params(params_env, input_files, args.jobs)
        print(f"Trial {trial.number:>3d}: Score={total:>9d}")
        return total

    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        pass

    print(f"\nBest score: {study.best_value}")
    print("Best parameters:")
    for optuna_name, env_name, _, _, _ in PARAM_DEFS:
        print(f"double {env_name:<18s} = {study.best_params[optuna_name]:.6f};")

if __name__ == "__main__":
    main()
