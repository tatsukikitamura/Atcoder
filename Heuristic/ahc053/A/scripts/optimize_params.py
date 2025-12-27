import subprocess
import optuna
import os
import glob
import re
import sys
from pathlib import Path

# Get project root (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

import threading
import uuid
import tempfile

# === CONFIGURATION ===
class Config:
    # Optimization settings
    N_TRIALS = 100             # Number of Optuna trials
    N_TEST_CASES = 10          # Number of test cases to run (0 means all)
    
    # Executable settings
    TIMEOUT = 3.9              # Timeout for each run in seconds (almost 4s limit)
    N_WORKERS = -1             # Number of parallel workers (-1 for auto/CPU count)
    
    # Parameter ranges to optimize
    # Format: "param_name": (min_val, max_val)
    # NOTE: Order matters! It must match command line arguments:
    # ./main [start_temp] [end_temp]
    PARAMS = {
        "start_temp": (1e11, 4e13),     # 10^11 to 4*10^13
        "end_temp": (1e1, 1e4)          # 10 to 10000
    }
    
    # Optimization direction
    DIRECTION = "maximize"     # "minimize" (Error) or "maximize" (Score)
    # Vis outputs AHC Contest Score (Higher is better)
    
# Paths
# CHANGE THIS: Path to the specific C++ source file you want to optimize
CPP_SOURCE = PROJECT_ROOT / "src" / "main.cpp" 
EXECUTABLE = PROJECT_ROOT / "build" / "main" # Update to main binary
INPUT_DIR = PROJECT_ROOT / "tools" / "in"
VIS_BINARY = PROJECT_ROOT / "tools" / "target" / "release" / "vis"

def compile_solver():
    print("Compiling solver...")
    # Compile C++ solver
    cmd = ["g++", "-O3", "-std=c++23", str(CPP_SOURCE), "-o", str(EXECUTABLE)]
    subprocess.run(cmd, check=True)
    
    # Check if visualizer exists
    if not VIS_BINARY.exists():
        print("Visualizer not found. Building...")
        subprocess.run(["cargo", "build", "-r", "--bin", "vis"], cwd=str(PROJECT_ROOT / "tools"), check=True)
        
    print("Compilation successful.")

def run_solver(param_values, input_file):
    # Unique output file for this run to avoid race conditions
    unique_id = uuid.uuid4().hex
    output_file = PROJECT_ROOT / "tmp" / f"out_{unique_id}.txt"
    
    try:
        # Run the solver
        # Construct command: ./optimizer_bin val1 val2 ...
        cmd = [str(EXECUTABLE)] + [str(p) for p in param_values]
        
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            subprocess.run(cmd, stdin=f_in, stdout=f_out, stderr=subprocess.DEVNULL, timeout=Config.TIMEOUT, check=True)
            
        # Run visualizer to get score
        # cargo run -r --bin vis in.txt out.txt
        vis_cmd = [str(VIS_BINARY), str(input_file), str(output_file)]
        result = subprocess.run(vis_cmd, capture_output=True, text=True, check=True)
        
        # Output format: "Score = 123456"
        match = re.search(r"Score = (\d+)", result.stdout)
        if match:
             score = int(match.group(1))
             # print(f"{input_file.name}: {score}")
             return score
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError:
        return None
    except Exception as e:
        print(f"Error running solver: {e}")
        return None
    finally:
        # Cleanup
        if output_file.exists():
            output_file.unlink()

def objective(trial):
    # Suggest parameters based on Config.PARAMS
    param_values = []
    
    # Iterate over Config.PARAMS
    for name, (low, high) in Config.PARAMS.items():
        if isinstance(low, int) and isinstance(high, int):
            val = trial.suggest_int(name, low, high)
        else:
            val = trial.suggest_float(name, low, high)
        param_values.append(val)
    
    total_score = 0
    
    # Select test cases
    files = sorted(INPUT_DIR.glob("*.txt"))
    if Config.N_TEST_CASES > 0:
        input_files = files[:Config.N_TEST_CASES]
    else:
        input_files = files
    
    # Determine penalty score for failures
    if Config.DIRECTION.lower() == "minimize":
        fail_score = float('inf')
    else:
        fail_score = float('-inf')

    if not input_files:
        print("No input files found!")
        return fail_score
        
    from concurrent.futures import ThreadPoolExecutor
    
    max_workers = Config.N_WORKERS if Config.N_WORKERS > 0 else (os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_solver, param_values, f) for f in input_files]
        scores = [f.result() for f in futures]
        
    for score in scores:
        if score is None:
             return fail_score
        total_score += score
        
    return total_score

if __name__ == "__main__":
    compile_solver()
    
    study = optuna.create_study(direction=Config.DIRECTION)
    
    # Enqueue the default parameters to establish a baseline if available
    # study.enqueue_trial({"prob_swap": 0.6, "prob_insert": 0.25})
    
    print("Starting optimization...")
    study.optimize(objective, n_trials=Config.N_TRIALS)
    
    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)
    
    # Save best params to a file or just print them clearly
    with open(PROJECT_ROOT / "tmp" / "best_params.txt", "w") as f:
        f.write(str(study.best_params))
