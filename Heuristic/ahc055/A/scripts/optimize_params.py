
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

# Paths
CPP_SOURCE = PROJECT_ROOT / "src" / "main.cpp"
EXECUTABLE = PROJECT_ROOT / "build" / "optimizer_bin"
INPUT_DIR = PROJECT_ROOT / "testcases" / "in"
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

def run_solver(prob_swap, prob_insert, input_file):
    # Unique output file for this run to avoid race conditions
    unique_id = uuid.uuid4().hex
    output_file = PROJECT_ROOT / "tmp" / f"out_{unique_id}.txt"
    
    try:
        # Run the solver
        cmd = [str(EXECUTABLE), str(prob_swap), str(prob_insert)]
        
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            subprocess.run(cmd, stdin=f_in, stdout=f_out, stderr=subprocess.DEVNULL, timeout=3.0, check=True)
            
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
        return float('inf')
        
    except subprocess.TimeoutExpired:
        return float('inf')
    except subprocess.CalledProcessError:
        return float('inf')
    except Exception as e:
        print(f"Error running solver: {e}")
        return float('inf')
    finally:
        # Cleanup
        if output_file.exists():
            output_file.unlink()

def objective(trial):
    # Suggest parameters
    # Original: PROB_SWAP=0.6, PROB_INSERT_FRONT=0.25
    prob_swap = trial.suggest_float("prob_swap", 0.0, 1.0)
    prob_insert = trial.suggest_float("prob_insert", 0.0, 1.0)
    
    total_score = 0
    
    # Use 10-20 test cases
    input_files = sorted(INPUT_DIR.glob("*.txt"))[:20]
    
    if not input_files:
        print("No input files found!")
        return float('inf')
        
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(run_solver, prob_swap, prob_insert, f) for f in input_files]
        scores = [f.result() for f in futures]
        
    for score in scores:
        if score == float('inf'):
             return float('inf')
        total_score += score
        
    return total_score

if __name__ == "__main__":
    compile_solver()
    
    study = optuna.create_study(direction="minimize")
    
    # Enqueue the default parameters to establish a baseline
    study.enqueue_trial({"prob_swap": 0.6, "prob_insert": 0.25})
    
    print("Starting optimization...")
    # Run for 50 trials
    study.optimize(objective, n_trials=50)
    
    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)
    
    # Save best params to a file or just print them clearly
    with open(PROJECT_ROOT / "tmp" / "best_params.txt", "w") as f:
        f.write(str(study.best_params))
