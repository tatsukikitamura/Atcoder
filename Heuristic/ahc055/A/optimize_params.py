
import subprocess
import optuna
import os
import glob
import re
import sys

# Paths
CPP_SOURCE = "1.cpp"
EXECUTABLE = "optimizer_bin"
INPUT_DIR = "in"

def compile_solver():
    print("Compiling solver...")
    cmd = ["g++", "-O3", "-std=c++17", CPP_SOURCE, "-o", EXECUTABLE]
    subprocess.run(cmd, check=True)
    print("Compilation successful.")

def run_solver(prob_swap, prob_insert, input_file):
    try:
        # Run the solver with the suggested parameters
        # Passing parameters as command line arguments
        cmd = [f"./{EXECUTABLE}", str(prob_swap), str(prob_insert)]
        
        # Read input file
        with open(input_file, "r") as f:
            # Capture stderr because that's where "Best: " is printed
            result = subprocess.run(cmd, stdin=f, capture_output=True, text=True, timeout=3.0)
            
        # Parse output for "Best: <score>"
        match = re.search(r"Best:\s*(\d+)", result.stderr)
        if match:
            return int(match.group(1))
        return float('inf') # Return infinity if failed
    except subprocess.TimeoutExpired:
        return float('inf')
    except Exception as e:
        print(f"Error running solver: {e}")
        return float('inf')

def objective(trial):
    # Suggest parameters
    # Original values: PROB_SWAP = 0.6, PROB_INSERT_FRONT = 0.25
    prob_swap = trial.suggest_float("prob_swap", 0.0, 1.0)
    prob_insert = trial.suggest_float("prob_insert", 0.0, 1.0)
    
    total_score = 0
    
    # Use a subset of test cases for speed during optimization
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))[:10]
    
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
    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))
