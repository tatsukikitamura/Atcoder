
import subprocess
import glob
import os
import concurrent.futures
import time

def run_case(args):
    """Run single case with params"""
    cmd, infile, t_start, t_end, p_swap, p_insert, sort_width, beam_width = args
    try:
        with open(infile, 'r') as f:
            # Run solver
            result = subprocess.run(
                [cmd, str(t_start), str(t_end), str(p_swap), str(p_insert), str(sort_width), str(beam_width)],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=10.0
            )
        # Parse score
        score = None
        for line in result.stderr.split('\n'):
            if "Best:" in line:
                try:
                    score = int(line.split("Best:")[-1].strip())
                except:
                    pass
        return score
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    solver = "./solver"
    in_dir = "in"
    test_files = sorted(glob.glob(os.path.join(in_dir, "*.txt")))
    
    # Best known params
    t_start = 125.0
    t_end = 0.01
    p_swap = 0.4
    p_insert = 0.4
    sort_width = 10
    
    # Parameters to tune
    beam_width_list = [10, 20, 30,35, 40]
    
    results = {}
    
    print(f"Starting tuning... {len(test_files)} files")
    start_time = time.time()
    
    for bw in beam_width_list:
        print(f"Testing BEAM_WIDTH={bw}...")
        
        tasks = []
        for tf in test_files:
            tasks.append((solver, tf, t_start, t_end, p_swap, p_insert, sort_width, bw))
        
        total_score = 0
        count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_case, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                s = future.result()
                if s is not None:
                    total_score += s
                    count += 1
        
        if count == len(test_files):
            print(f"  -> Total: {total_score}")
            results[bw] = total_score
        else:
            print(f"  -> Failed ({count}/{len(test_files)})")
    
    print("\n=== Tuning Results ===")
    best_score = float('inf')
    best_bw = None
    
    for bw, score in results.items():
        print(f"BW={bw} : {score}")
        if score < best_score:
            best_score = score
            best_bw = bw
            
    print(f"\nBest: BW={best_bw} (Score={best_score})")

if __name__ == "__main__":
    main()
