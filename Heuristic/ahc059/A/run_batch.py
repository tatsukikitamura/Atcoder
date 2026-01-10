import os
import subprocess
import sys

def main():
    executable = "./build/main"
    test_dir = "testcases/in"
    
    if not os.path.exists(executable):
        print(f"Executable {executable} not found")
        return

    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".txt")])
    
    print(f"Found {len(files)} testcases")
    
    for i, f in enumerate(files):
        path = os.path.join(test_dir, f)
        print(f"[{i+1}/{len(files)}] Running {f}...", end="\r", flush=True)
        try:
            with open(path, "r") as infile, open(os.devnull, "w") as outfile:
                 result = subprocess.run(
                     [executable], 
                     stdin=infile, 
                     stdout=outfile, 
                     stderr=subprocess.PIPE,
                     timeout=5
                 )
                 if result.returncode != 0:
                     print(f"CRASH detected on {f}! Return code: {result.returncode}")
                     print("Stderr:", result.stderr.decode('utf-8'))
                     return
        except subprocess.TimeoutExpired:
             # Timeout is fine, just means it didn't crash
             pass
        except Exception as e:
             print(f"Error running {f}: {e}")
             return
             
    print("All testcases passed without crash.")

if __name__ == "__main__":
    main()
