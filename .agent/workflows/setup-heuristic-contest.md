---
description: Setup a new Heuristic Contest environment with standard directory structure
---

When the user asks to setup a new Heuristic contest workspace (e.g., "Setup workspace for AHC056"), follow these steps to create a standardized environment.

1.  **Create Directory Structure**
    Create the following directory structure under `Heuristic/<ContestName>/A`:
    ```
    Heuristic/<ContestName>/A/
    ├── src/                    # Source code
    │   ├── main.cpp           # Main submission file
    │   └── experimental/      # Experimental versions
    ├── tools/                  # Official Rust tools (download/copy later)
    ├── testcases/
    │   ├── in/                # Input files
    │   └── out/               # Output files
    ├── scripts/               # Utility scripts
    ├── build/                 # Build artifacts
    └── tmp/                   # Temporary files
    ```

2.  **Create Configuration Files**
    
    **`Makefile`**
    Create a Makefile with the following standard targets:
    - `all`: Build `src/main.cpp` to `build/main`
    - `vis`: Run with visualizer (`cargo run --bin vis`)
    - `gen`: Generate testcases (`cargo run --bin gen`)
    - `test`: Run comparison scripts
    - `clean`: Remove build artifacts

    **`.gitignore`**
    ```gitignore
    build/
    *.o
    tmp/
    testcases/in/
    tools/target/
    __pycache__/
    ```

3.  **Setup Scripts**
    Create standard Python/Shell scripts in `scripts/`:
    - `optimize_params.py`: Template for Optuna optimization using official `vis` tool.
    - `compare.sh`: Script to run and compare solvers.

4.  **Official Tools**
    Remind the user to download the official tools (Rust) from the contest page and place them in the `tools/` directory.

5.  **Virtual Environment**
    Remind the user to use the shared venv: `source ~/Atcoder/venv/bin/activate`

This structure ensures consistency across all heuristic contests and enables reuse of optimization scripts and Makefiles.
