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
    Copy the standard Makefile template:
    `cp /Users/kitamuratatuki/Atcoder/.agent/templates/heuristic/Makefile Heuristic/<ContestName>/A/Makefile`

    **`.gitignore`**
    Copy the standard gitignore template:
    `cp /Users/kitamuratatuki/Atcoder/.agent/templates/heuristic/.gitignore Heuristic/<ContestName>/A/.gitignore`

3.  **Setup Scripts**
    Copy standard Python/Shell scripts to `scripts/`:
    - `cp /Users/kitamuratatuki/Atcoder/.agent/templates/heuristic/optimize_params.py Heuristic/<ContestName>/A/scripts/`
    - `cp /Users/kitamuratatuki/Atcoder/.agent/templates/heuristic/compare.sh Heuristic/<ContestName>/A/scripts/`

4.  **Official Tools**
    Remind the user to download the official tools (Rust) from the contest page and place them in the `tools/` directory.

5.  **Virtual Environment**
    Remind the user to use the shared venv: `source ~/Atcoder/venv/bin/activate`

This structure ensures consistency across all heuristic contests and enables reuse of optimization scripts and Makefiles.
