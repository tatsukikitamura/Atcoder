---
description: Setup a new Heuristic Contest environment with standard directory structure
---

# Setup Heuristic Contest Workspace

When the user asks to setup a new Heuristic contest workspace (e.g., "Setup workspace for AHC056"), follow these steps to create a standardized environment.

---

## Step 1: Copy Template Directory

Copy the entire template directory to the contest location:

// turbo
```bash
cp -r /Users/kitamuratatuki/Atcoder/.agent/templates/heuristic Heuristic/<ContestName>/A
```

This creates the following structure:
```
Heuristic/<ContestName>/A/
├── src/
│   ├── main.cpp           # Main submission file (with Timer/Random utilities)
│   └── experimental/      # Experimental versions
├── tools/                  # Official Rust tools (download later)
│   └── seeds.txt          # Seeds for test case generation
├── testcases/
│   ├── in/                # Input files
│   └── out/               # Output files
├── scripts/
│   ├── compare.sh         # Run all tests and calculate score
│   └── optimize_params.py # Optuna parameter optimization
├── submissions/           # Archived submissions with score
├── build/                 # Build artifacts
├── tmp/                   # Temporary files
└── Makefile               # Build and test commands
```

---

## Step 2: Remind User About Official Tools

Remind the user to:
- Download the official tools (Rust) from the contest page
- Extract them to the `tools/` directory (overwriting seeds.txt is fine)
- Build with: `cd tools && cargo build --release`

---

## Step 3: Generate Test Cases

After tools are downloaded:

```bash
cd Heuristic/<ContestName>/A
make gen
```

---

## Step 4: Activate Virtual Environment (Optional)

For parameter optimization, activate the shared Python venv:

```bash
source ~/Atcoder/venv/bin/activate
```

---

## Available Make Commands

| Command | Description |
|---------|-------------|
| `make` | Build main solver |
| `make vis` | Run solver and open visualizer |
| `make gen` | Generate test cases |
| `make test` | Run all tests using compare script |
| `make fast-test` | Quick test on seeds 0-9 |
| `make submit` | Calculate score, archive, copy to clipboard |
| `make clean` | Remove build artifacts |

---

## Directory Structure Rationale

| Directory | Purpose |
|-----------|---------|
| `src/` | Main source code and experimental versions |
| `tools/` | Official AtCoder visualizer/tester tools |
| `testcases/` | Input/output test files |
| `scripts/` | Utility scripts (compare, optimize) |
| `submissions/` | Archived submissions with score in filename |
| `build/` | Compiled binaries |
| `tmp/` | Temporary outputs during testing |

This structure ensures consistency across all heuristic contests and enables reuse of optimization scripts and Makefiles.