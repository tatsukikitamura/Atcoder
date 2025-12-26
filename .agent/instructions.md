# Agent Instructions for AtCoder Project

When working on this project, please adhere to the following guidelines.

## Directory Structure & File Placement

### Heuristic Contests (AHC)
- **C++ Source Code**: Place all source files in `src/`.
  - Main submission: `src/main.cpp`
  - Experiments: `src/experimental/`
- **Python Scripts**: Place all utility scripts in `scripts/`.
- **Test Cases**: Read inputs from `testcases/in/`.
- **Official Tools**: Expect Rust tools in `tools/` and use them via `make gen` or `make vis`.

### Algorithm Contests (ABC/ARC/AGC)
- Place solver files in the contest directory (e.g., `ARC100/A/main.cpp`).

## Language Standards

### C++
- Use **C++23** or later (as per AtCoder 2025 environment).
- **Do not use `#include <bits/stdc++.h>`**. Instead, include standard headers explicitly (e.g., `<vector>`, `<algorithm>`, `<iostream>`).
- Avoid `using namespace std;` in header files or reusable libraries, but it is permitted in single-file contest submissions.

### Python
- Use the shared virtual environment: `source /Users/kitamuratatuki/Atcoder/venv/bin/activate`
- Follow PEP 8 style where reasonable.

## Workflow Templates
- When setting up a new contest environment, copy templates from `.agent/templates/`.
- Refer to `.agent/workflows/` for complex setup procedures.
