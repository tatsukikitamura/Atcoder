# Agent Configuration Directory

This directory contains configuration, instructions, and templates for the AI agent to manage the AtCoder project environment effectively.

## Structure

### `instructions.md`
Global instructions that the agent should follow. Includes:
*   Directory structure rules for Heuristic vs Algorithm contests.
*   Language standards (C++ version, Python venvs).

### `workflows/`
Step-by-step guides for complex tasks.
*   `setup-heuristic-contest.md`: Standard procedure for new AHC workspaces.
*   `ahc-improvement-loop.md`: Methodology for iterative solver improvement.
*   `Annealing_method.md`: Reference guide for simulated annealing.

### `templates/`
Reusable code and configuration files.
*   `heuristic/`: Contains `Makefile`, `optimize_params.py`, `compare.sh`.
*   `algorithm/`: (Currently empty) Placeholder for ABC/ARC templates.
