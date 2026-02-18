# =============================================================================
# AHC061 - THIRD Programming Contest 2026
# =============================================================================
# Usage:
#   make          - Build main solver
#   make vis      - Run solver and open visualizer
#   make gen      - Generate test cases
#   make test     - Run all tests
#   make fast     - Quick test on seeds 0-9
#   make check    - Crash detection
#   make submit   - Calculate score and archive submission
#   make clean    - Remove build artifacts
# =============================================================================

CXX = g++
CXXFLAGS = -std=c++23 -O2 -Wall -static -DLOCAL

# OS detection for Windows compatibility
ifeq ($(OS),Windows_NT)
    PYTHON = python
    EXE = .exe
else
    PYTHON = python3
    EXE =
endif

SOLVER = build/main$(EXE)

.PHONY: all clean run test vis gen fast check submit

# -----------------------------------------------------------------------------
# Build Targets
# -----------------------------------------------------------------------------

# Default: build main solver
all: $(SOLVER)

# Build main solver
$(SOLVER): src/main.cpp
	@if not exist build mkdir build
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build experimental versions
build/v%$(EXE): src/experimental/v%.cpp
	@if not exist build mkdir build
	$(CXX) $(CXXFLAGS) -o $@ $<

# -----------------------------------------------------------------------------
# Run & Visualize
# -----------------------------------------------------------------------------

# Run with sample input using official visualizer
# Usage: make vis CASE=0049.txt
CASE ?= 0000.txt
vis: $(SOLVER)
	@if not exist tmp mkdir tmp
	cd tools && cargo run -r --bin tester ../$(SOLVER) < in/$(CASE) > ../tmp/output.txt
	cd tools && cargo run -r --bin vis in/$(CASE) ../tmp/output.txt

# -----------------------------------------------------------------------------
# Test Case Generation
# -----------------------------------------------------------------------------

# Generate test cases using official generator
gen:
	cd tools && cargo run -r --bin gen seeds.txt

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

# Run all tests
test: $(SOLVER)
	$(PYTHON) scripts/test.py

# Quick test on seeds 0-9
fast: $(SOLVER)
	$(PYTHON) scripts/test.py -n 10

# Crash detection
check: $(SOLVER)
	$(PYTHON) scripts/check.py

# -----------------------------------------------------------------------------
# Submission
# -----------------------------------------------------------------------------

# Submit: Run all tests, save submission with score, copy to clipboard
submit: $(SOLVER)
	@if not exist submissions mkdir submissions
	$(PYTHON) scripts/test.py | tee tmp/test_result.txt
	@cp src/main.cpp submissions/main_latest.cpp
	@echo "Saved to: submissions/main_latest.cpp"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

# Clean build artifacts
clean:
	rm -rf build/*
	rm -rf tmp/*.txt
