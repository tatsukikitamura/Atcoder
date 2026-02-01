# =============================================================================
# Heuristic Contest Makefile Template
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

CXX = clang++
CXXFLAGS = -std=c++23 -O2 -Wall

.PHONY: all clean run test vis gen fast check submit

# -----------------------------------------------------------------------------
# Build Targets
# -----------------------------------------------------------------------------

# Default: build main solver
all: build/main

# Build main solver
build/main: src/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build experimental versions
build/v%: src/experimental/v%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

# -----------------------------------------------------------------------------
# Run & Visualize
# -----------------------------------------------------------------------------

# Run with sample input using official visualizer
vis: build/main
	@mkdir -p tmp
	./build/main < tmp/input.txt > tmp/output.txt
	cd tools && cargo run -r --bin vis ../tmp/input.txt ../tmp/output.txt

# -----------------------------------------------------------------------------
# Test Case Generation
# -----------------------------------------------------------------------------

# Generate test cases using official generator
gen:
	cd tools && cargo run -r --bin gen seeds.txt --dir=../testcases/in

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

# Run all tests
test: build/main
	python3 scripts/test.py

# Quick test on seeds 0-9
fast: build/main
	python3 scripts/test.py -n 10

# Crash detection
check: build/main
	python3 scripts/check.py

# -----------------------------------------------------------------------------
# Submission
# -----------------------------------------------------------------------------

# Submit: Run all tests, save submission with score, copy to clipboard
submit: build/main
	@mkdir -p tmp submissions
	@echo "Running all test cases..."
	@total_score=0; \
	count=0; \
	for input in testcases/in/*.txt; do \
		name=$$(basename $$input .txt); \
		./build/main < $$input > tmp/out_$$name.txt 2>/dev/null; \
		score_line=$$(cd tools && cargo run -q -r --bin vis ../$$input ../tmp/out_$$name.txt 2>/dev/null | grep "Score ="); \
		if [ -n "$$score_line" ]; then \
			score=$$(echo $$score_line | cut -d ' ' -f 3); \
		else \
			score=0; \
		fi; \
		echo "Case $$name: $$score"; \
		total_score=$$((total_score + score)); \
		count=$$((count + 1)); \
	done; \
	echo "========================================"; \
	echo "Total Score: $$total_score ($$count cases)"; \
	echo "========================================"; \
	cp src/main.cpp submissions/main_$${total_score}.cpp; \
	echo "Saved to: submissions/main_$${total_score}.cpp"; \
	cat src/main.cpp | pbcopy; \
	echo "Source code copied to clipboard!"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

# Clean build artifacts
clean:
	rm -rf build/*
	rm -rf tmp/*.txt
