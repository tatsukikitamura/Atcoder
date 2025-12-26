CXX = g++
CXXFLAGS = -std=c++23 -O2 -Wall

.PHONY: all clean run test

# Default: build main solver
all: build/main

# Build main solver
build/main: src/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build experimental versions
build/v1: src/experimental/v1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

build/v2: src/experimental/v2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Run with sample input using official visualizer
vis: build/main
	./build/main < tmp/input.txt > tmp/output.txt
	cd tools && cargo run -r --bin vis ../tmp/input.txt ../tmp/output.txt
	open tools/vis.html

# Generate test cases using official generator
gen:
	cd tools && cargo run -r --bin gen seeds.txt --dir=../testcases/in

# Run tests using compare script
test: build/main
	bash scripts/compare.sh

# fast-test: Run tests on a small subset (0-10) using official scorer
fast-test: build/main
	@echo "Running fast test on seeds 0-9..."
	@total_score=0; \
	for i in $$(seq -f "%04g" 0 9); do \
		./build/main < testcases/in/$$i.txt > tmp/out_$$i.txt; \
		score=$$(cd tools && cargo run -q -r --bin vis ../testcases/in/$$i.txt ../tmp/out_$$i.txt 2>/dev/null | cut -d ' ' -f 3); \
		echo "Case $$i: $$score"; \
		total_score=$$((total_score + score)); \
	done; \
	echo "Total Score (0-9): $$total_score"

# Clean build artifacts
clean:
	rm -rf build/*
	rm -rf tmp/*.txt
