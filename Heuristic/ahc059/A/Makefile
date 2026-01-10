CXX = g++
CXXFLAGS = -std=c++23 -O2 -Wall

.PHONY: all clean run test

# デフォルト: メインソルバーをビルド
all: build/main

# メインソルバーをビルド
build/main: src/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

# 実験用バージョンをビルド
build/v1: src/experimental/v1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

build/v2: src/experimental/v2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# 公式ビジュアライザを使ってサンプル入力で実行
vis: build/main
	./build/main < tmp/input.txt > tmp/output.txt
	cd tools && cargo run -r --bin vis ../tmp/input.txt ../tmp/output.txt
	open tools/vis.html

# 公式ジェネレータでテストケースを生成
gen:
	cd tools && cargo run -r --bin gen seeds.txt --dir=../testcases/in

# compareスクリプトでテストを実行
test: build/main
	bash scripts/compare.sh

# fast-test: 公式スコアラーで小さなサブセット(0-9)のテストを実行
fast-test: build/main
	@echo "Running fast test on seeds 0-9..."
	@total_score=0; \
	for i in $$(seq -f "%04g" 0 9); do \
		./build/main < testcases/in/$$i.txt > tmp/out_$$i.txt; \
		score_line=$$(cd tools && cargo run -q -r --bin vis ../testcases/in/$$i.txt ../tmp/out_$$i.txt 2>/dev/null | grep "Score ="); \
		if [ -n "$$score_line" ]; then \
			score=$$(echo $$score_line | cut -d ' ' -f 3); \
		else \
			score=0; \
		fi; \
		echo "Case $$i: $$score"; \
		total_score=$$((total_score + score)); \
	done; \
	echo "Total Score (0-9): $$total_score"

# ビルド成果物を削除
clean:
	rm -rf build/*
	rm -rf tmp/*.txt
