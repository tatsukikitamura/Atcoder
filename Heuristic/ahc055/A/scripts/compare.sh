#!/bin/bash

# スクリプトのディレクトリを基準にプロジェクトルートを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# コンパイル
echo "Compiling..."
g++ -O3 -std=c++17 -o build/v1 src/experimental/v1.cpp 2>/dev/null
g++ -O3 -std=c++17 -o build/v2 src/experimental/v2.cpp 2>/dev/null

echo "Running comparison on all test cases..."
echo ""

total_orig=0
total_opt3=0
wins_orig=0
wins_opt3=0
ties=0

for i in $(seq -f "%04g" 0 30); do
    if [ -f "testcases/in/${i}.txt" ]; then
        # Run v1.cpp - extract score from "Best: score" format
        output_orig=$(./build/v1 < "testcases/in/${i}.txt" 2>&1)
        score_orig=$(echo "$output_orig" | grep "Best:" | awk '{print $2}')
        
        # Run v2.cpp
        output_opt3=$(./build/v2 < "testcases/in/${i}.txt" 2>&1)
        score_opt3=$(echo "$output_opt3" | grep "Best:" | awk '{print $2}')
        
        if [ -n "$score_orig" ] && [ -n "$score_opt3" ]; then
            total_orig=$((total_orig + score_orig))
            total_opt3=$((total_opt3 + score_opt3))
            
            if [ "$score_orig" -lt "$score_opt3" ]; then
                winner="<- orig wins"
                wins_orig=$((wins_orig + 1))
            elif [ "$score_opt3" -lt "$score_orig" ]; then
                winner="-> opt3 wins"
                wins_opt3=$((wins_opt3 + 1))
            else
                winner="   tie"
                ties=$((ties + 1))
            fi
            
            printf "Case %s: orig=%4d, opt3=%4d  %s\n" "$i" "$score_orig" "$score_opt3" "$winner"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "Total orig:  $total_orig"
echo "Total opt3:  $total_opt3"
echo "Difference:  $((total_orig - total_opt3)) (negative = opt3 better)"
echo ""
echo "Wins - orig: $wins_orig"
echo "Wins - opt3: $wins_opt3"
echo "Ties:        $ties"
echo "=========================================="
