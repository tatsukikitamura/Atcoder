#!/usr/bin/env python3
"""
2つのC++プログラムのスコアを比較するプログラム
inディレクトリ内のすべてのテストケースで実行し、どちらが優れているかを判定
"""

# ============================================
# 設定: 比較するプログラムのファイル名を指定
# ============================================
PROGRAM1_NAME = "1_1.cpp"  # 比較する1つ目のプログラム
PROGRAM2_NAME = "1_5.cpp"  # 比較する2つ目のプログラム
# ============================================

import subprocess
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

# ディレクトリ設定
BASE_DIR = Path(__file__).parent
IN_DIR = BASE_DIR / "in4"
PROGRAM1 = BASE_DIR / PROGRAM1_NAME
PROGRAM2 = BASE_DIR / PROGRAM2_NAME
EXEC1 = BASE_DIR / PROGRAM1_NAME.replace(".cpp", "")
EXEC2 = BASE_DIR / PROGRAM2_NAME.replace(".cpp", "")

def compile_program(cpp_file: Path, exec_file: Path) -> bool:
    """C++プログラムをコンパイル"""
    print(f"コンパイル中: {cpp_file.name}...")
    result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-o", str(exec_file), str(cpp_file)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"エラー: {cpp_file.name}のコンパイルに失敗しました")
        print(result.stderr)
        return False
    print(f"✓ {cpp_file.name}のコンパイル成功")
    return True

def extract_score(stderr_output: str) -> Optional[int]:
    """標準エラー出力からスコアを抽出"""
    # 「スコア: 123」の形式を探す
    match = re.search(r'スコア:\s*(\d+)', stderr_output)
    if match:
        return int(match.group(1))
    return None

def run_program(exec_file: Path, input_file: Path) -> Tuple[Optional[int], str, float]:
    """プログラムを実行してスコアを取得（実行時間も返す）"""
    try:
        with open(input_file, 'r') as f:
            start_time = time.time()
            result = subprocess.run(
                [str(exec_file)],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=30  # 30秒のタイムアウト
            )
            elapsed_time = (time.time() - start_time) * 1000  # ミリ秒に変換
        
        if result.returncode != 0:
            return None, f"実行エラー (終了コード: {result.returncode})", elapsed_time
        
        score = extract_score(result.stderr)
        if score is None:
            return None, "スコアが見つかりませんでした", elapsed_time
        
        return score, "", elapsed_time
    except subprocess.TimeoutExpired:
        return None, "タイムアウト", 30000.0  # タイムアウト時は30秒として記録
    except Exception as e:
        return None, f"エラー: {str(e)}", 0.0

def main():
    print("=" * 60)
    print(f"{PROGRAM1_NAME} と {PROGRAM2_NAME} のスコア比較")
    print("=" * 60)
    print()
    
    # コンパイル
    if not compile_program(PROGRAM1, EXEC1):
        return
    if not compile_program(PROGRAM2, EXEC2):
        return
    print()
    
    # テストケースを取得
    test_cases = sorted(IN_DIR.glob("*.txt"))
    if not test_cases:
        print(f"エラー: {IN_DIR}にテストケースが見つかりません")
        return
    
    print(f"テストケース数: {len(test_cases)}")
    print()
    
    # 結果を格納
    results = []
    wins_1 = 0
    wins_2 = 0
    ties = 0
    errors_1 = 0
    errors_2 = 0
    
    # プログラム名（拡張子なし）を取得
    name1 = PROGRAM1_NAME.replace(".cpp", "")
    name2 = PROGRAM2_NAME.replace(".cpp", "")
    
    # 各テストケースで実行
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {test_case.name}...", end=" ", flush=True)
        
        # PROGRAM1を実行
        score1, error1, time1 = run_program(EXEC1, test_case)
        if score1 is None:
            errors_1 += 1
            print(f"{name1}: エラー ({error1})", end=" | ")
        else:
            print(f"{name1}: {score1}", end=" | ", flush=True)
        
        # PROGRAM2を実行
        score2, error2, time2 = run_program(EXEC2, test_case)
        if score2 is None:
            errors_2 += 1
            print(f"{name2}: エラー ({error2})")
        else:
            print(f"{name2}: {score2}", end=" | ", flush=True)
            
            # 比較（スコアが小さい方が良い）
            if score1 is not None:
                if score1 < score2:
                    wins_1 += 1
                    result = f"{name1}勝"
                elif score2 < score1:
                    wins_2 += 1
                    result = f"{name2}勝"
                else:
                    ties += 1
                    result = "同点"
                
                # 2000msを超えた場合の警告を追加
                warnings = []
                if time1 > 2000:
                    warnings.append(f"{name1}({test_case.name}): {time1:.1f}ms")
                if time2 > 2000:
                    warnings.append(f"{name2}({test_case.name}): {time2:.1f}ms")
                
                if warnings:
                    result += " | [警告: 2000ms超過] " + ", ".join(warnings)
                
                print(result)
            else:
                print()
        
        results.append({
            'test_case': test_case.name,
            'score1': score1,
            'score2': score2,
            'error1': error1 if score1 is None else None,
            'error2': error2 if score2 is None else None,
            'time1': time1,
            'time2': time2
        })
    
    # 結果サマリー
    print()
    print("=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"総テストケース数: {len(test_cases)}")
    print(f"{PROGRAM1_NAME} の勝利: {wins_1} ({wins_1/len(test_cases)*100:.1f}%)")
    print(f"{PROGRAM2_NAME} の勝利: {wins_2} ({wins_2/len(test_cases)*100:.1f}%)")
    print(f"同点: {ties} ({ties/len(test_cases)*100:.1f}%)")
    print(f"{PROGRAM1_NAME} のエラー: {errors_1}")
    print(f"{PROGRAM2_NAME} のエラー: {errors_2}")
    print()
    
    # 平均スコアを計算
    valid_scores_1 = [r['score1'] for r in results if r['score1'] is not None]
    valid_scores_2 = [r['score2'] for r in results if r['score2'] is not None]
    
    if valid_scores_1:
        avg1 = sum(valid_scores_1) / len(valid_scores_1)
        print(f"{PROGRAM1_NAME} の平均スコア: {avg1:.2f}")
    if valid_scores_2:
        avg2 = sum(valid_scores_2) / len(valid_scores_2)
        print(f"{PROGRAM2_NAME} の平均スコア: {avg2:.2f}")
    
    if valid_scores_1 and valid_scores_2:
        diff = avg1 - avg2
        if diff < 0:
            print(f"{PROGRAM1_NAME} が平均で {abs(diff):.2f} 点優れています")
        elif diff > 0:
            print(f"{PROGRAM2_NAME} が平均で {diff:.2f} 点優れています")
        else:
            print("平均スコアは同じです")
    
    print()
    print("=" * 60)
    print("詳細結果（スコア差が大きい順）")
    print("=" * 60)
    
    # スコア差でソート（差があるもののみ）
    detailed_results = [
        r for r in results 
        if r['score1'] is not None and r['score2'] is not None and r['score1'] != r['score2']
    ]
    detailed_results.sort(key=lambda x: abs(x['score1'] - x['score2']), reverse=True)
    
    # 差があるもの全てを表示
    for r in detailed_results:
        diff = r['score1'] - r['score2']
        winner = name1 if diff < 0 else name2
        
        # 2000msを超えた場合の警告を追加
        warnings = []
        if r['time1'] > 2000:
            warnings.append(f"{name1}({r['test_case']}): {r['time1']:.1f}ms")
        if r['time2'] > 2000:
            warnings.append(f"{name2}({r['test_case']}): {r['time2']:.1f}ms")
        
        warning_text = " | [警告: 2000ms超過] " + ", ".join(warnings) if warnings else ""
        
        print(f"{r['test_case']:15} | {name1}: {r['score1']:6} | {name2}: {r['score2']:6} | "
              f"差: {diff:+6} | {winner}{warning_text}")

if __name__ == "__main__":
    main()

