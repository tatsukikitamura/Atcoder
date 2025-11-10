#!/usr/bin/env python3
import subprocess
import re
import os
from pathlib import Path

def extract_score(stderr_output):
    """標準エラー出力からスコアを抽出"""
    # "スコア: 123" の形式を探す
    match = re.search(r'スコア:\s*(\d+)', stderr_output)
    if match:
        return int(match.group(1))
    return None

def extract_time(stderr_output):
    """標準エラー出力から実行時間を抽出"""
    match = re.search(r'実行時間:\s*(\d+)\s*ms', stderr_output)
    if match:
        return int(match.group(1))
    return None

def run_test(executable, test_file):
    """テストケースを実行してスコアと実行時間を取得"""
    try:
        with open(test_file, 'r') as f:
            input_data = f.read()
        
        result = subprocess.run(
            [executable],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=3.0  # 3秒のタイムアウト
        )
        
        score = extract_score(result.stderr)
        time_ms = extract_time(result.stderr)
        
        return {
            'score': score,
            'time_ms': time_ms,
            'success': result.returncode == 0,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'score': None,
            'time_ms': None,
            'success': False,
            'stderr': 'Timeout'
        }
    except Exception as e:
        return {
            'score': None,
            'time_ms': None,
            'success': False,
            'stderr': str(e)
        }

def main():
    test_dir = Path('in')
    executable_1_1 = './1_1'
    executable_1_5 = './1_5'
    
    # テストケースを取得（最初の20個と最後の10個、中間の10個をサンプルとして使用）
    test_files = sorted(test_dir.glob('*.txt'))
    
    # サンプルとして最初の30個、中間の10個、最後の10個を使用（合計50個）
    sample_indices = list(range(30)) + list(range(40, 50)) + list(range(90, 100))
    sample_files = [test_files[i] for i in sample_indices if i < len(test_files)]
    
    print(f"テストケース数: {len(sample_files)}")
    print("=" * 80)
    
    results = []
    wins_1_1 = 0
    wins_1_5 = 0
    ties = 0
    total_score_1_1 = 0
    total_score_1_5 = 0
    valid_tests = 0
    
    for test_file in sample_files:
        test_name = test_file.name
        print(f"\nテストケース: {test_name}")
        
        result_1_1 = run_test(executable_1_1, test_file)
        result_1_5 = run_test(executable_1_5, test_file)
        
        score_1_1 = result_1_1['score']
        score_1_5 = result_1_5['score']
        time_1_1 = result_1_1['time_ms']
        time_1_5 = result_1_5['time_ms']
        
        if score_1_1 is not None and score_1_5 is not None:
            valid_tests += 1
            total_score_1_1 += score_1_1
            total_score_1_5 += score_1_5
            
            diff = score_1_1 - score_1_5
            if diff < 0:
                winner = "1_5 (勝ち)"
                wins_1_5 += 1
            elif diff > 0:
                winner = "1_1 (勝ち)"
                wins_1_1 += 1
            else:
                winner = "引き分け"
                ties += 1
            
            print(f"  1_1: スコア={score_1_1}, 時間={time_1_1}ms")
            print(f"  1_5: スコア={score_1_5}, 時間={time_1_5}ms")
            print(f"  差: {diff:+d} ({winner})")
            
            results.append({
                'test': test_name,
                'score_1_1': score_1_1,
                'score_1_5': score_1_5,
                'time_1_1': time_1_1,
                'time_1_5': time_1_5,
                'diff': diff
            })
        else:
            print(f"  1_1: スコア={score_1_1}, 成功={result_1_1['success']}")
            print(f"  1_5: スコア={score_1_5}, 成功={result_1_5['success']}")
            if not result_1_1['success']:
                print(f"  1_1 エラー: {result_1_1['stderr'][:100]}")
            if not result_1_5['success']:
                print(f"  1_5 エラー: {result_1_5['stderr'][:100]}")
    
    print("\n" + "=" * 80)
    print("集計結果")
    print("=" * 80)
    print(f"有効なテストケース数: {valid_tests}")
    print(f"1_1の勝利数: {wins_1_1}")
    print(f"1_5の勝利数: {wins_1_5}")
    print(f"引き分け: {ties}")
    print(f"\n平均スコア:")
    if valid_tests > 0:
        print(f"  1_1: {total_score_1_1 / valid_tests:.2f}")
        print(f"  1_5: {total_score_1_5 / valid_tests:.2f}")
        print(f"  差: {(total_score_1_1 - total_score_1_5) / valid_tests:+.2f}")
    
    # 詳細な統計
    if results:
        print(f"\n詳細統計:")
        diffs = [r['diff'] for r in results]
        print(f"  スコア差の平均: {sum(diffs) / len(diffs):+.2f}")
        print(f"  スコア差の最小: {min(diffs):+d}")
        print(f"  スコア差の最大: {max(diffs):+d}")
        
        # 1_1が勝ったケースの平均差
        wins_1_1_diffs = [r['diff'] for r in results if r['diff'] > 0]
        if wins_1_1_diffs:
            print(f"  1_1が勝ったケースの平均差: {sum(wins_1_1_diffs) / len(wins_1_1_diffs):+.2f}")
        
        # 1_5が勝ったケースの平均差
        wins_1_5_diffs = [r['diff'] for r in results if r['diff'] < 0]
        if wins_1_5_diffs:
            print(f"  1_5が勝ったケースの平均差: {sum(wins_1_5_diffs) / len(wins_1_5_diffs):+.2f}")
        
        # 実行時間の比較
        times_1_1 = [r['time_1_1'] for r in results if r['time_1_1'] is not None]
        times_1_5 = [r['time_1_5'] for r in results if r['time_1_5'] is not None]
        if times_1_1 and times_1_5:
            print(f"\n実行時間:")
            print(f"  1_1の平均: {sum(times_1_1) / len(times_1_1):.2f}ms")
            print(f"  1_5の平均: {sum(times_1_5) / len(times_1_5):.2f}ms")

if __name__ == '__main__':
    main()


