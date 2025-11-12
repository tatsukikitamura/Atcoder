#!/usr/bin/env python3
"""
2つのC++プログラムのスコアを比較するプログラム
inディレクトリ内のすべてのテストケースで実行し、どちらが優れているかを判定
"""

# ============================================
# 設定: 比較するプログラムのファイル名を指定
# ============================================
PROGRAM1_NAME = "second/2.cpp"  # 比較する1つ目のプログラム
PROGRAM2_NAME = "second/1.cpp"  # 比較する2つ目のプログラム
# ============================================

import subprocess
import os
import re
import time
import glob
from pathlib import Path
from typing import Optional, Tuple

# ディレクトリ設定
BASE_DIR = Path(__file__).parent
IN_DIR = BASE_DIR / "in2"
PROGRAM1 = BASE_DIR / PROGRAM1_NAME
PROGRAM2 = BASE_DIR / PROGRAM2_NAME
EXEC1 = BASE_DIR / PROGRAM1_NAME.replace(".cpp", "")
EXEC2 = BASE_DIR / PROGRAM2_NAME.replace(".cpp", "")

def get_sdk_path() -> Optional[str]:
    """macOS SDKのパスを取得（Xcode.appを優先、次にCommand Line Tools）"""
    # Xcode.appのSDKを優先（_bounds.hなどの新しいヘッダーが含まれる）
    xcode_sdk = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
    if os.path.exists(xcode_sdk):
        return xcode_sdk
    
    # Command Line ToolsのSDKを試す
    sdk_paths = glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk")
    if sdk_paths:
        return sorted(sdk_paths)[-1]  # 最新のSDKを使用
    sdk_path = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
    if os.path.exists(sdk_path):
        return sdk_path
    return None

def compile_program(cpp_file: Path, exec_file: Path) -> bool:
    """C++プログラムをコンパイル"""
    print(f"コンパイル中: {cpp_file.name}...")
    compile_cmd = ["g++-15", "-std=c++17", "-O2", "-Wall", "-Wno-narrowing", "-o", str(exec_file), str(cpp_file)]
    
    # macOS SDKのパスを指定（GCC 15がシステムヘッダーを見つけられるように）
    sdk_path = get_sdk_path()
    if sdk_path:
        compile_cmd.insert(1, sdk_path)
        compile_cmd.insert(1, "-isysroot")
    
    result = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"エラー: {cpp_file.name}のコンパイルに失敗しました")
        print(result.stderr)
        return False
    print(f"✓ {cpp_file.name}のコンパイル成功")
    return True

def parse_input(input_file: Path) -> Tuple[Optional[int], Optional[int], Optional[int], list, list, list]:
    """入力ファイルからN, K, T, TARGETS, V_WALLS, H_WALLSを取得"""
    try:
        with open(input_file, 'r') as f:
            first_line = f.readline().strip().split()
            if len(first_line) < 3:
                return None, None, None, [], [], []
            N = int(first_line[0])
            K = int(first_line[1])
            T = int(first_line[2])
            
            # 壁の情報を読み取る
            V_WALLS = []
            for _ in range(N):
                line = f.readline().strip()
                V_WALLS.append(line)
            
            H_WALLS = []
            for _ in range(N - 1):
                line = f.readline().strip()
                H_WALLS.append(line)
            
            # ターゲットを読み取る
            targets = []
            for _ in range(K):
                line = f.readline().strip().split()
                if len(line) >= 2:
                    targets.append((int(line[0]), int(line[1])))
            
            return N, K, T, targets, V_WALLS, H_WALLS
    except Exception as e:
        return None, None, None, [], [], []

def parse_output(stdout_output: str, N: int) -> Tuple[Optional[int], Optional[int], Optional[int], list, list]:
    """標準出力からC, Q, M, 遷移規則, ボードを取得"""
    try:
        lines = stdout_output.strip().split('\n')
        if len(lines) < 1:
            return None, None, None, [], []
        
        # 最初の行: C Q M
        first_line = lines[0].strip().split()
        if len(first_line) < 3:
            return None, None, None, [], []
        C = int(first_line[0])
        Q = int(first_line[1])
        M = int(first_line[2])
        
        # ボードを読み取る（1行目からN行目）
        board = []
        for i in range(1, 1 + N):
            if i >= len(lines):
                break
            board_line = lines[i].strip().split()
            board.append([int(x) for x in board_line])
        
        # 遷移規則を読み取る（N+1行目からM行）
        rules = []
        for i in range(1 + N, 1 + N + M):
            if i >= len(lines):
                break
            rule_line = lines[i].strip().split()
            if len(rule_line) >= 5:
                c = int(rule_line[0])
                q = int(rule_line[1])
                a = int(rule_line[2])
                s = int(rule_line[3])
                d = rule_line[4]
                rules.append((c, q, a, s, d))
        
        return C, Q, M, rules, board
    except Exception as e:
        return None, None, None, [], []

def can_move(i: int, j: int, d: str, N: int, V_WALLS: list, H_WALLS: list) -> bool:
    """移動可能かどうかを判定"""
    if d == 'U':
        if i == 0:
            return False
        return H_WALLS[i - 1][j] == '0'
    elif d == 'D':
        if i == N - 1:
            return False
        return H_WALLS[i][j] == '0'
    elif d == 'L':
        if j == 0:
            return False
        return V_WALLS[i][j - 1] == '0'
    elif d == 'R':
        if j == N - 1:
            return False
        return V_WALLS[i][j] == '0'
    return False

def calculate_visited_destinations(N: int, K: int, T: int, targets: list, rules: list, board: list, V_WALLS: list, H_WALLS: list) -> int:
    """遷移規則から訪問できた目的地の数Vを計算"""
    if not rules or not board:
        return 0
    
    # 遷移規則を辞書に変換: (c, q) -> (a, s, d)
    transition = {}
    for c, q, a, s, d in rules:
        transition[(c, q)] = (a, s, d)
    
    # シミュレーション: 初期状態から開始
    # 初期状態: q=0, 現在位置はtargets[0]
    visited = set()
    current_q = 0
    current_pos = targets[0]
    visited.add(0)  # 最初の目的地は訪問済み
    
    # ステップ数の上限までシミュレート
    for step in range(T):
        # 現在位置の色を取得
        i, j = current_pos
        if i < 0 or i >= N or j < 0 or j >= N:
            break
        c = board[i][j]
        
        key = (c, current_q)
        if key not in transition:
            break
        
        a, s, d = transition[key]
        current_q = s
        
        # 移動方向に応じて位置を更新（壁をチェック）
        next_pos = current_pos
        if d == 'U':
            if can_move(i, j, 'U', N, V_WALLS, H_WALLS):
                next_pos = (i - 1, j)
            else:
                break
        elif d == 'D':
            if can_move(i, j, 'D', N, V_WALLS, H_WALLS):
                next_pos = (i + 1, j)
            else:
                break
        elif d == 'L':
            if can_move(i, j, 'L', N, V_WALLS, H_WALLS):
                next_pos = (i, j - 1)
            else:
                break
        elif d == 'R':
            if can_move(i, j, 'R', N, V_WALLS, H_WALLS):
                next_pos = (i, j + 1)
            else:
                break
        elif d == 'S':
            # 停止
            pass
        else:
            break
        
        current_pos = next_pos
        
        # 目的地に到達したかチェック
        for i, target in enumerate(targets):
            if current_pos == target and i not in visited:
                visited.add(i)
                if len(visited) == K:
                    return K
    
    return len(visited)

def calculate_absolute_score(N: int, K: int, C: int, Q: int, V: int) -> int:
    """絶対スコアを計算"""
    if V == K:
        return C + Q
    else:
        return 2 * (N ** 4) + (K - V) * (N ** 2)

def extract_score(stderr_output: str) -> Optional[int]:
    """標準エラー出力からスコアを抽出（後方互換性のため）"""
    match = re.search(r'スコア:\s*(\d+)', stderr_output)
    if match:
        return int(match.group(1))
    return None

def run_program(exec_file: Path, input_file: Path) -> Tuple[Optional[int], str, float]:
    """プログラムを実行して正しい絶対スコアを取得（実行時間も返す）"""
    try:
        # 入力ファイルを読み取る
        N, K, T, targets, V_WALLS, H_WALLS = parse_input(input_file)
        if N is None or K is None or T is None:
            return None, "入力ファイルの解析に失敗", 0.0
        
        with open(input_file, 'r') as f:
            input_data = f.read()
        
        start_time = time.time()
        result = subprocess.run(
            [str(exec_file)],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=30  # 30秒のタイムアウト
        )
        elapsed_time = (time.time() - start_time) * 1000  # ミリ秒に変換
        
        if result.returncode != 0:
            return None, f"実行エラー (終了コード: {result.returncode})", elapsed_time
        
        # 標準出力からC, Q, M, 遷移規則, ボードを取得
        C, Q, M, rules, board = parse_output(result.stdout, N)
        if C is None or Q is None:
            # フォールバック: 標準エラー出力からスコアを取得
            score = extract_score(result.stderr)
            if score is None:
                return None, "スコアが見つかりませんでした", elapsed_time
            return score, "", elapsed_time
        
        # 訪問できた目的地の数Vを計算
        V = calculate_visited_destinations(N, K, T, targets, rules, board, V_WALLS, H_WALLS)
        
        # 正しい絶対スコアを計算
        absolute_score = calculate_absolute_score(N, K, C, Q, V)
        
        return absolute_score, "", elapsed_time
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
    
    # ウォームアップ（初回起動コストを除外するため、最初のテストで両実行ファイルを一度だけ実行）
    # 計測・比較には含めない
    try:
        warmup_case = test_cases[0]
        _ = run_program(EXEC1, warmup_case)
        _ = run_program(EXEC2, warmup_case)
        print(f"[ウォームアップ] {warmup_case.name} で両プログラムを事前実行完了")
        print()
    except Exception:
        # ウォームアップ失敗時は無視して続行（本番計測に影響しないようにする）
        pass
    
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

