#!/usr/bin/env python3
"""
指定したプログラムが各入力で2000ms以内に完走できるかを判定するユーティリティ
デフォルトは同ディレクトリの in4/*.txt を対象、ターゲットは 1_2.cpp をビルド・実行
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ルート・ディレクトリ
BASE_DIR = Path(__file__).parent


def get_sdk_path() -> Optional[str]:
    """macOS SDKのパスを取得（Xcode.app優先、次にCommand Line Tools）"""
    xcode_sdk = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
    if os.path.exists(xcode_sdk):
        return xcode_sdk
    sdk_paths = glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk")
    if sdk_paths:
        return sorted(sdk_paths)[-1]
    fallback = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
    if os.path.exists(fallback):
        return fallback
    return None


def compile_cpp(cpp_path: Path, exec_path: Path) -> bool:
    """C++をビルドして実行ファイルを作成"""
    print(f"コンパイル中: {cpp_path.name} -> {exec_path.name}")
    cmd = ["g++-15", "-std=c++17", "-O2", "-Wall", "-Wno-narrowing", "-o", str(exec_path), str(cpp_path)]
    sdk_path = get_sdk_path()
    if sdk_path:
        cmd.insert(1, sdk_path)
        cmd.insert(1, "-isysroot")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("コンパイル失敗")
        sys.stderr.write(result.stderr)
        return False
    print("✓ コンパイル成功")
    return True


def read_text(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


def run_once(exec_path: Path, input_path: Path, timeout_sec: float) -> Tuple[Optional[int], float, str]:
    """
    対象バイナリを1ケース実行
    戻り値: (return_code or None on timeout/error, elapsed_ms, error_message)
    """
    input_data = read_text(input_path)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(exec_path)],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if proc.returncode != 0:
            return proc.returncode, elapsed_ms, f"終了コード{proc.returncode}"
        return 0, elapsed_ms, ""
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return None, elapsed_ms, "タイムアウト"
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return None, elapsed_ms, f"実行時例外: {e}"


def list_inputs(in_dir: Path, pattern: str) -> List[Path]:
    return sorted(in_dir.glob(pattern))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="各入力に対して2000ms内完走を判定")
    p.add_argument("--cpp", type=str, default="1_2.cpp", help="ビルド対象のC++ファイル名（同ディレクトリ相対）")
    p.add_argument("--exec", type=str, default=None, help="実行ファイルのパス（指定時は--no-compile推奨）")
    p.add_argument("--no-compile", action="store_true", help="コンパイルをスキップ")
    p.add_argument("--in-dir", type=str, default="in4", help="入力ディレクトリ（同ディレクトリ相対）")
    p.add_argument("--pattern", type=str, default="*.txt", help="入力ファイルのglobパターン")
    p.add_argument("--limit-ms", type=int, default=2000, help="判定閾値(ms)")
    p.add_argument("--timeout-sec", type=float, default=3.0, help="サブプロセスのタイムアウト(秒)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 実行対象の解決
    if args.exec:
        exec_path = Path(args.exec)
    else:
        exec_path = BASE_DIR / Path(args.cpp).with_suffix("").name

    # コンパイル
    if not args.no_compile:
        cpp_path = BASE_DIR / args.cpp
        if not cpp_path.exists():
            print(f"エラー: C++ファイルが見つかりません: {cpp_path}")
            sys.exit(2)
        if not compile_cpp(cpp_path, exec_path):
            sys.exit(2)
    else:
        if not exec_path.exists():
            print(f"エラー: 実行ファイルが存在しません（--no-compile指定）: {exec_path}")
            sys.exit(2)

    # 入力一覧
    in_dir = BASE_DIR / args.in_dir
    tests = list_inputs(in_dir, args.pattern)
    if not tests:
        print(f"エラー: 入力が見つかりません: {in_dir}/{args.pattern}")
        sys.exit(2)

    print("=" * 60)
    print(f"ターゲット: {exec_path.name} | 入力: {in_dir}/{args.pattern} | 閾値: {args.limit_ms}ms | タイムアウト: {args.timeout_sec}s")
    print("=" * 60)
    print(f"テストケース数: {len(tests)}")
    print()

    num_ok = 0
    failures = []  # (path, elapsed_ms, reason)

    for i, case in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {case.name} ... ", end="", flush=True)
        rc, elapsed_ms, err = run_once(exec_path, case, args.timeout_sec)
        status = ""
        if rc is None:
            failures.append((case, elapsed_ms, err or "不明"))
            status = f"NG ({err}, {elapsed_ms:.1f}ms)"
        elif rc != 0:
            failures.append((case, elapsed_ms, err or f"終了コード{rc}"))
            status = f"NG ({err or f'終了コード{rc}'}, {elapsed_ms:.1f}ms)"
        elif elapsed_ms > args.limit_ms:
            failures.append((case, elapsed_ms, "時間超過"))
            status = f"NG (時間超過 {elapsed_ms:.1f}ms)"
        else:
            num_ok += 1
            status = f"OK {elapsed_ms:.1f}ms"
        print(status)

    print()
    print("=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"総テストケース数: {len(tests)}")
    print(f"OK: {num_ok}")
    print(f"NG: {len(failures)}")

    if failures:
        print()
        print("NG一覧（遅い順）")
        failures.sort(key=lambda x: x[1], reverse=True)
        for case, ms, reason in failures:
            print(f"{case.name:16} | {ms:8.1f}ms | {reason}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()


