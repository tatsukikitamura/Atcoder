#!/usr/bin/env python3
"""
AHC061 Parameter Tuning with Optuna

環境変数経由で C++ ソルバーのパラメータを動的に変更し、
Optuna の TPE サンプラーで最適化を行う。

Usage:
    pip install optuna
    python scripts/optuna_tune.py                     # デフォルト: 50ケース, 100トライアル
    python scripts/optuna_tune.py -n 100 -t 200       # 100ケース, 200トライアル
    python scripts/optuna_tune.py -j 8                # 8並列ワーカー
    python scripts/optuna_tune.py --no-build          # ビルドをスキップ
    python scripts/optuna_tune.py --db optuna_v7.db   # 新しいDBで回す

Params: 15 (TOP_RAMP, APPROACH_RADIUS は固定、評価6要素の主要重みをチューニング)
"""

import optuna
import subprocess
import os
import re
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Configuration =====
BASE_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = BASE_DIR / "tools"
INPUT_DIR = TOOLS_DIR / "in"

# OS-aware solver path
if sys.platform == "win32":
    SOLVER_PATH = BASE_DIR / "build" / "main.exe"
else:
    SOLVER_PATH = BASE_DIR / "build" / "main"

# Parameter definitions: (optuna_name, env_name, default, low, high)
PARAM_DEFS = [
    # === ④ Frontier level differential ===
    ("W_FLD",        "P_W_FLD",        1.340033,  0.3,   2.5),

    # === ② Top player VL penalty ===
    ("W_TOP",        "P_W_TOP",        1.489227,  0.1,   2.5),
    ("TOP_PHASE",    "P_TOP_PHASE",    0.201841,  0.05,  0.7),
    ("TOP_RAMP",     "P_TOP_RAMP",     0.3,       0.1,   0.5),

    # === ③ Dominance bonus ===
    ("DOMINANCE_W",     "P_DOMINANCE_W",     2.724709,  0.0,   10.0),

    # === Score ratio bonus ===
    ("RATIO_SCALE",  "P_RATIO_SCALE",  0.842358,  0.1,   3.0),
    ("RATIO_PHASE",  "P_RATIO_PHASE",  0.548726,  0.05,  0.8),

    # === Quick eval ===
    ("QE_CAPTURE",   "P_QE_CAPTURE",   4.812494,  0.5,   10.0),
    ("QE_ATK_BONUS", "P_QE_ATK_BONUS", 3.970362,  0.5,   10.0),
    ("QE_EMPTY_FUT", "P_QE_EMPTY_FUT", 0.288185,  0.05,  1.0),
    ("SAFE_MULT",    "P_SAFE_MULT",    0.364951,  0.0,   1.0),

    # === ⑥ Collision avoidance ===
    ("COL_NEAR",     "P_COL_NEAR",     265.893313,     30.0,  500.0),
    ("COL_DIST2",    "P_COL_DIST2",    72.552670,      10.0,  200.0),
    ("COL_TARGET",   "P_COL_TARGET",   163.330775,     20.0,  400.0),
]


def build_solver():
    """ソルバーをビルドする"""
    print("Building solver...")
    result = subprocess.run(
        ["make"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        shell=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    if not SOLVER_PATH.exists():
        print(f"Error: Solver not found at {SOLVER_PATH}")
        sys.exit(1)
    print(f"Build complete: {SOLVER_PATH}")


def run_single_case(args):
    """テスターを通じて1テストケースを実行し、スコアを返す"""
    input_file, env = args
    try:
        with open(input_file, 'r') as f_in:
            result = subprocess.run(
                ["cargo", "run", "-q", "-r", "--bin", "tester", str(SOLVER_PATH)],
                stdin=f_in,
                capture_output=True,
                text=True,
                cwd=str(TOOLS_DIR),
                timeout=30,
                env=env,
            )
        match = re.search(r"Score = (\d+)", result.stderr)
        if match:
            return int(match.group(1))
        return 0
    except subprocess.TimeoutExpired:
        return 0
    except Exception:
        return 0


def evaluate_params(params_env, input_files, n_jobs, max_iters=-1):
    """パラメータセットを全テストケースで評価する"""
    env = os.environ.copy()
    for k, v in params_env.items():
        env[k] = f"{v:.6f}"
    
    if max_iters > 0:
        env["P_MAX_ITERS"] = str(max_iters)

    tasks = [(f, env) for f in input_files]
    scores = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_single_case, t) for t in tasks]
        for future in as_completed(futures):
            scores.append(future.result())

    return sum(scores), scores


def create_objective(input_files, n_jobs, max_iters):
    """Optuna の objective 関数を作成する"""

    def objective(trial):
        # パラメータをサジェスト
        params_env = {}
        for optuna_name, env_name, default, low, high in PARAM_DEFS:
            val = trial.suggest_float(optuna_name, low, high)
            params_env[env_name] = val

        # テストケースで評価
        total, scores = evaluate_params(params_env, input_files, n_jobs, max_iters)
        valid = [s for s in scores if s > 0]
        avg = total / len(scores) if scores else 0
        min_s = min(valid) if valid else 0
        max_s = max(valid) if valid else 0

        print(f"  Trial {trial.number:>3d}: Total={total:>8d}  "
              f"Avg={avg:>6.0f}  Min={min_s:>5d}  Max={max_s:>5d}")

        return total

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="AHC061 Parameter Tuning with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optuna_tune.py                  # 50 cases, 100 trials
  python scripts/optuna_tune.py -n 100 -t 200    # 100 cases, 200 trials
  python scripts/optuna_tune.py -j 8             # 8 parallel workers
        """,
    )
    parser.add_argument("-n", "--num-cases", type=int, default=50,
                        help="テストケース数 (default: 50)")
    parser.add_argument("-t", "--trials", type=int, default=100,
                        help="Optunaトライアル数 (default: 100)")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="並列ワーカー数 (default: 4)")
    parser.add_argument("-i", "--iters", type=int, default=2000,
                        help="1ターンあたりの最大試行回数 (default: 2000, -1で時間指定)")
    parser.add_argument("--no-build", action="store_true",
                        help="ソルバーのビルドをスキップ")
    parser.add_argument("--db", type=str, default=None,
                         help="SQLiteデータベースパス (default: optuna_ahc061_v10.db)")
    args = parser.parse_args()

    # ソルバーをビルド
    if not args.no_build:
        build_solver()
    elif not SOLVER_PATH.exists():
        print(f"Error: Solver not found at {SOLVER_PATH}. Run 'make' first.")
        sys.exit(1)

    # テストケースを取得
    input_files = sorted(INPUT_DIR.glob("*.txt"))[:args.num_cases]
    if not input_files:
        print(f"Error: No test cases found in {INPUT_DIR}")
        print("Run 'make gen' to generate test cases first.")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"  AHC061 Optuna Parameter Tuning")
    print(f"{'=' * 60}")
    print(f"  Test cases : {len(input_files)}")
    print(f"  Trials     : {args.trials}")
    print(f"  Workers    : {args.jobs}")
    print(f"  Parameters : {len(PARAM_DEFS)}")
    print(f"  Solver     : {SOLVER_PATH}")
    print(f"{'=' * 60}")

    # ベースライン実行（デフォルトパラメータ）
    print(f"\n[Baseline] Running with default parameters (iters={args.iters})...")
    baseline_total, baseline_scores = evaluate_params({}, input_files, args.jobs, args.iters)
    baseline_avg = baseline_total / len(baseline_scores) if baseline_scores else 0
    print(f"[Baseline] Total={baseline_total}  Avg={baseline_avg:.0f}")
    print(f"{'=' * 60}")

    # Optuna study の作成（新しいDBで試す場合は --db で別名を指定）
    # v10: new parameters (DIST_ENEMY) added
    db_path = args.db or str(BASE_DIR / "optuna_ahc061_v10.db")
    storage = f"sqlite:///{db_path}"

    # Optuna のログレベルを調整
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="ahc061_params_v10",
        storage=storage,
        load_if_exists=True,
    )

    # デフォルト値を最初のトライアルとして登録
    default_params = {name: default for name, _, default, _, _ in PARAM_DEFS}
    study.enqueue_trial(default_params)

    # 最適化実行
    print(f"\nStarting optimization ({args.trials} trials)...")
    print(f"{'─' * 60}")

    objective = create_objective(input_files, args.jobs, args.iters)
    start_time = time.time()

    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("Results so far are saved to the database.")

    elapsed = time.time() - start_time

    # ===== 結果表示 =====
    print(f"\n{'=' * 60}")
    print(f"  OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Elapsed    : {elapsed/60:.1f} minutes")
    print(f"  Trials     : {len(study.trials)}")
    print(f"  Best trial : #{study.best_trial.number}")
    print(f"  Best score : {study.best_value}")
    print(f"  Baseline   : {baseline_total}")

    if baseline_total > 0:
        improvement = (study.best_value - baseline_total) / baseline_total * 100
        print(f"  Improvement: {improvement:+.2f}%")

    print(f"\n{'─' * 60}")
    print(f"  Best Parameters:")
    print(f"{'─' * 60}")
    for optuna_name, env_name, default, _, _ in PARAM_DEFS:
        best_val = study.best_params[optuna_name]
        diff = best_val - default
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {optuna_name:<14s} = {best_val:>10.6f}  (default: {default:>8.4f})  {arrow}")

    # C++ コードスニペット出力
    print(f"\n{'─' * 60}")
    print(f"  Copy to main.cpp (replace defaults):")
    print(f"{'─' * 60}")
    for optuna_name, env_name, default, _, _ in PARAM_DEFS:
        best_val = study.best_params[optuna_name]
        print(f"double {env_name:<14s} = {best_val:.6f};")

    # Top 5 トライアル
    print(f"\n{'─' * 60}")
    print(f"  Top 5 Trials:")
    print(f"{'─' * 60}")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, trial in enumerate(trials_sorted[:5]):
        score = trial.value if trial.value else 0
        print(f"  #{trial.number:>3d}: Score={score:>8.0f}")

    print(f"\n{'=' * 60}")
    print(f"  Database saved to: {db_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
