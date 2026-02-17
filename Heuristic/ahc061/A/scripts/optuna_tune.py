"""
AHC Multi-Player Territory Game - Optuna Hyperparameter Tuning
==============================================================

使い方 (プロジェクトルート A/ から実行):
  1. make でビルド
  2. python scripts/optuna_tune.py --tune --n_trials 100
  3. python scripts/optuna_tune.py --show_best
  4. python scripts/optuna_tune.py --export best_params.cfg
  5. python scripts/embed_params.py best_params.cfg src/main.cpp > submissions/main_tuned.cpp
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

# ===================== CONFIGURATION =====================
# All paths relative to project root (A/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SOLUTION_BIN = str(PROJECT_ROOT / "build" / "main.exe")
MAIN_CPP = str(PROJECT_ROOT / "src" / "main.cpp")
TESTER_BIN = str(PROJECT_ROOT / "tools" / "target" / "release" / "tester.exe")
TESTCASE_DIR = str(PROJECT_ROOT / "tools" / "in")
COMPILE_CMD = "g++ -std=c++23 -O2 -Wall -static -o build/main.exe src/main.cpp"
DB_PATH = f"sqlite:///{PROJECT_ROOT / 'optuna_ahc.db'}"
STUDY_NAME = "ahc_bitboard"
TIMEOUT_SEC = 10  # 1テストケースあたりのタイムアウト


# ===================== PARAMETER EXTRACTION =====================
def get_default_params_from_cpp(cpp_path: str) -> dict:
    """main.cpp から現在のパラメータ初期値を読み取る"""
    defaults = {}
    if not os.path.exists(cpp_path):
        print(f"[WARN] {cpp_path} が見つかりません。デフォルト値を取得できませんでした。")
        return defaults

    with open(cpp_path, "r", encoding="utf-8") as f:
        content = f.read()

    # struct HyperParams { ... } ブロックを探す (簡易的な検索)
    # double phase1 = 0.20818362;
    # int rollout_depth = 3;
    
    # 浮動小数点数
    float_matches = re.findall(r'double\s+(\w+)\s*=\s*([\d.]+);', content)
    for name, val in float_matches:
        defaults[name] = float(val)
        
    # 整数
    int_matches = re.findall(r'int\s+(\w+)\s*=\s*(\d+);', content)
    for name, val in int_matches:
        defaults[name] = int(val)
        
    print(f"[INFO] main.cpp から {len(defaults)} 個のデフォルト値を読み込みました")
    return defaults


# ===================== PARAMETER SPACE DEFINITION =====================
def define_params(trial: optuna.Trial, defaults: dict) -> dict:
    """Optunaのトライアルからハイパーパラメータを生成する"""
    params = {}

    # === Search Space (Focused for Optimization) ===
    # 探索空間を絞り、重要なパラメータのみを探索する。
    # Phase境界やGreedy重みは固定し、探索パラメータとして定義しない(=main.cppのデフォルト値を使用)。

    # --- TUNED PARAMETERS ---

    # 1. Strategy & Search
    # UCB定数: 探索と活用のバランス
    params["ucb_c"] = trial.suggest_float("ucb_c", 0.5, 3.0)
    
    # Rollout Depth: 深読みの手数 (Bitboard化で深くできる可能性)
    params["rollout_depth"] = trial.suggest_int("rollout_depth", 3, 20)
    
    # Beam Search
    params["beam_width"] = trial.suggest_int("beam_width", 10, 1500)
    params["beam_depth"] = trial.suggest_int("beam_depth", 5, 50)
    
    # Leader Multiplier: トッププレイヤーへの攻撃意欲
    params["leader_mult"] = trial.suggest_float("leader_mult", 0.8, 1.8)

    # 2. Evaluation Function Coefficients (Log scale)
    # 拡張、レベル上げ、連結成分などの評価重み。初期値周辺を探る。
    # Log scale探索なので、0に近い値も探索範囲に入れたい。
    low_e = 1e-6
    params["eval_expand"] = trial.suggest_float("eval_expand", low_e, 1e-3, log=True)
    params["eval_level"]  = trial.suggest_float("eval_level",  low_e, 1e-3, log=True)
    params["eval_reach"]  = trial.suggest_float("eval_reach",  1e-5,  1e-2, log=True)

    # 3. Particle Filter
    # 粒子数とノイズ
    params["num_particles"] = trial.suggest_int("num_particles", 100, 500)
    # ノイズは小さすぎると収束しすぎる、大きすぎると発散する
    params["pf_noise_w"]   = trial.suggest_float("pf_noise_w",   0.001, 0.1, log=True)
    params["pf_noise_eps"] = trial.suggest_float("pf_noise_eps", 0.001, 0.1, log=True)

    # 4. Input Adaptive Parameters
    # UやMに応じた補正係数
    params["u_wb_boost"]     = trial.suggest_float("u_wb_boost", 0.0, 1.0)
    params["u_wd_penalty"]   = trial.suggest_float("u_wd_penalty", 0.0, 1.0)
    params["m_leader_scale"] = trial.suggest_float("m_leader_scale", 0.0, 0.3)

    # --- FIXED PARAMETERS (Implicitly used by omission) ---
    # 以下のパラメータは define_params に含めないことで、
    # configファイルに出力されず、main.cpp のデフォルト値が使われる。
    # phase1, phase2
    # wa_early, wb_early, ... wd_late (計12個)

    return params


# ===================== PARAMETER FILE I/O =====================
def write_param_file(params: dict, filepath: str):
    """パラメータをconfigファイルに書き出す"""
    with open(filepath, "w") as f:
        for key, val in params.items():
            f.write(f"{key} {val}\n")


def read_param_file(filepath: str) -> dict:
    """configファイルからパラメータを読み込む"""
    params = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                params[parts[0]] = float(parts[1])
    return params


# ===================== TEST CASE MANAGEMENT =====================
def get_testcases() -> list:
    """テストケースのパスリストを返す"""
    tc_dir = Path(TESTCASE_DIR)
    if not tc_dir.exists():
        print(f"[ERROR] テストケースディレクトリ '{TESTCASE_DIR}' が見つかりません")
        print("  make gen でテストケースを生成してください")
        sys.exit(1)

    cases = sorted(tc_dir.glob("*.txt"))
    if not cases:
        print(f"[ERROR] '{TESTCASE_DIR}' にテストケースが見つかりません")
        sys.exit(1)

    return [str(c) for c in cases]


# ===================== SINGLE RUN =====================
def run_single(testcase_path: str, param_file: str) -> float:
    """
    1つのテストケースに対してソリューションを実行し、絶対スコアを返す。
    AHCテスタ形式: tester.exe solution_binary [args] < input
    """
    try:
        # AHCテスタ: tester solution param_file < testcase
        with open(testcase_path, "r") as tc_input:
            result = subprocess.run(
                [TESTER_BIN, SOLUTION_BIN, param_file],
                stdin=tc_input,
                capture_output=True, text=True,
                timeout=TIMEOUT_SEC,
                cwd=str(PROJECT_ROOT)
            )

        stderr = result.stderr
        stdout = result.stdout

        # スコアのパース
        score = None
        for line in (stderr + "\n" + stdout).split("\n"):
            m = re.search(r'[Ss]core\s*[=:]\s*([\d.]+)', line)
            if m:
                score = float(m.group(1))
                break

        if score is None:
            for line in reversed(stderr.strip().split("\n")):
                line = line.strip()
                if line and re.match(r'^[\d.]+$', line):
                    score = float(line)
                    break

        if score is None:
            # print(f"[WARN] スコアをパースできません: {testcase_path}")
            return 0.0

        return score

    except subprocess.TimeoutExpired:
        # print(f"[WARN] タイムアウト: {testcase_path}")
        return 0.0
    except Exception as e:
        print(f"[WARN] 実行エラー: {testcase_path}: {e}")
        return 0.0


# Global config set by main()
N_JOBS = 1
MAX_CASES = 0
CACHED_DEFAULTS = {}


def run_all_cases(param_file: str) -> list:
    """全テストケースを実行してスコアリストを返す (並列対応)"""
    testcases = get_testcases()
    if MAX_CASES > 0:
        testcases = testcases[:MAX_CASES]

    if N_JOBS <= 1:
        return [run_single(tc, param_file) for tc in testcases]
    else:
        scores = [0.0] * len(testcases)
        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            futures = {executor.submit(run_single, tc, param_file): i
                       for i, tc in enumerate(testcases)}
            for future in as_completed(futures):
                idx = futures[future]
                scores[idx] = future.result()
        return scores


def categorize_testcases() -> dict:
    """テストケースを (M, U) カテゴリに分類"""
    testcases = get_testcases()
    if MAX_CASES > 0:
        testcases = testcases[:MAX_CASES]

    categories = {}  # (m_cat, u_cat) -> [index]
    for i, tc in enumerate(testcases):
        with open(tc) as f:
            first_line = f.readline().strip().split()
            _n, m, _t, u = int(first_line[0]), int(first_line[1]), int(first_line[2]), int(first_line[3])
        m_cat = "low" if m <= 3 else ("mid" if m <= 5 else "high")
        u_cat = "low" if u <= 2 else ("mid" if u <= 3 else "high")
        key = (m_cat, u_cat)
        if key not in categories:
            categories[key] = []
        categories[key].append(i)
    return categories


# ===================== OBJECTIVE FUNCTION =====================
def objective_global(trial: optuna.Trial) -> float:
    """全テストケースの平均スコアを最大化"""
    params = define_params(trial, CACHED_DEFAULTS)

    param_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cfg", delete=False, prefix="optuna_params_"
    )
    write_param_file(params, param_file.name)
    param_file.close()

    try:
        scores = run_all_cases(param_file.name)
        return sum(scores) / len(scores) if scores else 0.0
    finally:
        try:
            os.unlink(param_file.name)
        except:
            pass


def objective_stratified(trial: optuna.Trial) -> float:
    """M/Uカテゴリ均等で平均スコアを最大化"""
    params = define_params(trial, CACHED_DEFAULTS)

    param_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cfg", delete=False, prefix="optuna_params_"
    )
    write_param_file(params, param_file.name)
    param_file.close()

    try:
        scores = run_all_cases(param_file.name)
        cat_map = categorize_testcases()

        cat_avgs = []
        for key, indices in cat_map.items():
            cat_scores = [scores[i] for i in indices]
            if cat_scores:
                cat_avgs.append(sum(cat_scores) / len(cat_scores))

        return sum(cat_avgs) / len(cat_avgs) if cat_avgs else 0.0
    finally:
        try:
            os.unlink(param_file.name)
        except:
            pass


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser(
        description="AHC Solution - Optuna Hyperparameter Tuning",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tune", action="store_true",
                        help="Optunaでチューニングを実行")
    parser.add_argument("--show_best", action="store_true",
                        help="最良パラメータを表示")
    parser.add_argument("--export", type=str, default=None,
                        help="最良パラメータをcfgファイルに出力")
    parser.add_argument("--compile", action="store_true",
                        help="ソリューションをコンパイル")
    parser.add_argument("--eval", type=str, default=None,
                        help="指定cfgファイルで全テストケースを評価")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Optunaのトライアル数")
    parser.add_argument("--jobs", type=int, default=1,
                        help="並列実行数 (テストケース内の並列)")
    parser.add_argument("--strategy", type=str, default="global",
                        choices=["global", "stratified"],
                        help="最適化戦略: global=全体平均, stratified=M/Uカテゴリ均等")
    parser.add_argument("--study_name", type=str, default=STUDY_NAME,
                        help="Optuna study名")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help="Optuna DBパス")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="使用するテストケース数を制限 (0=全件)")
    args = parser.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Solution:     {SOLUTION_BIN}")
    print(f"[INFO] Tester:       {TESTER_BIN}")

    # Set global config
    global N_JOBS, MAX_CASES, CACHED_DEFAULTS
    N_JOBS = args.jobs
    MAX_CASES = args.max_cases
    
    # Load defaults always
    CACHED_DEFAULTS = get_default_params_from_cpp(MAIN_CPP)

    if args.compile:
        print(f"[INFO] コンパイル中: {COMPILE_CMD}")
        result = subprocess.run(COMPILE_CMD, shell=True, capture_output=True, text=True,
                                cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"[ERROR] コンパイル失敗:\n{result.stderr}")
            sys.exit(1)
        print("[OK] コンパイル成功")

    if args.tune:
        if not os.path.exists(SOLUTION_BIN):
            print(f"[ERROR] {SOLUTION_BIN} が見つかりません。make または --compile で先にコンパイルしてください")
            sys.exit(1)
        if not os.path.exists(TESTER_BIN):
            print(f"[ERROR] {TESTER_BIN} が見つかりません。cd tools && cargo build -r でビルドしてください")
            sys.exit(1)

        testcases = get_testcases()
        if MAX_CASES > 0:
            testcases = testcases[:MAX_CASES]

        print(f"[INFO] テストケース数: {len(testcases)}")
        print(f"[INFO] トライアル数: {args.n_trials}")
        print(f"[INFO] 最適化戦略: {args.strategy}")
        print(f"[INFO] 並列数: {N_JOBS}")

        # Optuna study作成
        sampler = TPESampler(seed=42, n_startup_trials=20)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
        
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.db,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

        # Enqueue current parameters as the first trial!
        tuned_param_names = [
            "ucb_c", "rollout_depth", "leader_mult",
            "beam_width", "beam_depth",
            "eval_expand", "eval_level", "eval_reach",
            "num_particles", "pf_noise_w", "pf_noise_eps",
            "u_wb_boost", "u_wd_penalty", "m_leader_scale"
        ]
        
        initial_params = {}
        for name in tuned_param_names:
            if name in CACHED_DEFAULTS:
                initial_params[name] = CACHED_DEFAULTS[name]
        
        if initial_params:
            print("[INFO] 現在のデフォルト値を初期トライアルとしてキューに追加します:")
            for k, v in initial_params.items():
                print(f"  {k}: {v}")
            study.enqueue_trial(initial_params)
        else:
            print("[WARN] デフォルト値の抽出に失敗したか、対象パラメータが見つかりませんでした。初期値をキューに追加しません。")

        # 目的関数の選択
        obj_func = objective_stratified if args.strategy == "stratified" else objective_global

        print(f"[INFO] 最適化開始...")
        start_time = time.time()
        study.optimize(obj_func, n_trials=args.n_trials, show_progress_bar=True)
        elapsed = time.time() - start_time
        
        print(f"\n[OK] 最適化完了 ({elapsed:.1f}秒)")
        print(f"[BEST] スコア: {study.best_value:.2f}")
        print(f"[BEST] パラメータ:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

    if args.show_best:
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=args.db
            )
            print(f"[BEST] スコア: {study.best_value:.2f}")
            print(f"[BEST] パラメータ:")
            for k, v in sorted(study.best_params.items()):
                print(f"  {k}: {v}")
            print(f"\n[INFO] 完了トライアル数: {len(study.trials)}")
            
            trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
            print("\n[TOP5]")
            for i, t in enumerate(trials[:5]):
                print(f"  #{i+1}: score={t.value:.2f} (trial {t.number})")

        except Exception as e:
            print(f"[ERROR] Study読み込み失敗: {e}")

    if args.export:
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=args.db
            )
            export_path = PROJECT_ROOT / args.export
            write_param_file(study.best_params, str(export_path))
            print(f"[OK] 最良パラメータを {export_path} に出力しました")

            # 提出用: パラメータをC++のデフォルト値として埋め込むコードも生成
            cpp_file = str(export_path).replace(".cfg", "_hardcoded.txt")
            with open(cpp_file, "w") as f:
                f.write("// === Optuna最適パラメータ (C++埋め込み用) ===\n")
                f.write("// solution.cppのHyperParams構造体のデフォルト値を置き換えてください\n\n")
                for k, v in sorted(study.best_params.items()):
                    if k in ("rollout_depth", "num_particles"):
                        f.write(f"    // HP.{k} = {int(v)};\n")
                    else:
                        f.write(f"    // HP.{k} = {v:.6f};\n")
            print(f"[OK] C++埋め込み用コードを {cpp_file} に出力しました")

        except Exception as e:
            print(f"[ERROR] {e}")

    if args.eval:
        eval_path = str(PROJECT_ROOT / args.eval) if not os.path.isabs(args.eval) else args.eval
        if not os.path.exists(eval_path):
            print(f"[ERROR] パラメータファイル '{eval_path}' が見つかりません")
            sys.exit(1)

        testcases = get_testcases()
        if MAX_CASES > 0:
            testcases = testcases[:MAX_CASES]
        print(f"[INFO] {eval_path} で {len(testcases)} ケースを評価中... (並列数: {N_JOBS})")
        scores = run_all_cases(eval_path)
        for i, (tc, score) in enumerate(zip(testcases, scores)):
            print(f"  [{i+1}/{len(testcases)}] {Path(tc).name}: {score:.2f}")
        
        avg = sum(scores) / len(scores)
        print(f"\n[RESULT] 平均スコア: {avg:.2f}")
        print(f"[RESULT] 最小: {min(scores):.2f}, 最大: {max(scores):.2f}")


if __name__ == "__main__":
    main()
