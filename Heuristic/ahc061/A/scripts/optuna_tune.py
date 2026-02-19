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
    """Optunaのトライアルからハイパーパラメータを生成する (Top 11)"""
    params = {}

    def suggest_float_near(name, diff=0.3, min_val=0.0, max_val=2.0):
        val = defaults.get(name, 0.5)
        low = max(min_val, val - diff)
        high = min(max_val, val + diff)
        return trial.suggest_float(name, low, high)
    
    def suggest_log_near(name):
        val = defaults.get(name, 1e-5)
        low = val * 0.1
        high = val * 10
        return trial.suggest_float(name, low, high, log=True)

    # === Top 11 Parameters (from Importance Analysis) ===
    
    # 1. rollout_depth (Importance: 0.10)
    rd = defaults.get("rollout_depth", 6)
    params["rollout_depth"] = trial.suggest_int("rollout_depth", max(1, rd - 2), min(20, rd + 2))

    # 2. wb_mid (Importance: 0.10)
    params["wb_mid"] = suggest_float_near("wb_mid", 0.3)

    # 3. wc_late (Importance: 0.07)
    params["wc_late"] = suggest_float_near("wc_late", 0.3)

    # 4. eval_level (Importance: 0.07)
    params["eval_level"] = suggest_log_near("eval_level")

    # 5. wd_early (Importance: 0.07)
    params["wd_early"] = suggest_float_near("wd_early", 0.3)

    # 6. eval_attack (Importance: 0.06)
    params["eval_attack"] = suggest_log_near("eval_attack")

    # 7. u_wb_boost (Importance: 0.06)
    params["u_wb_boost"] = suggest_float_near("u_wb_boost", 0.2)

    # 8. wd_mid (Importance: 0.05)
    params["wd_mid"] = suggest_float_near("wd_mid", 0.3)

    # 9. wa_late (Importance: 0.04)
    params["wa_late"] = suggest_float_near("wa_late", 0.3)

    # 10. wb_late (Importance: 0.04)
    params["wb_late"] = suggest_float_near("wb_late", 0.3)

    # 11. wc_early (Importance: 0.04)
    params["wc_early"] = suggest_float_near("wc_early", 0.3)

    # 12. eval_trap
    params["eval_trap"] = suggest_log_near("eval_trap")

    return params


# ===================== PARAMETER FILE I/O =====================
def write_param_file(params: dict, filepath: str, target_u: int = 0):
    """パラメータをファイルに書き出す. target_u > 0 の場合 u{target_u}_key 形式で書き出す"""
    with open(filepath, "w") as f:
        for k, v in params.items():
            if target_u > 0:
                f.write(f"u{target_u}_{k} {v}\n")
            else:
                f.write(f"{k} {v}\n")


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

def get_testcases_by_u(target_u: int) -> list:
    """指定されたUのテストケースのみを返す"""
    all_cases = get_testcases()
    filtered = []
    for tc in all_cases:
        try:
            with open(tc, "r") as f:
                # First line: N M T U
                line = f.readline().split()
                if len(line) >= 4 and int(line[3]) == target_u:
                    filtered.append(tc)
        except:
            pass
    return filtered


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
CURRENT_TESTCASES = [] # Currently active testcases for the objective function

def run_all_cases(param_file: str) -> list:
    """全テストケースを実行してスコアリストを返す (並列対応)"""
    # Use global CURRENT_TESTCASES if set, otherwise get all
    testcases = CURRENT_TESTCASES if CURRENT_TESTCASES else get_testcases()
    
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
    testcases = CURRENT_TESTCASES if CURRENT_TESTCASES else get_testcases()
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
# Global context for objective function
CURRENT_TARGET_U = 0

def objective_global(trial: optuna.Trial) -> float:
    """全テストケースの平均スコアを最大化"""
    params = define_params(trial, CACHED_DEFAULTS)

    param_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cfg", delete=False, prefix="optuna_params_"
    )
    # Use global CURRENT_TARGET_U to prefix parameters if needed
    write_param_file(params, param_file.name, int(CURRENT_TARGET_U))
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
    # Use global CURRENT_TARGET_U to prefix parameters if needed
    write_param_file(params, param_file.name, int(CURRENT_TARGET_U))
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
    parser.add_argument("--target_u", type=int, default=0,
                        help="特定のUのみチューニング (0=全て)")
    args = parser.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Solution:     {SOLUTION_BIN}")
    print(f"[INFO] Tester:       {TESTER_BIN}")

    # Set global config
    global N_JOBS, MAX_CASES, CACHED_DEFAULTS, CURRENT_TESTCASES, CURRENT_TARGET_U
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

        # Iterate over U values
        u_range = [args.target_u] if args.target_u > 0 else [1, 2, 3, 4, 5]

        for u_val in u_range:
            CURRENT_TARGET_U = u_val # Set global context for objective function
            print(f"\n{'='*40}")
            print(f" STARTING TUNING FOR U={u_val} ")
            print(f"{'='*40}")

            # Filter testcases for this U
            CURRENT_TESTCASES = get_testcases_by_u(u_val)
            print(f"[INFO] U={u_val} のテストケース数: {len(CURRENT_TESTCASES)}")
            
            if not CURRENT_TESTCASES:
                print(f"[WARN] U={u_val} のテストケースが見つかりません。スキップします。")
                continue

            current_study_name = f"{args.study_name}_u{u_val}"

            print(f"[INFO] トライアル数: {args.n_trials}")
            print(f"[INFO] 最適化戦略: {args.strategy}")
            print(f"[INFO] 並列数: {N_JOBS}")
            print(f"[INFO] Study名: {current_study_name}")

            # Optuna study作成
            sampler = TPESampler(seed=42, n_startup_trials=20)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
            
            study = optuna.create_study(
                study_name=current_study_name,
                storage=args.db,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )

            # Enqueue current parameters as the first trial!
            tuned_param_names = [
                "rollout_depth",
                "wb_mid", "wc_late", "eval_level", "wd_early", 
                "eval_attack", "u_wb_boost", "wd_mid", 
                "wa_late", "wb_late", "wc_early", "eval_trap"
            ]
            
            initial_params = {}
            for name in tuned_param_names:
                if name in CACHED_DEFAULTS:
                    initial_params[name] = CACHED_DEFAULTS[name]
            
            # Only enqueue if the study is new (or empty) to avoid re-evaluating defaults every time
            if len(study.trials) == 0 and initial_params:
                print("[INFO] 現在のデフォルト値を初期トライアルとしてキューに追加します:")
                study.enqueue_trial(initial_params)

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
        u_range = [args.target_u] if args.target_u > 0 else [1, 2, 3, 4, 5]
        for u_val in u_range:
            try:
                current_study_name = f"{args.study_name}_u{u_val}"
                study = optuna.load_study(
                    study_name=current_study_name,
                    storage=args.db
                )
                print(f"\n=== Best for U={u_val} (Study: {current_study_name}) ===")
                print(f"[BEST] スコア: {study.best_value:.2f}")
                print(f"[BEST] パラメータ:")
                for k, v in sorted(study.best_params.items()):
                    print(f"  {k}: {v}")
                
            except Exception as e:
                # print(f"[WARN] Study読み込み失敗 (U={u_val}): {e}")
                pass

    if args.export:
        # Export best params for all U or specific U
        # Since we have separate HP[1]...HP[5], we need to export them properly.
        # Format: u{u}_{name} {value}
        
        all_best_params = {}
        
        # Collect params from all relevant studies
        u_range = [args.target_u] if args.target_u > 0 else [1, 2, 3, 4, 5]
        
        for u in u_range:
            try:
                s = optuna.load_study(study_name=f"{args.study_name}_u{u}", storage=args.db)
                # Add prefix
                for k, v in s.best_params.items():
                    all_best_params[f"u{u}_{k}"] = v
            except:
                pass
        
        if not all_best_params:
            print("[ERROR] No studies found to export.")
            sys.exit(1)

        export_path = PROJECT_ROOT / args.export
        # Use simple write, write_param_file adds prefix but we already added it?
        # write_param_file logic: if target_u > 0 writes u{target_u}_{k}.
        # Here we have mixed U. So just write manually.
        
        with open(export_path, "w") as f:
            for k, v in sorted(all_best_params.items()):
                f.write(f"{k} {v}\n")
        
        print(f"[OK] 最良パラメータを {export_path} に出力しました")

        # C++ Hardcoded snippet export
        cpp_file = str(export_path).replace(".cfg", "_hardcoded.txt")
        with open(cpp_file, "w") as f:
            f.write("// === Optuna最適パラメータ (C++埋め込み用) ===\n")
            f.write("// Paste this into adaptParams() or initialize HP[] with these values\n")
            
            for u in range(1, 6):
                # Try to load study for this U
                try:
                    s = optuna.load_study(study_name=f"{args.study_name}_u{u}", storage=args.db)
                    params = s.best_params
                    f.write(f"\n    // --- U={u} ---\n")
                    for k, v in sorted(params.items()):
                        if k in ("rollout_depth", "num_particles"):
                            f.write(f"    HP[{u}].{k} = {int(v)};\n")
                        else:
                            f.write(f"    HP[{u}].{k} = {v:.6f};\n")
                except:
                    pass
        
        print(f"[OK] C++埋め込み用コードを {cpp_file} に出力しました")


    if args.eval:
        # NOTE: If we want to evaluate with U-specific parameters, we need to load a file that has uX_ prefix.
        # The current implementation of main.cpp loadParams handles uX_ prefix.
        # So we just pass the file path.
        pass

if __name__ == "__main__":
    main()
