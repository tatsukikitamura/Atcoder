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
STUDY_NAME = "ahc_global_tuning"
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
    
    # Extract variable declarations
    # Matches: type var1 = val1, var2 = val2;
    # We look for lines starting with double or int inside the struct (or just generally in the file for simplicity as names are unique enough)
    
    # Remove comments to avoid parsing issues
    content = re.sub(r'//.*', '', content)
    
    # Find all declarations: type name = val, name2 = val2;
    # We'll just look for assignments of the form "name = value" preceded by type or comma
    
    # Strategy: Find all "double ...;" or "int ...;" lines
    declarations = re.findall(r'(?:double|int)\s+([^;]+);', content)
    
    for decl_line in declarations:
        # Split by comma to handle multiple declarations
        parts = decl_line.split(',')
        for part in parts:
            # Match "name = value" with support for scientific notation (e.g., 1.2e-3)
            m = re.search(r'(\w+)\s*=\s*([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)', part)
            if m:
                name = m.group(1)
                val = float(m.group(2))
                defaults[name] = val
                
    # Cast known ints
    int_keys = ["rollout_depth", "rollout_depth_max", "rollout_depth_min", "num_particles"]
    for k in int_keys:
        if k in defaults:
            defaults[k] = int(defaults[k])
        
    print(f"[INFO] main.cpp から {len(defaults)} 個のデフォルト値を読み込みました")
    return defaults


# ===================== PARAMETER SPACE DEFINITION =====================
def define_params(trial: optuna.Trial, defaults: dict) -> dict:
    """Optunaのトライアルからハイパーパラメータを生成する (全15個)"""
    params = {}

    def suggest_float_near(name, diff=0.4, min_val=0.0, max_val=2.0):
        val = defaults.get(name, 0.5)
        low = max(min_val, val - diff)
        high = min(max_val, val + diff)
        if high <= low: high = low + 1e-6
        return trial.suggest_float(name, low, high)

    # === greedyMove0 weights (8) ===
    params["wa_early"] = suggest_float_near("wa_early", 0.3)
    params["wb_early"] = suggest_float_near("wb_early", 0.3)
    params["wc_early"] = suggest_float_near("wc_early", 0.3)
    params["wd_early"] = suggest_float_near("wd_early", 0.3)
    params["wa_late"] = suggest_float_near("wa_late", 0.3)
    params["wb_late"] = suggest_float_near("wb_late", 0.3)
    params["wc_late"] = suggest_float_near("wc_late", 0.3)
    params["wd_late"] = suggest_float_near("wd_late", 0.3)

    # === MCTS (3) ===
    params["leader_mult"] = suggest_float_near("leader_mult", 0.5, min_val=0.5, max_val=3.0)
    params["ucb_c"] = suggest_float_near("ucb_c", 0.3, min_val=0.1, max_val=2.0)
    params["rollout_depth_max"] = trial.suggest_int("rollout_depth_max", 1, 15)
    params["rollout_depth_min"] = trial.suggest_int("rollout_depth_min", 1, params["rollout_depth_max"])

    # === eval (1) ===
    params["eval_trap"] = suggest_float_near("eval_trap", 0.04, min_val=0.0, max_val=0.2)

    # === U/M adaptation (2) ===
    params["u_wb_boost"] = suggest_float_near("u_wb_boost", 0.4, min_val=0.0, max_val=2.0)
    params["u_wd_penalty"] = suggest_float_near("u_wd_penalty", 0.3, min_val=0.0, max_val=1.0)

    return params


# ===================== PARAMETER FILE I/O =====================
def write_param_file(params: dict, filepath: str):
    """パラメータをファイルに書き出す"""
    with open(filepath, "w") as f:
        for k, v in params.items():
            f.write(f"{k} {v}\n")


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
CURRENT_TESTCASES = [] 

def run_all_cases(param_file: str) -> list:
    """全テストケースを実行してスコアリストを返す (並列対応)"""
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
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Optunaのトライアル数")
    parser.add_argument("--jobs", type=int, default=1,
                        help="並列実行数 (テストケース内の並列)")
    parser.add_argument("--study_name", type=str, default=STUDY_NAME,
                        help="Optuna study名")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help="Optuna DBパス")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="使用するテストケース数を制限 (0=全件)")
    # Deprecated / ignored
    parser.add_argument("--target_u", type=int, default=0, help="Deprecated")
    parser.add_argument("--strategy", type=str, default="global", help="Deprecated")

    args = parser.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Solution:     {SOLUTION_BIN}")
    
    # Set global config
    global N_JOBS, MAX_CASES, CACHED_DEFAULTS, CURRENT_TESTCASES
    N_JOBS = args.jobs
    MAX_CASES = args.max_cases
    
    # Load defaults
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

        print(f"\n{'='*40}")
        print(f" STARTING GLOBAL TUNING ")
        print(f"{'='*40}")

        # Optuna study作成
        sampler = TPESampler(seed=42, n_startup_trials=30)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
        
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.db,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

        # Enqueue current parameters as the first trial
        # We try to enqueue ALMOST ALL params that are keys in CACHED_DEFAULTS
        # But we must match the keys used in define_params
        
        # NOTE: define_params uses specific keys. We need to match defaults to those keys.
        # Most match exactly.
        
        initial_params = {}
        # List of all keys we expect to tune (based on define_params)
        expected_keys = [
            "wa_early", "wb_early", "wc_early", "wd_early",
            "wa_late", "wb_late", "wc_late", "wd_late",
            "leader_mult", "ucb_c",
            "rollout_depth_max", "rollout_depth_min",
            "eval_trap", "u_wb_boost", "u_wd_penalty"
        ]
        
        for k in expected_keys:
            if k in CACHED_DEFAULTS:
                initial_params[k] = CACHED_DEFAULTS[k]
        
        if len(study.trials) == 0 and initial_params:
            print(f"[INFO] 現在のデフォルト値を初期トライアルとしてキューに追加します ({len(initial_params)} params)")
            study.enqueue_trial(initial_params)

        print(f"[INFO] 最適化開始...")
        start_time = time.time()
        study.optimize(objective_global, n_trials=args.n_trials, show_progress_bar=True)
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
            print(f"\n=== Best for {args.study_name} ===")
            print(f"[BEST] スコア: {study.best_value:.2f}")
            print(f"[BEST] パラメータ:")
            for k, v in sorted(study.best_params.items()):
                print(f"  {k}: {v}")
            
        except Exception as e:
            # print(f"[WARN] Study読み込み失敗: {e}")
            pass

    if args.export:
        try:
            study = optuna.load_study(study_name=args.study_name, storage=args.db)
            export_path = PROJECT_ROOT / args.export
            write_param_file(study.best_params, str(export_path))
            print(f"[OK] 最良パラメータを {export_path} に出力しました")

            # C++ Hardcoded snippet export
            cpp_file = str(export_path).replace(".cfg", "_hardcoded.txt")
            with open(cpp_file, "w") as f:
                f.write("// === Optuna Global Tuned Params ===\n")
                f.write(f"// Score: {study.best_value:.2f}\n\n")
                f.write("// Paste into HyperParams declaration or init function\n")
                
                # We want to output initialization lines for the struct defaults
                # e.g. double phase1 = 0....;
                
                params = study.best_params
                
                # Output format:
                # double name = val;
                
                # Group by type/category for readability
                f.write("// [Phases]\n")
                for k in ["phase1", "phase2"]:
                    if k in params: f.write(f"double {k} = {params[k]:.8f};\n")

                f.write("\n// [Weights]\n")
                for k in sorted([p for p in params if "early" in p or "mid" in p or "late" in p]):
                    f.write(f"double {k} = {params[k]:.8f};\n")

                f.write("\n// [MCTS]\n")
                for k in ["leader_mult", "ucb_c"]:
                    if k in params: f.write(f"double {k} = {params[k]:.8f};\n")

                f.write("\n// [Eval]\n")
                for k in sorted([p for p in params if p.startswith("eval_")]):
                    f.write(f"double {k} = {params[k]:.8f};\n")

                f.write("\n// [Particles/Rollout]\n")
                if "rollout_depth_max" in params: f.write(f"int rollout_depth_max = {int(params['rollout_depth_max'])};\n")
                if "rollout_depth_min" in params: f.write(f"int rollout_depth_min = {int(params['rollout_depth_min'])};\n")
                if "num_particles" in params: f.write(f"int num_particles = {int(params['num_particles'])};\n")
                if "pf_noise_w" in params: f.write(f"double pf_noise_w = {params['pf_noise_w']:.8f};\n")
                if "pf_noise_eps" in params: f.write(f"double pf_noise_eps = {params['pf_noise_eps']:.8f};\n")
                
                f.write("\n// [Boosts]\n")
                for k in ["u_wb_boost", "u_wd_penalty", "m_leader_scale"]:
                    if k in params: f.write(f"double {k} = {params[k]:.8f};\n")

            print(f"[OK] C++埋め込み用コードを {cpp_file} に出力しました")

        except Exception as e:
            print(f"[ERROR] Export failed: {e}")


if __name__ == "__main__":
    main()
