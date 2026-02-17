#!/usr/bin/env python3
"""
Optuna最適パラメータをC++に埋め込んで提出用バイナリを生成するスクリプト

使い方:
  python embed_params.py best_params.cfg solution.cpp > solution_submit.cpp

これにより、configファイル不要の自己完結したC++ファイルが生成される。
"""

import sys
import re


def main():
    if len(sys.argv) < 3:
        print("Usage: python embed_params.py <params.cfg> <solution.cpp>", file=sys.stderr)
        sys.exit(1)

    cfg_path = sys.argv[1]
    cpp_path = sys.argv[2]

    # パラメータ読み込み
    params = {}
    with open(cfg_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                params[parts[0]] = float(parts[1])

    # C++ソース読み込み
    with open(cpp_path) as f:
        cpp = f.read()

    # HyperParams構造体のデフォルト値を置き換え
    field_map = {
        "phase1": "double phase1",
        "phase2": "double phase2",
        "wa_early": "double wa_early", "wb_early": "double wb_early",
        "wc_early": "double wc_early", "wd_early": "double wd_early",
        "wa_mid": "double wa_mid", "wb_mid": "double wb_mid",
        "wc_mid": "double wc_mid", "wd_mid": "double wd_mid",
        "wa_late": "double wa_late", "wb_late": "double wb_late",
        "wc_late": "double wc_late", "wd_late": "double wd_late",
        "leader_mult": "double leader_mult",
        "ucb_c": "double ucb_c",
        "eval_expand": "double eval_expand",
        "eval_level": "double eval_level",
        "eval_reach": "double eval_reach",
        "rollout_depth": "int rollout_depth",
        "num_particles": "int num_particles",
        "pf_noise_w": "double pf_noise_w",
        "pf_noise_eps": "double pf_noise_eps",
        "u_wb_boost": "double u_wb_boost",
        "u_wd_penalty": "double u_wd_penalty",
        "m_leader_scale": "double m_leader_scale",
    }

    for param_name, val in params.items():
        if param_name in field_map:
            field_type = field_map[param_name]
            # "double phase1 = 0.3;" のようなパターンを探して置換
            type_keyword = field_type.split()[0]
            var_name = field_type.split()[1]

            if type_keyword == "int":
                new_val = str(int(val))
            else:
                new_val = f"{val:.8f}"

            pattern = rf'({type_keyword}\s+{var_name}\s*=\s*)([\d.eE\-+]+)(;)'
            replacement = rf'\g<1>{new_val}\3'
            cpp_new = re.sub(pattern, replacement, cpp)
            if cpp_new != cpp:
                cpp = cpp_new
            else:
                print(f"[WARN] パラメータ '{param_name}' の埋め込みに失敗", file=sys.stderr)

    # loadParams呼び出しを無効化 (提出時はconfigファイルなし)
    cpp = cpp.replace(
        'if (argc >= 2) loadParams(argv[1]);',
        '// if (argc >= 2) loadParams(argv[1]); // Params hardcoded by embed_params.py'
    )

    print(cpp)
    print(f"// Embedded params from: {cfg_path}", file=sys.stderr)
    print("[OK] 提出用C++コードを標準出力に出力しました", file=sys.stderr)


if __name__ == "__main__":
    main()
