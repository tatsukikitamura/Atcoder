#!/usr/bin/env python3
"""
入力ファイルの分布を分析するスクリプト
生成仕様:
- N = rand(10, 20)  (10以上20以下)
- K = rand(N, N²)   (N以上N²以下)
- T = rand(X, 2X)   (X以上2X以下、Xは全目的地を順に訪れる最小移動回数)
"""

import os
import re
from collections import defaultdict
import statistics

def parse_input_file(filepath):
    """入力ファイルを解析してN, K, Tを返す"""
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        if len(parts) < 3:
            return None
        N = int(parts[0])
        K = int(parts[1])
        T = int(parts[2])
        return N, K, T

def analyze_distribution():
    """inディレクトリ内の全ファイルを分析"""
    in_dir = "in2"
    files = sorted([f for f in os.listdir(in_dir) if f.endswith('.txt')])
    
    data = []
    N_values = []
    K_values = []
    T_values = []
    K_by_N = defaultdict(list)  # NごとのKの分布
    
    for filename in files:
        filepath = os.path.join(in_dir, filename)
        result = parse_input_file(filepath)
        if result:
            N, K, T = result
            data.append((N, K, T))
            N_values.append(N)
            K_values.append(K)
            T_values.append(T)
            K_by_N[N].append(K)
    
    print(f"分析対象ファイル数: {len(data)}")
    print()
    
    # Nの分布分析
    print("=" * 60)
    print("1. N (盤面サイズ) の分布")
    print("=" * 60)
    print(f"  仕様: rand(10, 20)  (10以上20以下)")
    print(f"  実際の範囲: {min(N_values)} ～ {max(N_values)}")
    print(f"  平均: {statistics.mean(N_values):.2f}")
    print(f"  中央値: {statistics.median(N_values):.2f}")
    print()
    
    N_counts = defaultdict(int)
    for N in N_values:
        N_counts[N] += 1
    
    print("  Nの出現頻度:")
    for n in sorted(N_counts.keys()):
        count = N_counts[n]
        expected = len(data) / 11  # 10-20は11通り
        ratio = count / expected
        print(f"    N={n:2d}: {count:3d}回 ({count/len(data)*100:5.1f}%) [期待値: {expected:.1f}, 比率: {ratio:.2f}]")
    print()
    
    # Kの分布分析（全体）
    print("=" * 60)
    print("2. K (目的地の個数) の分布（全体）")
    print("=" * 60)
    print(f"  仕様: rand(N, N²)  (N以上N²以下)")
    print(f"  実際の範囲: {min(K_values)} ～ {max(K_values)}")
    print(f"  平均: {statistics.mean(K_values):.2f}")
    print(f"  中央値: {statistics.median(K_values):.2f}")
    print()
    
    # NごとのKの分布
    print("=" * 60)
    print("3. K (目的地の個数) の分布（Nごと）")
    print("=" * 60)
    for n in sorted(K_by_N.keys()):
        k_list = K_by_N[n]
        k_min = min(k_list)
        k_max = max(k_list)
        k_avg = statistics.mean(k_list)
        k_median = statistics.median(k_list)
        n_squared = n * n
        expected_min = n
        expected_max = n_squared
        
        print(f"  N={n:2d}: Kの範囲 {k_min:4d} ～ {k_max:4d} (平均: {k_avg:6.1f}, 中央値: {k_median:6.1f})")
        print(f"          仕様範囲: {expected_min:4d} ～ {expected_max:4d}")
        
        # 仕様範囲内かチェック
        if k_min < expected_min or k_max > expected_max:
            print(f"          ⚠️  警告: 仕様範囲外の値があります！")
        print()
    
    # Tの分布分析
    print("=" * 60)
    print("4. T (ステップ数の上限) の分布")
    print("=" * 60)
    print(f"  仕様: rand(X, 2X)  (X以上2X以下、Xは全目的地を順に訪れる最小移動回数)")
    print(f"  実際の範囲: {min(T_values)} ～ {max(T_values)}")
    print(f"  平均: {statistics.mean(T_values):.2f}")
    print(f"  中央値: {statistics.median(T_values):.2f}")
    print()
    
    # Tのヒストグラム（大まかな分布）
    T_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]
    T_hist = defaultdict(int)
    for T in T_values:
        for i in range(len(T_bins) - 1):
            if T_bins[i] <= T < T_bins[i+1]:
                T_hist[(T_bins[i], T_bins[i+1])] += 1
                break
        else:
            T_hist[(T_bins[-1], float('inf'))] += 1
    
    print("  Tの分布（大まかな区間）:")
    for (low, high), count in sorted(T_hist.items()):
        if high == float('inf'):
            print(f"    {low:5d}以上: {count:3d}回 ({count/len(data)*100:5.1f}%)")
        else:
            print(f"    {low:5d}～{high:5d}: {count:3d}回 ({count/len(data)*100:5.1f}%)")
    print()
    
    # 仕様違反のチェック
    print("=" * 60)
    print("5. 仕様違反のチェック")
    print("=" * 60)
    violations = []
    
    for i, (N, K, T) in enumerate(data):
        filename = files[i]
        
        # Nのチェック
        if N < 10 or N > 20:
            violations.append(f"{filename}: N={N} は仕様範囲(10-20)外")
        
        # Kのチェック
        if K < N or K > N * N:
            violations.append(f"{filename}: N={N}, K={K} は仕様範囲({N}-{N*N})外")
    
    if violations:
        print(f"  ⚠️  仕様違反が {len(violations)} 件見つかりました:")
        for v in violations[:20]:  # 最初の20件のみ表示
            print(f"    {v}")
        if len(violations) > 20:
            print(f"    ... 他 {len(violations) - 20} 件")
    else:
        print("  ✓ 仕様違反は見つかりませんでした")
    print()
    
    # 統計サマリー
    print("=" * 60)
    print("6. 統計サマリー")
    print("=" * 60)
    print(f"  総ファイル数: {len(data)}")
    print(f"  Nの範囲: {min(N_values)} ～ {max(N_values)} (仕様: 10-20)")
    print(f"  Kの範囲: {min(K_values)} ～ {max(K_values)}")
    print(f"  Tの範囲: {min(T_values)} ～ {max(T_values)}")
    print()
    
    # Nの分布が一様かチェック（カイ二乗検定の簡易版）
    print("=" * 60)
    print("7. Nの分布の一様性チェック（簡易）")
    print("=" * 60)
    expected_per_N = len(data) / 11  # 10-20は11通り
    chi_square = 0
    for n in range(10, 21):
        observed = N_counts.get(n, 0)
        expected = expected_per_N
        if expected > 0:
            chi_square += (observed - expected) ** 2 / expected
    
    print(f"  カイ二乗統計量: {chi_square:.2f}")
    print(f"  自由度: 10 (11-1)")
    print(f"  カイ二乗検定の臨界値(α=0.05): 18.31")
    if chi_square < 18.31:
        print(f"  ✓ 一様分布と矛盾しない（p > 0.05）")
    else:
        print(f"  ⚠️  一様分布と矛盾する可能性がある（p < 0.05）")
    print()

if __name__ == "__main__":
    analyze_distribution()

