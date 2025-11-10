#!/usr/bin/env python3
"""
分布を可視化するスクリプト
"""

import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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

def visualize():
    """分布を可視化"""
    in_dir = "in"
    files = sorted([f for f in os.listdir(in_dir) if f.endswith('.txt')])
    
    data = []
    N_values = []
    K_values = []
    T_values = []
    K_by_N = defaultdict(list)
    
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
    
    # 図の作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Nの分布（ヒストグラム）
    ax1 = axes[0, 0]
    N_counts = defaultdict(int)
    for N in N_values:
        N_counts[N] += 1
    
    N_sorted = sorted(N_counts.keys())
    counts = [N_counts[n] for n in N_sorted]
    expected = len(data) / 11  # 10-20は11通り
    
    x = np.arange(len(N_sorted))
    width = 0.35
    ax1.bar(x - width/2, counts, width, label='実際', alpha=0.7)
    ax1.bar(x + width/2, [expected] * len(N_sorted), width, label='期待値（一様分布）', alpha=0.7)
    ax1.set_xlabel('N (盤面サイズ)')
    ax1.set_ylabel('出現回数')
    ax1.set_title('Nの分布（仕様: rand(10, 20)）')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in N_sorted])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Kの分布（全体）
    ax2 = axes[0, 1]
    ax2.hist(K_values, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('K (目的地の個数)')
    ax2.set_ylabel('出現回数')
    ax2.set_title('Kの分布（全体）')
    ax2.grid(True, alpha=0.3)
    
    # 3. K/N²の比率（正規化）
    ax3 = axes[1, 0]
    K_ratio = []
    for N, K in zip(N_values, K_values):
        if N > 0:
            K_ratio.append(K / (N * N))
    ax3.hist(K_ratio, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('K / N² (正規化)')
    ax3.set_ylabel('出現回数')
    ax3.set_title('K/N²の分布（仕様: rand(N, N²) → 0.0～1.0に正規化）')
    ax3.axvline(0.0, color='r', linestyle='--', alpha=0.5, label='下限 (K=N)')
    ax3.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='上限 (K=N²)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Tの分布（対数スケール）
    ax4 = axes[1, 1]
    ax4.hist(T_values, bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('T (ステップ数の上限)')
    ax4.set_ylabel('出現回数')
    ax4.set_title('Tの分布')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("分布グラフを 'distribution_analysis.png' に保存しました。")
    
    # 詳細統計を出力
    print("\n詳細統計:")
    print(f"  Nの範囲: {min(N_values)} ～ {max(N_values)} (仕様: 10-20)")
    print(f"  Kの範囲: {min(K_values)} ～ {max(K_values)}")
    print(f"  Tの範囲: {min(T_values)} ～ {max(T_values)}")
    print(f"  K/N²の範囲: {min(K_ratio):.3f} ～ {max(K_ratio):.3f} (期待値: 0.0～1.0)")

if __name__ == "__main__":
    try:
        visualize()
    except ImportError:
        print("matplotlibがインストールされていません。")
        print("インストール: pip install matplotlib")

