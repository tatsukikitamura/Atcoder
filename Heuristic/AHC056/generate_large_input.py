import random

# パラメータ設定（より大きな値）
N = 30  # グリッドサイズ（元は15）
K = 200  # ターゲット数（元は106）
T = 5000  # 時間制限（元は1406）

# ランダムシードを固定（再現可能にする）
random.seed(42)

# 出力ファイル
output_file = "input_large.txt"

with open(output_file, 'w') as f:
    # 1行目: N K T
    f.write(f"{N} {K} {T}\n")
    
    # V_WALLS: N行、各行N文字
    # '0'が大部分、たまに'a'などの文字を入れる
    for i in range(N):
        row = ['0'] * N
        # 最初の行だけ'a'を1つ入れる（元のファイルのパターンに合わせる）
        if i == 0:
            pos = random.randint(0, N - 1)
            row[pos] = 'a'
        else:
            # たまにランダムな文字を入れる
            if random.random() < 0.05:  # 5%の確率
                pos = random.randint(0, N - 1)
                row[pos] = random.choice(['a', 'b', 'c'])
        f.write(''.join(row) + '\n')
    
    # H_WALLS: N-1行、各行N文字
    for i in range(N - 1):
        row = ['0'] * N
        # たまにランダムな文字を入れる
        if random.random() < 0.05:  # 5%の確率
            pos = random.randint(0, N - 1)
            row[pos] = random.choice(['a', 'b', 'c'])
        f.write(''.join(row) + '\n')
    
    # TARGETS: K個の座標ペア
    # 0 <= x, y < N の範囲でランダムに生成
    for _ in range(K):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        f.write(f"{x} {y}\n")

print(f"大きなテストケースを生成しました: {output_file}")
print(f"パラメータ: N={N}, K={K}, T={T}")

