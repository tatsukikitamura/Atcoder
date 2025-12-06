import sys

# 標準入力を高速化
input = sys.stdin.readline

K = int(input())
Blocks = []
N = 0 # 総長 N

for _ in range(K):
    M, X = map(int, input().split())
    Blocks.append((M, X))
    N += M

NUM = N // 2 # 前半の長さ N/2
count = 0

# --- right ポインタの初期位置決め ---
right_idx = 0
right_M = 0   # rightブロックの残り個数
skip_count = NUM # NUM 個だけスキップする

for i in range(K):
    M, X = Blocks[i]
    if skip_count >= M:
        # このブロックは丸ごと前半 (A_1 ... A_NUM) に含まれる
        skip_count -= M
    else:
        # A_{NUM+1} はこのブロックにある
        right_idx = i
        # このブロックのうち、後半 (A_{NUM+1} ...) に含まれる個数
        right_M = M - skip_count 
        skip_count = 0 # スキップ完了
        break # right の初期位置が確定したのでループを抜ける

# --- left と right の同時走査 (尺取り法) ---
left_idx = 0
left_M = Blocks[left_idx][0] # leftブロックの残り個数

processed_count = 0 # 処理済みのペアの数

while processed_count < NUM:
    # 現在の left と right の値を取得
    left_X = Blocks[left_idx][1]
    right_X = Blocks[right_idx][1]

    # leftブロックとrightブロックで、同時に処理できるペアの数
    overlap = min(left_M, right_M)

    # overlap個のペア |left_X - right_X| を一括で計算
    count += abs(left_X - right_X) * overlap

    # 処理した分だけ両方の残量を減らす
    left_M -= overlap
    right_M -= overlap
    processed_count += overlap # 処理済みペア数を加算

    # left ブロックを使い切ったら、次の left ブロックへ
    if left_M == 0:
        left_idx += 1
        if processed_count < NUM: # まだ処理が残っている場合のみ更新
           left_M = Blocks[left_idx][0]

    # right ブロックを使い切ったら、次の right ブロックへ
    if right_M == 0:
        right_idx += 1
        if right_idx < K: 
            right_M = Blocks[right_idx][0]

print(count)