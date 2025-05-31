def solve():
    a, b = map(int, input().split())

    if a == 0:
        print(0)
        return

    # 1. 差分配列の準備
    # count_diff[i] は、数字 i (0-indexed) で始まる区間の数と、
    # 数字 i-1 で終わる区間の数の差分を記録します。
    # サイズは a+1 とし、インデックス a は、区間が数字 a-1 (1-indexedでa) で終わる場合の後処理用です。
    count_diff = [0] * (a + 1)

    # 2. 各区間の情報を差分配列に記録
    # 区間 [x, y] (1-indexed) が与えられたら、
    # count_diff[x-1] に +1 (区間の開始点)
    # count_diff[y] に -1 (区間の終了点の「次」の点)
    for _ in range(b):
        x, y = map(int, input().split())  # x, y は 1-indexed
        
        count_diff[x - 1] += 1
        # y は区間の終点(1-indexed)なので、0-indexedでは y-1 が終点。
        # その次の点である count_diff[y] で減少させます。
        # y の最大値は a なので、count_diff[a] へのアクセスが最大です。
        # count_diff のサイズは a+1 なので、これは配列の範囲内です。
        count_diff[y] -= 1
            
    # 3. 差分配列から実際のカウント数を計算（累積和）
    # actual_count[i] は、数字 i+1 がいくつの区間に含まれているかを示します。
    actual_count = [0] * a
    current_sum = 0
    for i in range(a):
        current_sum += count_diff[i]
        actual_count[i] = current_sum

    # 4. カウント数の最小値を出力
    # a > 0 の場合、actual_count は空ではないので min() を安全に呼び出せます。
    print(min(actual_count))

solve()