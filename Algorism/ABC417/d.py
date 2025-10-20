import sys
input = sys.stdin.readline

N = int(input())
presents = [list(map(int, input().split())) for _ in range(N)]
Q = int(input())
queries = [int(input()) for _ in range(Q)]

K = 5000 

# dp[t]には、初期テンションtのときの最終的なテンションが入る
# 最初は、操作がないので初期テンションと結果は同じ
dp = list(range(K + 1))

# O(N * K)の前計算パート
for p, a, b in presents:
    # i番目のプレゼントを適用した後の結果を保存する新しいdpテーブル
    new_dp = [0] * (K + 1)
    for i in range(K + 1):
        # i-1番目のプレゼント適用後のテンション
        tension = dp[i]
        
        # プレゼントの価値Pとテンションを比較して、ルールを適用
        if p >= tension:
            result_tension = tension + a
        else:
            result_tension = tension - b
            if result_tension < 0:
                result_tension = 0
        
        new_dp[i] = result_tension
    
    # dpテーブルを更新
    dp = new_dp

# O(Q)で各クエリに回答
for x in queries:
    if x <= K:
        # K以下なら前計算の結果を出力
        print(dp[x])
    else:
        # Kより大きい場合、テンションの差は1ずつ増減すると仮定して計算
        # f(x) = f(K) + (x - K)
        print(dp[K] + (x - K))