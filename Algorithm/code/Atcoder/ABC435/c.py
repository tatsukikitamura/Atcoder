T = int(input())

for _ in range(T):
    sum = 0
    w = []
    p = []
    N= int(input()) # ナップサックの許容量
    for _ in range(N):
        W,P = map(int,input().split())
        w.append(W)
        p.append(P)
        sum += W
    W = sum // 2

    dp = [[[0,0] for x in range(W+1)] for i in range(N+1)] # DPの配列作成

    for i in range(N):
        for j in range(W+1):
            if j < w[i]: # この時点では許容量を超えていないので選択しない
                dp[i+1][j][0] = dp[i][j][0] # ただ選択はしていないが、今回の情報をそのままi+1の方へ移す
                dp[i+1][j][1] += 1# ただ選択はしていないが、今回の情報をそのままi+1の方へ移す
            else:
                dp[i+1][j][0] = max(dp[i][j][0], dp[i][j-w[i]][0]+p[i])
                dp[i+1][j][1] += 1
    
    print(N-dp[N][W][1])




