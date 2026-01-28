import math

N = 10**7
count = [0] * (N + 1)

# x < y を満たす全ペアを列挙
for x in range(1, int(math.sqrt(N)) + 1):
    for y in range(x + 1, int(math.sqrt(N - x*x)) + 1):
        s = x * x + y * y
        if s <= N:
            count[s] += 1

ans = [i for i in range(1, N + 1) if count[i] == 1]

# 提出用コードを生成
print(f"good = {ans}")
print(f"# 個数: {len(ans)}")