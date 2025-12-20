import sys


T = int(input())

for _ in range(T):
    N = int(input())
    costs = []
    total_power = 0
    for _ in range(N):
        w,p = map(int,input().split())    
        costs.append(w + p)
        total_power += p
    costs.sort()
    ans = 0
    cost_sum = 0
    for c in costs:
        if cost_sum + c <= total_power:
            cost_sum += c
            ans += 1
        else:
            break
    print(ans)