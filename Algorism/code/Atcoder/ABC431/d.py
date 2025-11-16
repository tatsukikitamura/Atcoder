N = int(input())
items = []
sum_happy = 0
sum_weight = 0

for x in range(N):
    W,H,B = map(int,input().split())
    items.append([W,H-B])
    sum_happy += B
    sum_weight += W

max_weight_change = sum_weight // 2
dp = [0] * int(max_weight_change+1)



for x,y in items:
    for j in range(max_weight_change,x-1,-1):
        dp[j] = max(dp[j],dp[j-x]+y)

#前の方から引いてしまうので最も大きな値になるようにするプロセスが足りない

    
print(max(dp)+sum_happy)    
