N = int(input())
K,M = map(int,input().split())
silver = True
sum = 0
count = 0
q = list(map(int,input().split()))

for x in range(N):

    sum += q[x]
    if K <= q[x]:
        count += 1
    
    
if count < 3:
    silver = False

if sum < M:
    silver = False


if silver:
    print("silver")
else:
    print("bronze")