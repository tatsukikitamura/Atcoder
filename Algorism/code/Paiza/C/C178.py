N,M = map(int,input().split())

A = sorted(list(map(int,input().split())))
B = sorted(list(map(int,input().split())))


count = 0

try:
    for x in range(M):
        if A[count] <= B[x]:
            count += 1
except:
    pass
    

if count == N:
    print("YES")
else:
    print("NO")