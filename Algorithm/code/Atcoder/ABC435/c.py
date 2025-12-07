N = int(input())
a_list = list(map(int,input().split()))
domino = [0] * N

for x in range(N):
    if a_list[x] == 1:
        continue
    domino[x] += 1
    if x+a_list[x] < N:
        domino[x+a_list[x]-1] -= 1
    else:
        domino[N-1] -= 1
#print(domino)

count = 0
ans = N
for x in range(N):
    count += domino[x]
    if count == 0:
        ans = x+1
        break


print(ans)