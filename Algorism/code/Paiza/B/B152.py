N,X = map(int,input().split())
LIST = []
USE = []
count = []
for _ in range(2):
    LIST.append(list(map(int,input().split())))

for x in range(N):
    count.append([x])

for x in range(N-1):
    for y in range(x+1,N):
        count.append([x,y])

for x in range(N-2):
    for y in range(x+1,N-1):
        for z in range(y+1,N):
            count.append([x,y,z])

ans = []
for x in range(len(count)):
    sum = 0
    sum2 = 0
    for y in range(len(count[x])):
        sum += LIST[1][count[x][y]-1]
        sum2 += LIST[0][count[x][y]-1]
    if sum2 <= X:
        ans.append(sum)

print(max(ans))
      