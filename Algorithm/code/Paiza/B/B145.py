N,K = map(int,input().split())
List = []

for _ in range(N):
    List.append(list(map(int,input().split())))

c_list = list(map(int,input().split()))

for x in range(N):
    for y in range(N):
        if List[x][y] in c_list:
            List[x][y] = 0
        
count = 0

for x in range(N):
    count1 = 0
    for y in range(N):
        if List[x][y] != 0:
            break
        count1 += 1
    if count1 == N:
        count += 1


for x in range(N):
    count2 = 0
    for y in range(N):
        if List[y][x] != 0:
            break
        count2 += 1
    if count2 == N:
        count += 1

count3 = 0
for x in range(N):
    for y in range(N):
        if x+y == N-1:
            if List[x][y] != 0:
                break
            count3 += 1
    if count3 == N:
        count += 1

count4 = 0
for x in range(N):
    for y in range(N):
        if x == y:
            if List[x][y] != 0:
                break
            count4 += 1
    if count4 == N:
        count += 1

print(count)
            