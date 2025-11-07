N = int(input())
LIST = []
for _ in range(N):
    LIST.append(list(map(int,input().split())))

ANS_LIST = []
for x in range(N):
    USE = []
    for y in range(N):
        USE_NUM =  min(x,N-x-1,y,N-y-1)
        USE.append(USE_NUM+1)
       
    ANS_LIST.append(USE)

count = 0
for x in range(N):
    for y in range(N):
        count += LIST[x][y] - ANS_LIST[x][y]

print(count)





