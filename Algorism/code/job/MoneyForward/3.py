N = int(input())
user_initial = []

for _ in range(N):
    A,B = map(int,input().split())
    user_initial.append([A,B,False])


count = 0
for x in range(N):
    for y in range(x+1,N):
        if user_initial[x][0] == user_initial[y][1] and user_initial[x][2] == False and user_initial[y][2] == False:
            if user_initial[x][1] == user_initial[y][0]:
                count += 1
                user_initial[x][2] = True
                user_initial[y][2] = True
        
print(count)
