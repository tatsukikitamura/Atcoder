N,M = map(int,input().split())

grid = []

for x in range(N):
    q = list(map(int,input().split()))
    grid.append(q)

ans = []

for x in range(N):
    check_over = False
    for y in range(N):
        if grid[y][x] >= M:
            check_over = True
    
    if not check_over:
        ans.append(x+1)


if ans == []:
    print("wait")
else:
    print(" ".join([str(n) for n in ans]))   

print()  
        
