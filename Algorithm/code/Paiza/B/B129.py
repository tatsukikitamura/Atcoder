n,m = map(int,input().split())
h,w = map(int,input().split())

grid = [[0]*w for _ in range(h)]


ans = {}


for _ in range(n):
    a,b,c,d,e = map(int,input().split())
    for x in range(a-1,b):
        for y in range(c-1,d):
            if grid[x][y] == 0:
                grid[x][y] = e
            else:

                if grid[x][y] not in ans:
                    ans[grid[x][y]] = 1
                    grid[x][y] = e
                else:
                    ans[grid[x][y]] += 1
                    grid[x][y] = e

#print(ans)
#print(grid)

for x in range(h):
    for y in range(w):
        if grid[x][y] == 0:
            grid[x][y] = "."
        else:
            grid[x][y] = str(grid[x][y])

for x in range(1,m+1):
    if x in ans:
        print(ans[x])
    else:
        print(0)

for x in range(h):
    print("".join(grid[x]))