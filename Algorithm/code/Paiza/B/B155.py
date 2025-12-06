from tarfile import LNKTYPE


stamp_H,stamp_W,N = map(int,input().split())
stamp = []
for _ in range(N):
    use = []
    for _ in range(stamp_H):
        a = (str(input()))
        use.append(a)
    stamp.append(use)

grid_H,grid_W = map(int,input().split())
grid = []
for _ in range(grid_H):
    grid.append(list(map(int,input().split())))



ans = []
for x in range(grid_H):
    use = []    
    for y in range(grid_W):
        use.append(stamp[grid[x][y]-1])
    ans.append(use)


print(ans)


final_ans = []
for x in range(grid_H):
    use = []
   


print(final_ans)