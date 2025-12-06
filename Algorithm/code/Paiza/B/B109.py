N,H,W,P,Q = map(int,input().split())
grid = [[0]*W for _ in range(H)]

def manhattan_distance(x1,y1,x2,y2):
    return abs(x1-x2) + abs(y1-y2)

for _ in range(N):
    x,y = map(int,input().split())
    grid[x][y] = 1

print(grid)

ans = {}
min_ans = 1000000000

for x in range(H):
    for y in range(W):
        if grid[x][y] == 0:
            ans[x,y] = manhattan_distance(x,y,P,Q)
            min_ans = min(min_ans,manhattan_distance(x,y,P,Q))

#print(min_ans)

for x in range(H):
    for y in range(W):
        if (x,y) in ans:
            if ans[x,y] == min_ans:
                print(x,y)

       