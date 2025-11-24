import sys

sys.setrecursionlimit(2000000)

H, W = map(int, input().split())
grid = []


for _ in range(H):
    row_str = input()
    row_int = []
    for x in range(len(row_str)):
        row_int.append(row_str[x])
    grid.append(row_int)

print(grid)
DX = [1, -1, 0, 0]
DY = [0, 0, 1, -1]

#copyしたリストを汚染するので問題なし
def dfs_paint(x, y, grid, H, W,color):
    if not (0 <= x < H and 0 <= y < W) or grid[x][y] != color:
        return   
    # 訪問済みにする
    if grid[x][y] == color:
        grid[x][y] = 0
    
    # 4方向へ
    for i in range(4):
        nx = x + DX[i]
        ny = y + DY[i]
        dfs_paint(nx, ny, grid, H, W,color)
    return

r_count = 0
b_count = 0
g_count = 0


ans = 0
for r in range(H):
    for c in range(W):           
        #連結成分の数を数える
        if grid[r][c] == "R":
            r_count += 1
        elif grid[r][c] == "B":
            b_count += 1
        elif grid[r][c] == "G":
            g_count += 1
        component_count = 0
        if grid[r][c] != 0: 
            #grid[r][c] = 0   
            dfs_paint(r, c, grid, H, W,grid[r][c])
            
       
        
   
print(r_count,g_count,b_count)