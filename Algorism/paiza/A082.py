import sys

sys.setrecursionlimit(2000000)

H, W = map(int, input().split())
grid = []


for _ in range(H):
    row_str = input()
    row_int = []
    for char in row_str:
        if char == "#":
            row_int.append(1)
        else:
            row_int.append(0)
    grid.append(row_int)


DX = [1, -1, 0, 0]
DY = [0, 0, 1, -1]

#copyしたリストを汚染するので問題なし
def dfs_paint(x, y, temp_grid, H, W):
    if not (0 <= x < H and 0 <= y < W) or temp_grid[x][y] == 0:
        return   
    # 訪問済みにする
    temp_grid[x][y] = 0
    
    # 4方向へ
    for i in range(4):
        nx = x + DX[i]
        ny = y + DY[i]
        dfs_paint(nx, ny, temp_grid, H, W)
    return



buildable_count = 0 


for r in range(H):
    for c in range(W):
        is_buildable = False 
        if grid[r][c] == 0:
            is_buildable = True        
        else:
            #汚染されないコピー
            temp_grid = [row.copy() for row in grid]
            #コピーを汚染する 
            temp_grid[r][c] = 0
            
            #連結成分の数を数える
            component_count = 0
            for i in range(H):
                for j in range(W):
                    if temp_grid[i][j] == 1: 
                        component_count += 1      
                        dfs_paint(i, j, temp_grid, H, W)
            
            
            if component_count <= 1:
                is_buildable = True
        
   
    
        if is_buildable:
            buildable_count += 1


print(buildable_count)