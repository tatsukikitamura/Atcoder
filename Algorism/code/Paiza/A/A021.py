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

temp_grid = [row.copy() for row in grid]

DX = [1, -1, 0, 0]
DY = [0, 0, 1, -1]

#copyしたリストを汚染するので問題なし
def dfs_paint(x, y, temp_grid, H, W ,use):
    if not (0 <= x < H and 0 <= y < W) or temp_grid[x][y] == 0:
        return   
    # 訪問済みにする
    use.append([x,y])
    temp_grid[x][y] = 0
    
    # 4方向へ
    for i in range(4):
        nx = x + DX[i]
        ny = y + DY[i]
        dfs_paint(nx, ny, temp_grid, H, W,use)
    return

def dfs_count(x, y, temp_grid, H, W):
    count = 0
    
    # 4方向へ
    for i in range(4):
        nx = x + DX[i]
        ny = y + DY[i]
        if not (0 <= nx < H and 0 <= ny < W):
            #print(f"nx:{nx}, ny:{ny}")
            count += 1

        elif temp_grid[nx][ny] == 1:
            continue

        elif temp_grid[nx][ny] == 0:
            count+= 1

    return count


total_use = []

        #汚染されないコピー
             #コピーを汚染する 
            
        #連結成分の数を数える
for i in range(H):
    for j in range(W):
        use = []
        if grid[i][j] == 1: 
            dfs_paint(i, j, grid, H, W,use)
            total_use.append(use)

   
#print(total_use)


ans = []
for use in total_use:
    count = 0
    for x in use:
        count += dfs_count(x[0], x[1], temp_grid, H, W)

    ans.append([len(use),count])

ans.sort(key=lambda x: (-x[0],-x[1]))

for x in ans:
    print(x[0],x[1])
        
