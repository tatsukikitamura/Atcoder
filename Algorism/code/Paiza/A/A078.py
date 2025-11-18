from collections import deque

class Node:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.nears = []
        self.value = True
    def __repr__(self):
        return f"(x:{self.x}, y:{self.y}, nears:{self.nears}, value:{self.value})"



nodes = []

N = int(input())
grid = []
for x in range(N):
    row = str(input())
    row_int = []
    for y in range(N):
        if row[y] == ".":
            row_int.append(0)
        else:
            row_int.append(int(row[y]))
            nodes.append(Node(x,y,row_int[y]))
    grid.append(row_int)

moves = [
            (-1, 0),  # 上
            (1, 0),   # 下
            (0, -1),  # 左
            (0, 1)    # 右
        ]
for node in nodes:
    for move in moves:
        nx = node.x + move[0]
        ny = node.y + move[1]
        if 0 <= nx < N and 0 <= ny < N:
            if grid[nx][ny] == grid[node.x][node.y]:
                node.nears.append([nx,ny])
            else:
                node.value = False
            

while nodes:
    use = []
    for node in nodes:
        if node.value == True:
            use.append([node.x,node.y])
            for near in node.nears:
                use.append(near)
   
    use = [list(x) for x in set(tuple(x) for x in use)]
    print(f"use:{use}")

    for list in use:
        grid[list[0]][list[1]] = 0
        for node in nodes:
            if node.x == list[0] and node.y == list[1]:
                nodes.remove(node)
        
    for x in range(N):
        y = 0
        while True:
            if grid[N-y-1][x] == 0:
                y += 1
            else:  
                break
        for z in range(0,N-y):
            grid[N-y-1][z] = 0









print(f"nodes:{nodes}")




