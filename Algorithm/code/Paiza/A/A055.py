class Node:
    """

    区画の情報を管理

    Attributes:
        row (int): 区画の行番号（0-indexed）
        col (int): 区画の列番号（0-indexed）
        nears (list): 隣接リスト
        arrival (bool): 探索フラグ
    """

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.nears = []
        self.arrival = False

    def __repr__(self):
        return f"(arrival:{self.arrival})"


H, W = map(int, input().split())

locs = [input() for _ in range(H)]

stack = []


nodes = [[Node(i, j) for j in range(W)] for i in range(H)]



adjacents = [(0, 1), (0, -1), (1, 0), (-1, 0)]


for i in range(H):
    for j in range(W):
     
        for dx, dy in adjacents:
            nx, ny = i + dx, j + dy
            if (0 <= nx < H) & (0 <= ny < W):
                if locs[nx][ny] != "#":
                    nodes[i][j].nears.append([nx, ny])

        if locs[i][j] == "S":
            stack.append(nodes[i][j])
            nodes[i][j].arrival = True

while stack:

    node = stack.pop()
    nears = node.nears
    for r, c in nears:
        if nodes[r][c].arrival == False:
            nodes[r][c].arrival = True
            stack.append(nodes[r][c])


for x in range(H):
    for y in range(W):
        if x == 0 or x == H-1 or y == 0 or y == W-1:
            if nodes[x][y].arrival == True:
                print("YES")
                exit()
print("NO")