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
    def __repr__(self):
        return f"(row:{self.row}, col:{self.col}, nears:{self.nears})"

N,R = map(int,input().split())
LIST = []
for _ in range(N):
    LIST.append(list(map(int,input().split())))

def check(x:list,y:list):
    if (x[0]-y[0])**2 + (x[1]-y[1])**2 <= 2*R**2:
        return True
    else:
        return False

stack = []
nodes = []

for x in range(N):
    nodes.append(Node(LIST[x][0],LIST[x][1]))

stack.append(nodes[0])


print(LIST)

node = stack.pop()
print(node)
for node in nodes:
    for x in range(N):
        print(node.row,nodes[x].row,node.col,nodes[x].col)
        if node.row == nodes[x].row and node.col == nodes[x].col:
            continue
        elif check([node.row,node.col],LIST[x]):
            node.nears.append(nodes[x])

print(nodes)
        
