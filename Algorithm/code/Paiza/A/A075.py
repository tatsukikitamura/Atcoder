class Node:
    """
    ノード情報の管理
    Attributes:
        index (int): ノード番号
        nears (list): 隣接リスト（隣接するノード番号を格納）
        arrival (bool): 探索済フラグ（到達済の場合Trueが返される）
    """
    def __init__(self, height, width, depth):
        self.height = height
        self.width = width
        self.depth = depth
        self.nears = []
    def __repr__(self):
        return f"(height:{self.height}, width:{self.width}, depth:{self.depth}, nears:{self.nears})"


N = int(input())

height_width_depth = []
nodes = []

for _ in range(N):
    H,W,D = map(int,input().split())
    nodes.append(Node(H,W,D))

for node in nodes:
    for other_node in nodes:
        if node.width >= other_node.width and node.depth >= other_node.depth and node != other_node:
            node.nears.append(other_node)

#print(nodes)

ans = 0

for x in range(N):
    use_nodes = nodes[:]
    max_ans = 0
    use_ans = []
    stack = []
    stack.append(use_nodes[x])
    print(stack)
    while stack:
        node = stack.pop()
        max_ans += node.height
        if node.nears != []:
            for near in use_nodes.nears:
                near.height += max_ans
                stack.append(near)
        else:
            use_ans.append(node.height)
    ans = max(ans,max(use_ans))

print(ans)

        





            
        


