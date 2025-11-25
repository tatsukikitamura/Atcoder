import copy 

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
        return f"(H:{self.height}, W:{self.width}, D:{self.depth}, nears_count:{len(self.nears)})"


N = int(input())

nodes = []

for _ in range(N):
    H,W,D = map(int,input().split())
    nodes.append(Node(H,W,D))

for i in range(N):
    for j in range(N):
        if i != j:
            node = nodes[i]
            other_node = nodes[j]
           
            if node.width >= other_node.width and node.depth >= other_node.depth:
                       node.nears.append(other_node)


ans = 0

for x in range(N):
    use_nodes = copy.deepcopy(nodes)
    
    stack = [(use_nodes[x], use_nodes[x].height)] 
    max_ans_for_path = use_nodes[x].height 
    
    
    while stack:
        print(stack)
        node, current_height = stack.pop()
 
        max_ans_for_path = max(max_ans_for_path, current_height)
        
        if node.nears:
            for near_node in node.nears:
                new_height = current_height + near_node.height
                stack.append((near_node, new_height))

    ans = max(ans, max_ans_for_path)

print(ans)