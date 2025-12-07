import sys
input = sys.stdin.readline

class Node:
    """
    ノード情報の管理
    Attributes:
        index (int): ノード番号
        nears (list): 隣接リスト（隣接するノード番号を格納）
        arrival (bool): 探索済フラグ（到達済の場合Trueが返される）
    """

    def __init__(self,index):
        self.index = index
        self.nears = []
        self.arrival = False

    def __repr__(self):
        return f"(index:{self.index}, nears:{self.nears}, arrival:{self.arrival})"

N,M = map(int,input().split())
graph = [Node(i) for i in range(N)]
query = []
for _ in range(M):
    A,B = map(int,input().split())
    graph[B-1].nears.append(A-1)

def dfs(node):
    stack = [node]
    while stack:
        node = stack.pop()

        for near in graph[node].nears:
            if not graph[near].arrival:
                stack.append(near)
                graph[near].arrival = True

Q = int(input())

for _ in range(Q):
    A,B = map(int,input().split())
    if A == 1:
        if not graph[B-1].arrival:  
            graph[B-1].arrival = True
            dfs(B-1)
    
    elif A == 2:
        if graph[B-1].arrival:
            print("Yes")
        else:
            print("No")
    #print(graph)



