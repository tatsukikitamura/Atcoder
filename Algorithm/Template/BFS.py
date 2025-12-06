from collections import deque

class Node:
    """
    ノード情報の管理

    Attributes:
        index (int): ノード番号
        nears (list): 隣接リスト（隣接するノード番号を格納）
        arrival (bool): 探索済フラグ（到達済の場合Trueが返される）
    """

    def __init__(self, index):
        self.index = index
        self.nears = []
        self.arrival = False

    def __repr__(self):
        return f"(index:{self.index}, nears:{self.nears}, arrival:{self.arrival})"

def bfs_with_node_class(start_node):
    """
    Nodeクラスのインスタンスを使用してBFSを実行します。
    ノードの 'arrival' 属性を訪問済みフラグとして利用します。
    
    Args:
        start_node (Node): 探索を開始するNodeインスタンス。

    Returns:
        list: 探索で訪問したノードのインデックスの順序リスト。
    """
    
    # キュー（BFS用）：ノードインスタンスを格納
    queue = deque([start_node])
    path_order = []
    
    # ノードクラスの属性を直接操作
    start_node.arrival = True
    
    while queue:
        # キューの先頭からノードインスタンスを取り出す
        node = queue.popleft()
        path_order.append(node.index)
        
        # 隣接ノードを処理
        for neighbor in node.nears:
            if not neighbor.arrival:
                neighbor.arrival = True  # 訪問済みに設定
                queue.append(neighbor)
                    
    return path_order