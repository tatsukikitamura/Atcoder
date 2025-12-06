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
