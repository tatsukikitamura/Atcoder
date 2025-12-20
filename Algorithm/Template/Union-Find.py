from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parent = [-1] * n  # 負の値はルートを表し、その絶対値がグループのサイズ

    def find(self, x):
        if self.parent[x] < 0:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
            return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        # サイズの大きい方に小さい方をつなげる（Union by size）
        if self.parent[root_x] > self.parent[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_x] += self.parent[root_y]
        self.parent[root_y] = root_x
        return True

    def size(self, x):
        return -self.parent[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)
    
    def groups(self):
        groups = defaultdict(list)
        for i in range(self.n):
            groups[self.find(i)].append(i)
        return list(groups.values())