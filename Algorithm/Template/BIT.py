class BIT:
    """
    Binary Indexed Tree (Fenwick Tree) の実装
    
    1-indexed の配列に対して、以下の操作を O(logN) で行えます：
    - add(i, x): 位置 i に値 x を加算
    - sum(i): 位置 1 から i までの累積和を取得
    - query_range(l, r): 区間 [l, r] の和を取得
    
    Attributes:
        n (int): 配列のサイズ
        data (list): BITの内部データ（1-indexed）
    
    Time Complexity:
        - __init__: O(N)
        - add: O(logN)
        - sum: O(logN)
        - query_range: O(logN)
    
    Space Complexity: O(N)
    """
    
    def __init__(self, n: int):
        """
        BITを初期化します。
        
        Args:
            n (int): 配列のサイズ（0-indexedでn個の要素）
        """
        self.n = n
        self.data = [0] * (n + 1)  # 1-indexed
    
    def add(self, i: int, x: int) -> None:
        """
        位置 i に値 x を加算します。
        
        Args:
            i (int): 更新する位置（1-indexed、1 <= i <= n）
            x (int): 加算する値
        """
        while i <= self.n:
            self.data[i] += x
            i += i & (-i)  # 最下位ビットを取得して加算
    
    def sum(self, i: int) -> int:
        """
        位置 1 から i までの累積和を取得します。
            i -= i & (-i)  
        Args:
            i (int): 累積和を取得する終了位置（1-indexed、1 <= i <= n）
        
        Returns:
            int: 位置 1 から i までの累積和
        """
        s = 0
        while i > 0:
            s += self.data[i]
            i -= i & (-i)  # 最下位ビットを取得して減算
        return s
    
    def query_range(self, l: int, r: int) -> int:
        """
        区間 [l, r] の和を取得します。
        
        Args:
            l (int): 区間の開始位置（1-indexed、1 <= l <= n）
            r (int): 区間の終了位置（1-indexed、1 <= r <= n）
        
        Returns:
            int: 区間 [l, r] の和（l > r の場合は 0）
        """
        if l > r:
            return 0
        return self.sum(r) - self.sum(l - 1)


# 使用例
if __name__ == "__main__":
    # 例1: 基本的な使用
    bit = BIT(10)
    
    # 位置 3 に 5 を加算
    bit.add(3, 5)
    # 位置 5 に 3 を加算
    bit.add(5, 3)
    # 位置 3 に 2 を加算
    bit.add(3, 2)
    
    # 累積和の取得
    print(f"sum(5) = {bit.sum(5)}")  # 5 + 3 + 2 = 10
    print(f"sum(2) = {bit.sum(2)}")  # 0
    
    # 区間和の取得
    print(f"query_range(3, 5) = {bit.query_range(3, 5)}")  # 5 + 2 + 3 = 10
    
    # 例2: 配列の初期化
    arr = [1, 2, 3, 4, 5]
    bit2 = BIT(len(arr))
    for i in range(len(arr)):
        bit2.add(i + 1, arr[i])  # 1-indexedなので +1
    
    print(f"配列の累積和: sum(3) = {bit2.sum(3)}")  # 1 + 2 + 3 = 6
    print(f"区間[2, 4]の和: {bit2.query_range(2, 4)}")  # 2 + 3 + 4 = 9

