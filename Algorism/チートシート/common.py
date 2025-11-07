# 便利な関数集

import sys
from collections import deque, defaultdict, Counter
from itertools import accumulate, permutations, combinations
from bisect import bisect_left, bisect_right
from heapq import heapify, heappush, heappop
import math

# 入力関数（高速化）
def input(): return sys.stdin.readline().rstrip()
def int_input(): return int(input())
def map_int(): return map(int, input().split())
def list_int(): return list(map_int())

# 入力関数の使い方と保存される形式
# 
# 1. 文字列1行
#    使用例: s = input()
#    入力例: "hello"
#    保存形式: str型 "hello"
#
# 2. 整数1つ
#    使用例: n = int_input()
#    入力例: "42"
#    保存形式: int型 42
#
# 3. 整数複数（アンパック）
#    使用例: a, b = map_int()
#    入力例: "5 10"
#    保存形式: a = int型 5, b = int型 10
#
# 4. 整数リスト
#    使用例: A = list_int()
#    入力例: "1 2 3 4 5"
#    保存形式: list型 [1, 2, 3, 4, 5]
#
# 5. 文字列リスト（split使用）
#    使用例: words = input().split()
#    入力例: "apple banana orange"
#    保存形式: list型 ["apple", "banana", "orange"]
#
# 6. 複数行入力（リスト内包表記）
#    使用例: 
#    n = int_input()
#    A = [int_input() for _ in range(n)]
#    入力例:
#    3
#    10
#    20
#    30
#    保存形式: A = list型 [10, 20, 30]

# 二分探索
def binary_search(arr, x):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left

# ユークリッドの互除法（最大公約数）
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# 最小公倍数
def lcm(a, b):
    return a * b // gcd(a, b)

# 素因数分解
def prime_factors(n):
    factors = []
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

# 累積和
def cumulative_sum(arr):
    """累積和を計算
    例: [1, 2, 3, 4] -> [0, 1, 3, 6, 10]
    """
    return list(accumulate([0] + arr))

# ============================================
# グラフアルゴリズム
# ============================================

# DFS（深さ優先探索）
def dfs(grid, start_x, start_y, H, W, visited=None):
    """グリッド上のDFS
    4方向（上下左右）に移動可能な場合の連結成分を探索
    
    使用例:
        grid = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
        visited = [[False] * W for _ in range(H)]
        dfs(grid, 0, 0, H, W, visited)
    """
    if visited is None:
        visited = [[False] * W for _ in range(H)]
    
    DX = [1, -1, 0, 0]
    DY = [0, 0, 1, -1]
    
    def _dfs(x, y):
        if not (0 <= x < H and 0 <= y < W) or visited[x][y] or grid[x][y] == 0:
            return
        visited[x][y] = True
        
        for i in range(4):
            nx = x + DX[i]
            ny = y + DY[i]
            _dfs(nx, ny)
    
    _dfs(start_x, start_y)
    return visited

# ============================================
# 動的プログラミング
# ============================================

# 1次元DP（ナップサック問題など）
def knapsack_01(weights, values, capacity):
    """0-1ナップサック問題
    dp[i][w] = i番目までのアイテムで重さw以下での最大価値
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(n):
        for w in range(capacity + 1):
            if w >= weights[i]:
                dp[i+1][w] = max(dp[i][w], dp[i][w-weights[i]] + values[i])
            else:
                dp[i+1][w] = dp[i][w]
    
    return dp[n][capacity]

# 1次元DP（最適化版）
def knapsack_01_optimized(weights, values, capacity):
    """0-1ナップサック問題（メモリ最適化版）"""
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# 2次元DP（グリッドDP）
def grid_dp(grid, H, W):
    """グリッド上のDP
    例: 各マスから移動可能な場合の最大値
    """
    dp = [[0] * W for _ in range(H)]
    dp[0][0] = grid[0][0]
    
    for i in range(H):
        for j in range(W):
            if i > 0:
                dp[i][j] = max(dp[i][j], dp[i-1][j] + grid[i][j])
            if j > 0:
                dp[i][j] = max(dp[i][j], dp[i][j-1] + grid[i][j])
    
    return dp[H-1][W-1]

# ============================================
# BIT（Binary Indexed Tree / Fenwick Tree）
# ============================================

class BIT:
    """Binary Indexed Tree（Fenwick Tree）
    区間和の更新と取得をO(log N)で行う
    
    使用例:
        bit = BIT(10)
        bit.add(1, 5)  # インデックス1に5を加算
        bit.add(3, 3)  # インデックス3に3を加算
        print(bit.sum(3))  # [1, 3]の区間和 = 8
        print(bit.query_range(1, 3))  # [1, 3]の区間和 = 8
    """
    def __init__(self, n):
        self.n = n
        self.data = [0] * (n + 1)
    
    def add(self, i, x):
        """インデックスiにxを加算（1-indexed）"""
        while i <= self.n:
            self.data[i] += x
            i += i & (-i)  # 最下位ビットを取得
    
    def sum(self, i):
        """[1, i]の区間和を取得（1-indexed）"""
        s = 0
        while i > 0:
            s += self.data[i]
            i -= i & (-i)
        return s
    
    def query_range(self, l, r):
        """[l, r]の区間和を取得（1-indexed）"""
        if l > r:
            return 0
        return self.sum(r) - self.sum(l - 1)

# ============================================
# 差分配列
# ============================================

def difference_array(n, intervals):
    """差分配列を使用した区間更新
    複数の区間[l, r]に値を加算する操作を効率的に行う
    
    使用例:
        n = 10
        intervals = [(1, 5, 2), (3, 7, 3)]  # [1,5]に2、[3,7]に3を加算
        result = difference_array(n, intervals)
        # result[i] = i番目の要素の最終的な値
    
    時間計算量: O(n + m) where m = len(intervals)
    """
    diff = [0] * (n + 1)
    
    # 区間更新を差分配列に記録
    for l, r, val in intervals:
        diff[l] += val
        if r + 1 <= n:
            diff[r + 1] -= val
    
    # 累積和を取って実際の値を計算
    result = [0] * n
    current = 0
    for i in range(n):
        current += diff[i]
        result[i] = current
    
    return result

# ============================================
# 尺取り法（Two Pointers）
# ============================================

def two_pointers(arr, target):
    """尺取り法の基本形
    ソート済み配列で、和がtarget以下となる部分配列の数を数える
    
    使用例:
        arr = [1, 2, 3, 4, 5]
        target = 5
        count = two_pointers(arr, target)  # 和が5以下となる部分配列の数
    """
    n = len(arr)
    left = 0
    count = 0
    current_sum = 0
    
    for right in range(n):
        current_sum += arr[right]
        
        while current_sum > target and left <= right:
            current_sum -= arr[left]
            left += 1
        
        count += right - left + 1
    
    return count

def sliding_window_max(arr, k):
    """スライディングウィンドウの最大値
    長さkの連続する部分配列の最大値を全て求める
    
    使用例:
        arr = [1, 3, -1, -3, 5, 3, 6, 7]
        k = 3
        result = sliding_window_max(arr, k)
        # result = [3, 3, 5, 5, 6, 7]
    """
    from collections import deque
    n = len(arr)
    dq = deque()
    result = []
    
    for i in range(n):
        # ウィンドウの範囲外の要素を削除
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # 現在の要素より小さい要素を削除（最大値を保持）
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        
        dq.append(i)
        
        # ウィンドウサイズがkになったら結果に追加
        if i >= k - 1:
            result.append(arr[dq[0]])
    
    return result

# ============================================
# 累積和の応用
# ============================================

def cumulative_sum_2d(grid, H, W):
    """2次元累積和
    グリッド上の矩形領域の和をO(1)で取得
    
    使用例:
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        cumsum = cumulative_sum_2d(grid, 3, 3)
        # [x1, y1]から[x2, y2]の矩形和 = cumsum[x2+1][y2+1] - cumsum[x1][y2+1] - cumsum[x2+1][y1] + cumsum[x1][y1]
    """
    cumsum = [[0] * (W + 1) for _ in range(H + 1)]
    
    for i in range(H):
        for j in range(W):
            cumsum[i+1][j+1] = cumsum[i][j+1] + cumsum[i+1][j] - cumsum[i][j] + grid[i][j]
    
    return cumsum

def query_2d_range(cumsum, x1, y1, x2, y2):
    """2次元累積和から矩形領域の和を取得
    [x1, y1]から[x2, y2]の矩形和を計算（0-indexed、両端含む）
    """
    return cumsum[x2+1][y2+1] - cumsum[x1][y2+1] - cumsum[x2+1][y1] + cumsum[x1][y1]

# デバッグ用
def debug(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

