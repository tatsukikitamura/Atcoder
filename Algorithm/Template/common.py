"""
競技プログラミング用の便利ライブラリと関数集

このファイルには、競技プログラミングでよく使われる標準ライブラリのimportと
便利関数が含まれています。

使用方法:
    from Template.common import *
    または
    from Template.common import input, int_input, map_int, list_int
"""

import sys
from collections import deque, defaultdict, Counter
from itertools import accumulate, permutations, combinations, product
from bisect import bisect_left, bisect_right
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
import math
from functools import lru_cache, reduce
from operator import add, mul, sub, truediv

# ============================================================================
# 高速入力関数
# ============================================================================

def input() -> str:
    """
    標準入力から1行を読み取り、末尾の改行を除去して返します。
    sys.stdin.readline()を使用して高速化しています。
    
    Returns:
        str: 入力された文字列（改行文字を除く）
    """
    return sys.stdin.readline().rstrip()


def int_input() -> int:
    """
    標準入力から1つの整数を読み取ります。
    
    Returns:
        int: 入力された整数
    """
    return int(input())


def map_int() -> map:
    """
    標準入力から複数の整数を読み取り、mapオブジェクトを返します。
    
    Returns:
        map: 整数のmapオブジェクト
    
    Example:
        a, b, c = map_int()  # "1 2 3" を入力すると a=1, b=2, c=3
    """
    return map(int, input().split())


def list_int() -> list[int]:
    """
    標準入力から整数のリストを読み取ります。
    
    Returns:
        list[int]: 整数のリスト
    
    Example:
        A = list_int()  # "1 2 3 4 5" を入力すると A=[1,2,3,4,5]
    """
    return list(map_int())


def list_str() -> list[str]:
    """
    標準入力から文字列のリストを読み取ります。
    
    Returns:
        list[str]: 文字列のリスト
    
    Example:
        S = list_str()  # "abc def ghi" を入力すると S=['abc','def','ghi']
    """
    return input().split()


# ============================================================================
# 数学・数値計算の便利関数
# ============================================================================

def gcd(a: int, b: int) -> int:
    """
    最大公約数（GCD）を計算します。
    
    Args:
        a (int): 整数1
        b (int): 整数2
    
    Returns:
        int: aとbの最大公約数
    """
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    """
    最小公倍数（LCM）を計算します。
    
    Args:
        a (int): 整数1
        b (int): 整数2
    
    Returns:
        int: aとbの最小公倍数
    """
    return a * b // math.gcd(a, b)


def is_prime(n: int) -> bool:
    """
    数が素数かどうかを判定します。
    
    Args:
        n (int): 判定する数
    
    Returns:
        bool: nが素数の場合True
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def divisors(n: int) -> list[int]:
    """
    数の約数をすべて取得します。
    
    Args:
        n (int): 約数を求める数
    
    Returns:
        list[int]: 約数のリスト（ソート済み）
    """
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)


# ============================================================================
# リスト・配列操作の便利関数
# ============================================================================

def cumsum(arr: list[int]) -> list[int]:
    """
    累積和を計算します。
    
    Args:
        arr (list[int]): 元の配列
    
    Returns:
        list[int]: 累積和の配列（長さはlen(arr)+1、最初の要素は0）
    
    Example:
        cumsum([1, 2, 3, 4]) -> [0, 1, 3, 6, 10]
    """
    return [0] + list(accumulate(arr))


def transpose(matrix: list[list]) -> list[list]:
    """
    2次元配列を転置します。
    
    Args:
        matrix (list[list]): 2次元配列
    
    Returns:
        list[list]: 転置された2次元配列
    
    Example:
        transpose([[1,2,3], [4,5,6]]) -> [[1,4], [2,5], [3,6]]
    """
    return list(zip(*matrix))


# ============================================================================
# デバッグ用関数
# ============================================================================

def debug(*args, **kwargs):
    """
    デバッグ用のprint関数。
    開発時のみ使用し、提出時にはコメントアウトまたは削除してください。
    
    Args:
        *args: 出力する値
        **kwargs: print関数のキーワード引数
    """
    print(*args, **kwargs, file=sys.stderr)


# ============================================================================
# 再帰制限の設定
# ============================================================================

def set_recursion_limit(limit: int = 2000000) -> None:
    """
    再帰の深さ制限を設定します。
    深い再帰を使う場合に呼び出してください。
    
    Args:
        limit (int): 再帰の深さ制限（デフォルト: 2000000）
    """
    sys.setrecursionlimit(limit)


# ============================================================================
# 使用例
# ============================================================================

if __name__ == "__main__":
    # 入力の例
    print("=== 入力の例 ===")
    # N = int_input()
    # A, B = map_int()
    # arr = list_int()
    
    # 数学関数の例
    print(f"gcd(12, 18) = {gcd(12, 18)}")  # 6
    print(f"lcm(12, 18) = {lcm(12, 18)}")  # 36
    print(f"is_prime(17) = {is_prime(17)}")  # True
    print(f"divisors(12) = {divisors(12)}")  # [1, 2, 3, 4, 6, 12]
    
    # リスト操作の例
    print(f"cumsum([1, 2, 3, 4]) = {cumsum([1, 2, 3, 4])}")
    print(f"transpose([[1,2,3], [4,5,6]]) = {transpose([[1,2,3], [4,5,6]])}")
    
    # collectionsの例
    dq = deque([1, 2, 3])
    dq.appendleft(0)
    print(f"deque: {dq}")
    
    dd = defaultdict(int)
    dd['a'] += 1
    print(f"defaultdict: {dd}")
    
    cnt = Counter([1, 2, 2, 3, 3, 3])
    print(f"Counter: {cnt}")
    
    # bisectの例
    arr = [1, 3, 5, 7, 9]
    idx = bisect_left(arr, 5)
    print(f"bisect_left([1,3,5,7,9], 5) = {idx}")  # 2
    
    # heapqの例
    heap = [3, 1, 4, 1, 5]
    heapify(heap)
    print(f"heap: {heap}")
    heappush(heap, 2)
    print(f"after push 2: {heap}")
    min_val = heappop(heap)
    print(f"min value: {min_val}, heap: {heap}")

