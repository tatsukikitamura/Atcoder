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
    return list(accumulate([0] + arr))

# デバッグ用
def debug(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

