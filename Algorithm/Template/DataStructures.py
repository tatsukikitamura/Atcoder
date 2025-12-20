"""
Python データ構造比較: List, Dict, Set

=============================================================================
| 操作           | List          | Dict          | Set           | tuple
=============================================================================
| 追加           | append(): O(1)| d[k]=v: O(1)  | add(): O(1)   | tuple.append(): O(1)
| 削除           | remove(): O(n)| del d[k]: O(1)| remove(): O(1)| remove(): O(1)
|                | pop(): O(1)*  | pop(): O(1)   | discard(): O(1)| pop(): O(1)*
| 検索(in)       | O(n)          | O(1)          | O(1)          | O(1)
| インデックス   | O(1)          | O(1)          | ✗ 不可        | O(1)
| 順序保持       | ○             | ○ (3.7+)      | ✗             | ○
| 重複許可       | ○             | キー: ✗       | ✗             | ○
| ミュータブル   | ○             | ○             | ○             | ✗
=============================================================================

【使い分けの指針】
- List: 順序が重要、インデックスアクセスが必要、重複を許可
- Dict: キーと値のペア、高速な検索が必要
- Set: 重複排除、高速な存在確認、集合演算
"""

# =============================================================================
# List (リスト)
# =============================================================================
"""
特徴:
- 順序付きのシーケンス
- インデックスでアクセス可能
- 重複要素を許可
- ミュータブル（変更可能）

主な用途:
- 順序が重要なデータの保存
- スタック/キュー的な操作
- インデックスによる高速アクセス
"""

# 基本操作
lst = [1, 2, 3]
lst.append(4)           # 末尾に追加: O(1)
lst.insert(0, 0)        # 指定位置に挿入: O(n)
lst.pop()               # 末尾から削除: O(1)
lst.pop(0)              # 先頭から削除: O(n)
lst.remove(2)           # 値を指定して削除: O(n)
x = lst[0]              # インデックスアクセス: O(1)
exists = 3 in lst       # 存在確認: O(n) ← 遅い！

# リスト内包表記
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]


# =============================================================================
# Dict (辞書)
# =============================================================================
"""
特徴:
- キーと値のペアを保存
- キーは一意（重複不可）
- キーでの検索が O(1)
- Python 3.7+ で挿入順序を保持

主な用途:
- キーによる高速な検索/更新
- カウンター（collections.Counter）
- グラフの隣接リスト表現
"""

# 基本操作
d = {'a': 1, 'b': 2}
d['c'] = 3              # 追加/更新: O(1)
del d['a']              # 削除: O(1)
val = d.get('x', 0)     # 存在しない場合のデフォルト値
exists = 'b' in d       # 存在確認: O(1) ← 高速！

# よく使うパターン
# カウンター
from collections import Counter
cnt = Counter([1, 1, 2, 3, 3, 3])  # {3: 3, 1: 2, 2: 1}

# デフォルト値付き辞書
from collections import defaultdict
graph = defaultdict(list)
graph[1].append(2)      # キーがなくても自動で空リスト作成
#defaultdict(<class "list">, {1: [2]})

# 辞書内包表記
squares = {x: x**2 for x in range(5)}


# =============================================================================
# Set (集合)
# =============================================================================
"""
特徴:
- 重複なしの要素の集まり
- 順序は保証されない
- 存在確認が O(1)
- 集合演算が可能

主な用途:
- 重複の排除
- 高速な存在確認
- 集合演算（和、積、差）
"""

# 基本操作
s = {1, 2, 3}
s.add(4)                # 追加: O(1)
s.remove(2)             # 削除（なければエラー）: O(1)
s.discard(5)            # 削除（なくてもエラーなし）: O(1)
exists = 3 in s         # 存在確認: O(1) ← 高速！

# 集合演算
a = {1, 2, 3}
b = {2, 3, 4}
union = a | b           # 和集合: {1, 2, 3, 4}
intersect = a & b       # 積集合: {2, 3}
diff = a - b            # 差集合: {1}
sym_diff = a ^ b        # 対称差: {1, 4}

# リストから重複を除去
lst = [1, 1, 2, 2, 3]
unique = list(set(lst)) # [1, 2, 3] ※順序は保証されない

# 集合内包表記
evens = {x for x in range(10) if x % 2 == 0}


# =============================================================================
# 競プロでの使い分け例
# =============================================================================

# 例1: 訪問済みチェック（BFS/DFS）→ Set を使う
visited = set()
if node not in visited:  # O(1)
    visited.add(node)

# 例2: グラフの隣接リスト → Dict + List を使う
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# 例3: 要素の出現回数 → Counter を使う
from collections import Counter
cnt = Counter(arr)
most_common = cnt.most_common(3)  # 上位3つ

# 例4: 重複チェック → Set を使う
def has_duplicate(arr):
    return len(arr) != len(set(arr))

# 例5: 2つの配列の共通要素 → Set を使う
common = set(arr1) & set(arr2)


# =============================================================================
# 注意点
# =============================================================================
"""
1. List の in 演算は O(n) なので、頻繁に検索するなら Set か Dict を使う
2. Dict と Set のキーはハッシュ可能な型のみ（list は不可、tuple は可）
3. Set は順序を保証しないので、順序が必要なら list(sorted(s)) を使う
4. defaultdict は存在しないキーにアクセスすると自動で作成される
"""


# =============================================================================
# Tuple (タプル)
# =============================================================================
"""
特徴:
- イミュータブル（変更不可）なシーケンス
- リストと似ているが、一度作成すると変更できない
- ハッシュ可能なので、Dict のキーや Set の要素にできる
- リストより若干メモリ効率が良い

主な用途:
- 座標 (x, y) の管理
- Dict のキーとして使用
- 複数の値をまとめて返す
- heapq で複数の値を扱う
"""

# 基本操作
t = (1, 2, 3)
t = 1, 2, 3             # 括弧は省略可能
single = (1,)           # 要素1つの場合はカンマ必須
x = t[0]                # インデックスアクセス: O(1)
# t[0] = 10             # エラー！変更不可

# アンパック
a, b, c = (1, 2, 3)
first, *rest = (1, 2, 3, 4)  # first=1, rest=[2, 3, 4]

# Dict のキーとして使用（リストは不可）
visited = {}
visited[(0, 0)] = True  # OK: タプルはキーにできる
# visited[[0, 0]] = True  # エラー！リストはキーにできない

# Set の要素として使用
seen = set()
seen.add((1, 2))        # OK: タプルは追加できる
# seen.add([1, 2])      # エラー！リストは追加できない

# heapq で複数の値を扱う（ダイクストラ法など）
import heapq
h = []
heapq.heappush(h, (0, start_node))  # (距離, 頂点) のタプル
dist, node = heapq.heappop(h)

# ソートのキーとして
points = [(3, 2), (1, 4), (2, 2)]
points.sort()                       # 第1要素 → 第2要素 の順でソート
points.sort(key=lambda p: (p[1], p[0]))  # y → x の順でソート

# 複数の値を返す
def min_max(arr):
    return min(arr), max(arr)       # タプルとして返す
mi, ma = min_max([3, 1, 4, 1, 5])


# =============================================================================
# Deque (両端キュー)
# =============================================================================
"""
特徴:
- 両端からの追加・削除が O(1)
- リストの pop(0) は O(n) だが、deque の popleft() は O(1)
- スレッドセーフ

主な用途:
- BFS（幅優先探索）
- スライディングウィンドウ
- 両端からの操作が必要な場合
"""

from collections import deque

# 基本操作
q = deque()
q = deque([1, 2, 3])    # リストから初期化

# 追加
q.append(4)             # 右端に追加: O(1) → [1, 2, 3, 4]
q.appendleft(0)         # 左端に追加: O(1) → [0, 1, 2, 3, 4]

# 削除
q.pop()                 # 右端から削除: O(1)
q.popleft()             # 左端から削除: O(1) ← list.pop(0) より高速！

# 拡張
q.extend([5, 6])        # 右端に複数追加
q.extendleft([0, -1])   # 左端に複数追加（逆順で追加される）

# 回転
q.rotate(1)             # 右に1つ回転
q.rotate(-1)            # 左に1つ回転

# BFS での使用例
def bfs(start, graph):
    q = deque([start])
    visited = {start}
    while q:
        node = q.popleft()  # O(1) で取り出し
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return visited

# スライディングウィンドウの最大値（長さ k）
def max_sliding_window(nums, k):
    q = deque()  # インデックスを保存
    result = []
    for i, num in enumerate(nums):
        # 範囲外のインデックスを削除
        while q and q[0] < i - k + 1:
            q.popleft()
        # 現在の値より小さい値を削除
        while q and nums[q[-1]] < num:
            q.pop()
        q.append(i)
        if i >= k - 1:
            result.append(nums[q[0]])
    return result


# =============================================================================
# SortedContainers (ソート済みコンテナ) ※外部ライブラリ
# =============================================================================
"""
特徴:
- 要素を常にソートされた状態で保持
- 追加・削除・検索が O(log n)
- C++ の std::set, std::multiset, std::map に相当

主な用途:
- ソート順を維持したまま要素の追加・削除
- k番目に小さい/大きい要素の取得
- 二分探索が必要な場面
"""

# インストール: pip install sortedcontainers
# from sortedcontainers import SortedList, SortedSet, SortedDict

# ----- SortedList（重複あり） -----
# sl = SortedList([3, 1, 4, 1, 5])  # [1, 1, 3, 4, 5] 自動ソート
# sl.add(2)                         # 追加: O(log n) → [1, 1, 2, 3, 4, 5]
# sl.remove(1)                      # 削除: O(log n) → [1, 2, 3, 4, 5]
# sl[0]                             # 最小値: O(1)
# sl[-1]                            # 最大値: O(1)
# sl[2]                             # k番目: O(log n)
# sl.bisect_left(3)                 # 3以上の最小インデックス
# sl.bisect_right(3)                # 3より大きい最小インデックス

# ----- SortedSet（重複なし） -----
# ss = SortedSet([3, 1, 4, 1, 5])   # [1, 3, 4, 5] 重複除去+ソート
# ss.add(2)                         # 追加: O(log n)
# ss.discard(3)                     # 削除: O(log n)
# ss[0]                             # 最小値
# ss[-1]                            # 最大値

# ----- SortedDict（キーでソート） -----
# sd = SortedDict({'c': 3, 'a': 1, 'b': 2})
# sd['d'] = 4                       # 追加
# sd.peekitem(0)                    # 最小キーの (key, value)
# sd.peekitem(-1)                   # 最大キーの (key, value)

# ----- AtCoder での代替手段 -----
# SortedContainers が使えない場合の代替:
# 1. heapq: 最小値の取得のみなら十分
# 2. bisect: ソート済みリストへの二分探索挿入
# 3. 平衡二分探索木を自作（複雑）

import bisect

# bisect を使った擬似的な SortedList
sorted_list = []
bisect.insort(sorted_list, 3)      # ソート順を保って挿入: O(n)
bisect.insort(sorted_list, 1)      # [1, 3]
bisect.insort(sorted_list, 2)      # [1, 2, 3]
idx = bisect.bisect_left(sorted_list, 2)   # 2以上の最小インデックス: O(log n)
# 注意: insort は O(n) なので、大量の挿入には向かない

