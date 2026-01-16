"""
sortedcontainers ライブラリの使い方テンプレート

AtCoderで使用可能な二分探索ベースのソート済みデータ構造。
通常のlistと比較して、検索・挿入・削除が O(log N) で可能。

インストール: pip install sortedcontainers
"""

from sortedcontainers import SortedList, SortedDict, SortedSet


# ============================================================================
# SortedList: ソート状態を維持するリスト
# ============================================================================

def sorted_list_example():
    """
    SortedList の基本的な使い方
    
    計算量比較:
        操作            | list      | SortedList
        ----------------|-----------|------------
        追加            | O(N)      | O(log N)
        削除（値）      | O(N)      | O(log N)
        削除（idx）     | O(N)      | O(log N)
        検索（値）      | O(N)      | O(log N)
        インデックス    | O(1)      | O(log N)
    """
    sl = SortedList([5, 1, 3, 2, 4])
    print(f"初期化: {sl}")  # SortedList([1, 2, 3, 4, 5])
    
    # 追加（自動的にソート位置に挿入）
    sl.add(3)  # O(log N)
    print(f"add(3): {sl}")  # SortedList([1, 2, 3, 3, 4, 5])
    
    # 削除（値で削除）
    sl.remove(3)  # O(log N) - 最初の3を削除
    print(f"remove(3): {sl}")  # SortedList([1, 2, 3, 4, 5])
    
    # 削除（インデックスで削除）
    val = sl.pop(2)  # O(log N)
    print(f"pop(2): {sl}, 削除した値: {val}")
    
    # 二分探索
    sl = SortedList([1, 3, 5, 7, 9])
    print(f"\n二分探索: {sl}")
    print(f"bisect_left(5): {sl.bisect_left(5)}")   # 2 (5以上の最小インデックス)
    print(f"bisect_right(5): {sl.bisect_right(5)}") # 3 (5より大きい最小インデックス)
    
    # 範囲内の要素数
    left = sl.bisect_left(3)
    right = sl.bisect_right(7)
    print(f"3以上7以下の要素数: {right - left}")  # 3 (3, 5, 7)
    
    # k番目に小さい要素
    print(f"2番目に小さい要素: {sl[1]}")  # 3


# ============================================================================
# SortedSet: ソート済みの集合（重複なし）
# ============================================================================

def sorted_set_example():
    """
    SortedSet の基本的な使い方
    set と同様に重複を許さないが、ソート順を維持する。
    """
    ss = SortedSet([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"初期化（重複除去）: {ss}")  # SortedSet([1, 2, 3, 4, 5, 6, 9])
    
    # 追加・削除
    ss.add(7)
    ss.discard(4)  # なくてもエラーにならない
    print(f"add(7), discard(4): {ss}")
    
    # k番目に小さい要素
    print(f"3番目に小さい要素: {ss[2]}")
    
    # ある値のインデックス
    print(f"5のインデックス: {ss.index(5)}")


# ============================================================================
# SortedDict: キーがソートされた辞書
# ============================================================================

def sorted_dict_example():
    """
    SortedDict の基本的な使い方
    キーがソート順に維持される辞書。
    """
    sd = SortedDict({'c': 3, 'a': 1, 'b': 2})
    print(f"初期化: {sd}")  # SortedDict({'a': 1, 'b': 2, 'c': 3})
    
    sd['d'] = 4
    print(f"キー順: {list(sd.keys())}")  # ['a', 'b', 'c', 'd']
    
    # k番目に小さいキー
    print(f"最小のキー: {sd.peekitem(0)}")   # ('a', 1)
    print(f"最大のキー: {sd.peekitem(-1)}")  # ('d', 4)


# ============================================================================
# AtCoderでよく使うパターン
# ============================================================================

def atcoder_patterns():
    """AtCoderでよく使うパターン集"""
    
    # パターン1: 動的な中央値管理
    print("=== 動的な中央値 ===")
    sl = SortedList()
    for x in [5, 2, 8, 1, 9, 3]:
        sl.add(x)
        n = len(sl)
        median = sl[n // 2] if n % 2 == 1 else (sl[n // 2 - 1] + sl[n // 2]) / 2
        print(f"追加: {x}, 中央値: {median}")
    
    # パターン2: 座標圧縮（値 → 順位）
    print("\n=== 座標圧縮 ===")
    values = [100, 30, 50, 30, 80]
    ss = SortedSet(values)
    rank = {v: i for i, v in enumerate(ss)}
    compressed = [rank[v] for v in values]
    print(f"元の値: {values}")
    print(f"圧縮後: {compressed}")  # [3, 0, 1, 0, 2]
    
    # パターン3: lower_bound / upper_bound
    print("\n=== lower_bound / upper_bound ===")
    sl = SortedList([1, 3, 5, 7, 9])
    x = 6
    lb = sl.bisect_left(x)   # x以上の最小インデックス
    ub = sl.bisect_right(x)  # xより大きい最小インデックス
    print(f"x={x}: lower_bound={lb}, upper_bound={ub}")
    
    # x以上の最小値
    if lb < len(sl):
        print(f"{x}以上の最小値: {sl[lb]}")  # 7
    
    # x以下の最大値
    if lb > 0:
        print(f"{x}以下の最大値: {sl[lb - 1]}")  # 5
    
    # パターン4: 範囲内の要素を取得
    print("\n=== 範囲内の要素 ===")
    sl = SortedList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    left, right = 3, 7
    start = sl.bisect_left(left)
    end = sl.bisect_right(right)
    print(f"{left}以上{right}以下の要素: {list(sl[start:end])}")  # [3, 4, 5, 6, 7]


# ============================================================================
# 実行
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SortedList の例")
    print("=" * 60)
    sorted_list_example()
    
    print("\n" + "=" * 60)
    print("SortedSet の例")
    print("=" * 60)
    sorted_set_example()
    
    print("\n" + "=" * 60)
    print("SortedDict の例")
    print("=" * 60)
    sorted_dict_example()
    
    print("\n" + "=" * 60)
    print("AtCoderでよく使うパターン")
    print("=" * 60)
    atcoder_patterns()
