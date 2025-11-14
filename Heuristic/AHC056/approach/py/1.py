import sys
from collections import deque

# sys.setrecursionlimit(2000) # 必要に応じて

# --- 1. 入力読み込み ---
N, K, T = map(int, sys.stdin.readline().split())
V_WALLS = [sys.stdin.readline().strip() for _ in range(N)]
H_WALLS = [sys.stdin.readline().strip() for _ in range(N - 1)]
TARGETS = []
for _ in range(K):
    TARGETS.append(tuple(map(int, sys.stdin.readline().split())))

# --- 2. ヘルパー関数 (壁チェック・移動) ---

# (i, j) から方向 d ('U', 'D', 'L', 'R') へ移動可能か
def can_move(i, j, d):
    if d == 'U':
        if i == 0: return False
        return H_WALLS[i - 1][j] == '0'
    if d == 'D':
        if i == N - 1: return False
        return H_WALLS[i][j] == '0'
    if d == 'L':
        if j == 0: return False
        return V_WALLS[i][j - 1] == '0'
    if d == 'R':
        if j == N - 1: return False
        return V_WALLS[i][j] == '0'
    return False # 'S' や不正な入力

# (i, j) から d 方向に移動した後の座標
def get_next_pos(i, j, d):
    if d == 'U': return (i - 1, j)
    if d == 'D': return (i + 1, j)
    if d == 'L': return (i, j - 1)
    if d == 'R': return (i, j + 1)
    return (i, j)

# pos1 から pos2 への移動方向 (U/D/L/R)
def get_direction(pos1, pos2):
    r1, c1 = pos1
    r2, c2 = pos2
    if r2 < r1: return 'U'
    if r2 > r1: return 'D'
    if c2 < c1: return 'L'
    if c2 > c1: return 'R'
    return 'S' # 同じマス (デバッグ用)

# --- 3. BFSによる最短経路探索 ---
# start_pos から goal_pos への最短経路(座標リスト)を返す
def bfs(start_pos, goal_pos):
    q = deque([(start_pos, [start_pos])]) # (現在地, そこまでのパス)
    visited = {start_pos}

    while q:
        (curr_i, curr_j), path = q.popleft()

        if (curr_i, curr_j) == goal_pos:
            return path # 最短経路を発見

        for d in ['U', 'D', 'L', 'R']:
            if can_move(curr_i, curr_j, d):
                next_pos = get_next_pos(curr_i, curr_j, d)
                if next_pos not in visited:
                    visited.add(next_pos)
                    new_path = path + [next_pos]
                    q.append((next_pos, new_path))
    
    return None # 到達不能 (問題制約上ありえない [cite: 71])

# --- 4. 全経路の計算 & 経路マスの特定 ---
all_paths = []     # 区間ごとの最短経路(座標リスト)を格納
path_cells = set() # 経路上で通過する全てのユニークなマス

for k in range(K - 1):
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    
    path = bfs(start_node, goal_node)
    all_paths.append(path)
    
    # 経路上のマスをすべてセットに追加
    # (start_node は path[0] なので含まれる)
    path_cells.update(path)

# --- 5. 色 (C) と 状態 (Q) の決定と割り当て ---
Q = K # 状態数は目的地の数

# 経路上のマスに 1 から C-1 までの色を割り当て
color_map = {} # (r, c) -> color
current_color = 1
for cell in path_cells:
    if cell not in color_map: # 既に割り当てられていないか確認
        color_map[cell] = current_color
        current_color += 1

# C = (ユニークな経路マス数) + 1 (ダミー色 0)
C = current_color 

# 初期盤面 s の生成 (経路外はすべて 0)
initial_board = [[0] * N for _ in range(N)]
for (r, c), color in color_map.items():
    initial_board[r][c] = color

# --- 6. 遷移規則 (M) の生成 ---
rules = set() # (c, q, A, S, D) のタプルを格納

for k in range(K - 1):
    path = all_paths[k] # k -> k+1 への経路
    current_q = k       # 現在の状態
    
    # パスの各ステップについてルールを生成
    for p in range(len(path) - 1):
        pos = path[p]
        next_pos = path[p+1]
        
        c = color_map[pos] # 現在地の色
        d = get_direction(pos, next_pos) # 移動方向
        A = c # 色は塗り替えない
        
        # 状態遷移の決定
        if next_pos == TARGETS[k+1]:
            # 次の目的地に到着した
            S = k + 1 # 状態を k+1 に更新
        else:
            # まだ移動の途中
            S = k # 状態は k のまま
            
        rules.add((c, current_q, A, S, d))

M = len(rules)

# --- 7. 出力 ---
print(C, Q, M)

for r in range(N):
    print(*(initial_board[r]))

for rule in sorted(list(rules)): # ソートは必須ではないが、デバッグしやすい
    print(*rule)