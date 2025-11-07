import sys
from collections import deque
import heapq # ### NEW ###: 優先度付きキュー(ダイクストラ法)のため

# sys.setrecursionlimit(2000)

# --- 1. 入力読み込み (変更なし) ---
N, K, T = map(int, sys.stdin.readline().split())
V_WALLS = [sys.stdin.readline().strip() for _ in range(N)]
H_WALLS = [sys.stdin.readline().strip() for _ in range(N - 1)]
TARGETS = []
for _ in range(K):
    TARGETS.append(tuple(map(int, sys.stdin.readline().split())))

# --- 2. ヘルパー関数 (変更なし) ---

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
    return False

def get_next_pos(i, j, d):
    if d == 'U': return (i - 1, j)
    if d == 'D': return (i + 1, j)
    if d == 'L': return (i, j - 1)
    if d == 'R': return (i, j + 1)
    return (i, j)

def get_direction(pos1, pos2):
    r1, c1 = pos1
    r2, c2 = pos2
    if r2 < r1: return 'U'
    if r2 > r1: return 'D'
    if c2 < c1: return 'L'
    if c2 > c1: return 'R'
    return 'S'

# --- 3. BFS (最短経路長計算用に残す) ---
# ### CHANGED ###: 経路リストではなく、最短経路長(int)のみを返すように簡略化
def bfs_shortest_length(start_pos, goal_pos):
    q = deque([(start_pos, 0)]) # (現在地, ステップ数)
    visited = {start_pos}

    while q:
        (curr_i, curr_j), steps = q.popleft()

        if (curr_i, curr_j) == goal_pos:
            return steps # 最短ステップ数

        for d in ['U', 'D', 'L', 'R']:
            if can_move(curr_i, curr_j, d):
                next_pos = get_next_pos(curr_i, curr_j, d)
                if next_pos not in visited:
                    visited.add(next_pos)
                    q.append((next_pos, steps + 1))
    
    return float('inf') # 到達不能 (問題制約上ありえない)

# --- 3.5. ダイクストラ法による経路探索 (C最小化) ---
# ### NEW ###
def find_path_dijkstra(start_pos, goal_pos, step_limit, total_path_cells):
    # pq: (cost, steps, pos, path_list)
    # cost: 新規マス数 (最優先)
    # steps: ステップ数 (第二優先)
    pq = [(0, 0, start_pos, [start_pos])] 
    
    # dist[pos][steps] = min_cost
    # (pos, steps) の状態に、最小コスト cost で到達したかを記録
    dist = {} 
    dist[(start_pos, 0)] = 0
    
    # ゴールに到達した経路を (cost, steps, path) で記録
    found_paths = []

    while pq:
        cost, steps, pos, path = heapq.heappop(pq)
        
        # 既にこの (pos, steps) に、より低いコストで到達済みならスキップ
        if dist.get((pos, steps), float('inf')) < cost:
            continue
            
        # ゴールに到達
        if pos == goal_pos:
            heapq.heappush(found_paths, (cost, steps, path))
            # ここで break しない (よりステップ数が多くても cost が低い経路があるかもしれない)
            continue
            
        # ステップ数上限を超えそうなら、それ以上進まない
        if steps + 1 > step_limit:
            continue
            
        (curr_i, curr_j) = pos
        for d in ['U', 'D', 'L', 'R']:
            if can_move(curr_i, curr_j, d):
                next_pos = get_next_pos(curr_i, curr_j, d)
                next_steps = steps + 1
                
                # 新規コストの計算
                new_cost = cost
                if next_pos not in total_path_cells and next_pos not in path:
                    new_cost += 1
                
                # dist の更新チェック
                if dist.get((next_pos, next_steps), float('inf')) > new_cost:
                    dist[(next_pos, next_steps)] = new_cost
                    new_path = path + [next_pos]
                    heapq.heappush(pq, (new_cost, next_steps, next_pos, new_path))
    
    # ゴールに到達した経路のうち、(cost, steps) の辞書順で最小のものを返す
    if found_paths:
        best_cost, best_steps, best_path = heapq.heappop(found_paths)
        return best_path
    
    # 万が一 (step_limit が厳しすぎる場合など) 
    # 見つからなければ None (対策が必要)
    return None 

# --- 4. 全経路の計算 & 経路マスの特定 ---
# ### CHANGED ###: BFS -> ダイクストラ法 に変更

# 4.1. まず全区間の最短経路長 (X_k) と総最短ステップ (X) を計算
shortest_lengths = []
total_shortest_steps_X = 0
for k in range(K - 1):
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    length = bfs_shortest_length(start_node, goal_node)
    shortest_lengths.append(length)
    total_shortest_steps_X += length

# 4.2. ダイクストラ法で経路を決定
all_paths = []
path_cells = {TARGETS[0]} # スタート地点は最初から含む
total_steps_used = 0
X_future = total_shortest_steps_X # これから通過する最短経路長の合計

for k in range(K - 1):
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    X_k = shortest_lengths[k] # この区間の最短長
    
    # この区間で使えるステップ数の上限を計算
    # (全体で残されたステップ) - (未来の区間の最短ステップ合計)
    steps_allowed_total = T - total_steps_used
    margin_for_future = X_future - X_k
    step_limit = steps_allowed_total - margin_for_future
    
    # ダイクストラ法で経路探索
    path = find_path_dijkstra(start_node, goal_node, step_limit, path_cells)

    # ### 安全策 ###
    # もしダイクストラ法が経路を見つけられなかったら (step_limitが厳しすぎた等)
    # BFSで最短経路を強制的に使う (V=K を死守するため)
    if path is None:
        # (この実装では bfs が経路リストを返さないので、再度呼び出す必要がある)
        q = deque([(start_node, [start_node])])
        visited = {start_node}
        while q:
            p, pa = q.popleft()
            if p == goal_node:
                path = pa
                break
            for d in ['U','D','L','R']:
                if can_move(p[0], p[1], d):
                    np = get_next_pos(p[0], p[1], d)
                    if np not in visited:
                        visited.add(np)
                        q.append((np, pa + [np]))
    # ### 安全策 終わり ###

    all_paths.append(path)
    path_cells.update(path) # 新しく通ったマスを追加
    
    # total_steps_used と X_future を更新
    total_steps_used += (len(path) - 1)
    X_future -= X_k


# --- 5. 色 (C) と 状態 (Q) の決定と割り当て (変更なし) ---
Q = K 
color_map = {} 
current_color = 1
for cell in path_cells:
    if cell not in color_map:
        color_map[cell] = current_color
        current_color += 1
C = current_color
initial_board = [[0] * N for _ in range(N)]
for (r, c), color in color_map.items():
    initial_board[r][c] = color

# --- 6. 遷移規則 (M) の生成 (変更なし) ---
rules = set()
for k in range(K - 1):
    path = all_paths[k]
    current_q = k
    for p in range(len(path) - 1):
        pos = path[p]
        next_pos = path[p+1]
        
        c = color_map[pos]
        d = get_direction(pos, next_pos)
        A = c 
        
        if next_pos == TARGETS[k+1]:
            S = k + 1
        else:
            S = k
            
        rules.add((c, current_q, A, S, d))
M = len(rules)

# --- 7. 出力 (変更なし) ---
print(C, Q, M)
for r in range(N):
    print(*(initial_board[r]))
for rule in sorted(list(rules)):
    print(*rule)