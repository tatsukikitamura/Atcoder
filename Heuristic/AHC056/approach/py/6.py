import sys
from collections import deque, defaultdict
import heapq

# --- 1. 入力読み込み (変更なし) ---
N, K, T = map(int, sys.stdin.readline().split())
V_WALLS = [sys.stdin.readline().strip() for _ in range(N)]
H_WALLS = [sys.stdin.readline().strip() for _ in range(N - 1)]
TARGETS = []
for _ in range(K):
    TARGETS.append(tuple(map(int, sys.stdin.readline().split())))

# --- 2. ヘルパー関数 (変更なし) ---

def can_move(i, j, d):
    # (変更なし)
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
    # (変更なし)
    if d == 'U': return (i - 1, j)
    if d == 'D': return (i + 1, j)
    if d == 'L': return (i, j - 1)
    if d == 'R': return (i, j + 1)
    return (i, j)

def get_direction(pos1, pos2):
    # (変更なし)
    r1, c1 = pos1
    r2, c2 = pos2
    if r2 < r1: return 'U'
    if r2 > r1: return 'D'
    if c2 < c1: return 'L'
    if c2 > c1: return 'R'
    return 'S'

# --- 3. BFS (最短経路「パス」と「長さ」を返す) ---
# (変更なし)
def bfs_path_and_length(start_pos, goal_pos):
    q = deque([(start_pos, [start_pos])]) # (現在地, そこまでのパス)
    visited = {start_pos}

    while q:
        (curr_i, curr_j), path = q.popleft()

        if (curr_i, curr_j) == goal_pos:
            return (len(path) - 1, path) # (長さ, パスリスト)

        for d in ['U', 'D', 'L', 'R']:
            if can_move(curr_i, curr_j, d):
                next_pos = get_next_pos(curr_i, curr_j, d)
                if next_pos not in visited:
                    visited.add(next_pos)
                    new_path = path + [next_pos]
                    q.append((next_pos, new_path))
    
    return (float('inf'), None) # 到達不能

# --- 3.6. パレート最適ダイクストラ (変更なし) ---
def find_path_dijkstra(start_pos, goal_pos, step_limit, total_path_cells):
    # (前回の実装から変更なし)
    pq = [(0, 0, start_pos, [start_pos])] 
    dist = {} 
    dist[start_pos] = {0: 0}
    found_paths = []

    while pq:
        cost, steps, pos, path = heapq.heappop(pq)
        
        if dist.get(pos, {}).get(steps, float('inf')) < cost:
            continue
        if pos == goal_pos:
            heapq.heappush(found_paths, (cost, steps, path))
            continue
        if steps + 1 > step_limit:
            continue
            
        (curr_i, curr_j) = pos
        for d in ['U', 'D', 'L', 'R']:
            if can_move(curr_i, curr_j, d):
                next_pos = get_next_pos(curr_i, curr_j, d)
                next_steps = steps + 1
                
                new_cost = cost
                if next_pos not in total_path_cells and next_pos not in path:
                    new_cost += 1
                
                is_dominated = False
                if next_pos in dist:
                    for existing_steps, existing_cost in dist[next_pos].items():
                        if existing_steps <= next_steps and existing_cost <= new_cost:
                            is_dominated = True
                            break
                if is_dominated:
                    continue
                
                if next_pos not in dist:
                    dist[next_pos] = {}
                else:
                    dominated_steps = []
                    for existing_steps, existing_cost in dist[next_pos].items():
                        if existing_steps >= next_steps and existing_cost >= new_cost:
                            dominated_steps.append(existing_steps)
                    for s in dominated_steps:
                        if s in dist[next_pos]:
                           del dist[next_pos][s]

                dist[next_pos][next_steps] = new_cost
                new_path = path + [next_pos]
                heapq.heappush(pq, (new_cost, next_steps, next_pos, new_path))
    
    if found_paths:
        best_cost, best_steps, best_path = heapq.heappop(found_paths)
        return best_path
    
    return None

# --- 4. メインロジック (経路決定) ---
# ### CHANGED ###: 戦略3 (最長経路優先) に基づいて書き換え

# 4.1. 全区間の最短経路情報を収集
segments_info = []
total_shortest_steps_X = 0

for k in range(K - 1):
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    
    X_k, path_k = bfs_path_and_length(start_node, goal_node)
    
    if path_k is None: # 到達不能 (ありえないが安全策)
        X_k = float('inf')
        path_k = []
        
    if X_k != float('inf'):
        total_shortest_steps_X += X_k

    segments_info.append({
        'k': k,                 # 区間番号
        'X_k': X_k,             # 最短経路長
        'path_k': path_k,       # 最短経路リスト (フォールバック用)
    })

# 4.2. 最短経路長 (X_k) が「長い」順 (降順) にソート
# ### CHANGED ###: ソートキーの変更
sorted_segments = sorted(segments_info, key=lambda x: x['X_k'], reverse=True)

# 4.3. ソートされた順に経路を決定 (このブロックは戦略1, 2と変更なし)
all_paths = [None] * (K - 1) # 最終的なパス (k=0..K-2 の順)
path_cells = {TARGETS[0]}
total_steps_used = 0
X_future = total_shortest_steps_X # これからルーティングする経路の最短長の合計

for segment in sorted_segments:
    k = segment['k']
    X_k = segment['X_k']
    path_k_fallback = segment['path_k']
    
    # X_k が inf の場合 (到達不能)、スキップ
    if X_k == float('inf'):
        continue
        
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]

    # この区間で使えるステップ数の上限を計算
    steps_allowed_total = T - total_steps_used
    margin_for_future = X_future - X_k # この区間以外が最短経路を使った場合
    step_limit = steps_allowed_total - margin_for_future
    
    if step_limit < X_k:
        step_limit = X_k 

    # ダイクストラ法で経路探索
    path = find_path_dijkstra(start_node, goal_node, step_limit, path_cells)

    # 安全策 (ダイクストラ失敗時)
    if path is None:
        path = path_k_fallback
    
    all_paths[k] = path # 正しいインデックス k に格納
    path_cells.update(path)
    
    total_steps_used += (len(path) - 1)
    X_future -= X_k # この区間のルーティングが完了した


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
    if path is None:
        continue 
        
    current_q = k
    for p in range(len(path) - 1):
        pos = path[p]
        next_pos = path[p+1]
        
        c = color_map.get(pos, 0) 
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