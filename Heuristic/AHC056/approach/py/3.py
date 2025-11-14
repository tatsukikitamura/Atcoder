import sys
from collections import deque, defaultdict # ### NEW ###
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

# --- 3. BFS (変更なし) ---
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
# ### CHANGED ###: パレート最適化による枝刈りを導入
def find_path_dijkstra(start_pos, goal_pos, step_limit, total_path_cells):
    # pq: (cost, steps, pos, path_list)
    pq = [(0, 0, start_pos, [start_pos])] 
    
    # dist[pos] = {steps: min_cost}
    # (pos) に (steps) で到達したときの最小 (cost) を記録
    dist = {} 
    dist[start_pos] = {0: 0} # {steps: cost}
    
    found_paths = []

    while pq:
        cost, steps, pos, path = heapq.heappop(pq)
        
        # 既にこの (pos, steps) に、より低いコストで到達済みならスキップ
        if dist.get(pos, {}).get(steps, float('inf')) < cost:
            continue
            
        if pos == goal_pos:
            heapq.heappush(found_paths, (cost, steps, path))
            continue
            
        # ステップ数上限チェック (移動 *前* にチェック)
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
                
                # ### パレート最適チェック (枝刈り) ###
                # (next_pos) に、(new_cost, new_steps) より優れた経路が
                # (既存の経路に) 存在するかチェック
                is_dominated = False
                if next_pos in dist:
                    for existing_steps, existing_cost in dist[next_pos].items():
                        if existing_steps <= next_steps and existing_cost <= new_cost:
                            # 既存の経路の方が優れている (または等しい)
                            is_dominated = True
                            break
                
                if is_dominated:
                    continue # この (new_cost, new_steps) 経路は枝刈り
                
                # ### 既存経路の刈り取り (Domination) ###
                # (new_cost, new_steps) が既存の経路より優れている場合、
                # 既存の劣後した経路を削除する
                if next_pos not in dist:
                    dist[next_pos] = {}
                else:
                    dominated_steps = []
                    for existing_steps, existing_cost in dist[next_pos].items():
                        if existing_steps >= next_steps and existing_cost >= new_cost:
                            # 既存の経路 (existing) は劣後した
                            dominated_steps.append(existing_steps)
                    
                    for s in dominated_steps:
                        # (new_steps, new_cost) と全く同じ場合も削除される
                        if s in dist[next_pos]:
                           del dist[next_pos][s]

                # dist を更新し、pq に追加
                dist[next_pos][next_steps] = new_cost
                new_path = path + [next_pos]
                heapq.heappush(pq, (new_cost, next_steps, next_pos, new_path))
    
    # ゴールに到達した経路のうち、(cost, steps) の辞書順で最小のものを返す
    if found_paths:
        best_cost, best_steps, best_path = heapq.heappop(found_paths)
        return best_path
    
    # 万が一 (step_limit が厳しすぎる場合など) 
    return None # フォールバックロジック (4.2.) で処理される

# --- 4. 全経路の計算 & 経路マスの特定 ---
# (4.1. は変更なし)
# ...

# 4.1. まず全区間の最短経路長 (X_k) と総最短ステップ (X) を計算
shortest_lengths = []
total_shortest_steps_X = 0
for k in range(K - 1):
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    length = bfs_shortest_length(start_node, goal_node)
    shortest_lengths.append(length)
    total_shortest_steps_X += length

# 4.2. ダイクストラ法で経路を決定 (変更なし、呼び出し先が変わっただけ)
all_paths = []
path_cells = {TARGETS[0]} 
total_steps_used = 0
X_future = total_shortest_steps_X 

for k in range(K - 1):
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    X_k = shortest_lengths[k]
    
    steps_allowed_total = T - total_steps_used
    margin_for_future = X_future - X_k
    step_limit = steps_allowed_total - margin_for_future

    # 修正された find_path_dijkstra を呼び出す
    path = find_path_dijkstra(start_node, goal_node, step_limit, path_cells)

    # 安全策 (変更なし)
    if path is None:
        # (BFSによるフォールバック)
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
    
    all_paths.append(path)
    path_cells.update(path) 
    
    total_steps_used += (len(path) - 1)
    X_future -= X_k

# --- 5, 6, 7 (変更なし) ---
# ... (色割り当て、ルール生成、出力)
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

print(C, Q, M)
for r in range(N):
    print(*(initial_board[r]))
for rule in sorted(list(rules)):
    print(*rule)