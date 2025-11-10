import sys
from collections import deque, defaultdict
import heapq
import time # TLE 対策用

# --- 1. 入力読み込み (変更なし) ---
N, K, T = map(int, sys.stdin.readline().split())
V_WALLS = [sys.stdin.readline().strip() for _ in range(N)]
H_WALLS = [sys.stdin.readline().strip() for _ in range(N - 1)]
TARGETS = []
for _ in range(K):
    TARGETS.append(tuple(map(int, sys.stdin.readline().split())))

# --- ビームサーチ設定 ---
BEAM_WIDTH =4 # 保持する解候補の数
N_PATHS_PER_STATE = 3 # 1つの状態から分岐するパスの数

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

# --- 3. BFS (戦略1で必要なもの) ---
def bfs_path_and_length(start_pos, goal_pos):
    # (変更なし)
    q = deque([(start_pos, [start_pos])]) 
    visited = {start_pos}
    while q:
        (curr_i, curr_j), path = q.popleft()
        if (curr_i, curr_j) == goal_pos:
            return (len(path) - 1, path)
        for d in ['U', 'D', 'L', 'R']:
            if can_move(curr_i, curr_j, d):
                next_pos = get_next_pos(curr_i, curr_j, d)
                if next_pos not in visited:
                    visited.add(next_pos)
                    new_path = path + [next_pos]
                    q.append((next_pos, new_path))
    return (float('inf'), None)

def bfs_second_path_length(start_pos, goal_pos, forbidden_edges):
    # (変更なし / 戦略1の自由度計算用)
    q = deque([(start_pos, 0)])
    visited = {start_pos}
    while q:
        (curr_i, curr_j), steps = q.popleft()
        if (curr_i, curr_j) == goal_pos:
            return steps
        for d in ['U', 'D', 'L', 'R']:
            if can_move(curr_i, curr_j, d):
                next_pos = get_next_pos(curr_i, curr_j, d)
                edge = ((curr_i, curr_j), next_pos)
                if edge in forbidden_edges:
                    continue
                if next_pos not in visited:
                    visited.add(next_pos)
                    q.append((next_pos, steps + 1))
    return float('inf')

# --- 3.6. パレート最適ダイクストラ (ビームサーチ用に改造) ---
# ### CHANGED ###: 複数の有望なパスを返すように変更
def find_path_dijkstra_beam(start_pos, goal_pos, step_limit, total_path_cells):
    # 戻り値: [(cost, steps, path), (cost, steps, path), ...]
    pq = [(0, 0, start_pos, [start_pos])] 
    dist = {} 
    dist[start_pos] = {0: 0}
    found_paths = [] # (cost, steps, path)

    while pq:
        cost, steps, pos, path = heapq.heappop(pq)
        
        if dist.get(pos, {}).get(steps, float('inf')) < cost:
            continue
        if pos == goal_pos:
            # ゴールに到達したものをすべて記録する
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
    
    # 1つもパスが見つからなかった場合
    if not found_paths:
        return None
        
    # ### CHANGED ###
    # 見つかったパス (found_paths) から、パレート最適なものを
    # N_PATHS_PER_STATE 個まで返す (簡易版として nsmallest を使う)
    return heapq.nsmallest(N_PATHS_PER_STATE, found_paths)

# --- 4. メインロジック (ビームサーチ) ---
# ### CHANGED ###: ビームサーチのロジックに全面書き換え

# 4.0. ビームの状態を管理するクラス
class BeamState:
    def __init__(self, paths_list, path_cells, total_steps, total_C):
        self.paths_list = paths_list # (K-1)長のリスト
        self.path_cells = path_cells # set
        self.total_steps = total_steps
        self.total_C = total_C

    # ビームのソート用 (C優先、次にSteps)
    def __lt__(self, other):
        if self.total_C != other.total_C:
            return self.total_C < other.total_C
        return self.total_steps < other.total_steps

# 4.1. 全区間の自由度を計算 (戦略1と同じ)
segments_info = []
total_shortest_steps_X = 0
shortest_lengths = [float('inf')] * (K - 1)
all_shortest_paths_fallback = [None] * (K - 1)

for k in range(K - 1):
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    
    X_k, path_k = bfs_path_and_length(start_node, goal_node)
    all_shortest_paths_fallback[k] = path_k
    shortest_lengths[k] = X_k
    
    if path_k is None:
        X_k = float('inf')
        freedom_score = float('inf')
    else:
        total_shortest_steps_X += X_k
        forbidden_edges = set()
        for i in range(len(path_k) - 1):
            forbidden_edges.add((path_k[i], path_k[i+1]))
        X_prime_k = bfs_second_path_length(start_node, goal_node, forbidden_edges)
        freedom_score = X_prime_k - X_k

    segments_info.append({
        'k': k, 'X_k': X_k, 'path_k': path_k, 'freedom': freedom_score
    })

# 4.2. 自由度が「低い」順にソート (戦略1と同じ)
sorted_segments = sorted(segments_info, key=lambda x: x['freedom'])

# 4.3. ビームサーチの実行
current_beam = []
# 初期状態 (何もルーティングしていない)
initial_paths = [None] * (K - 1)
initial_cells = {TARGETS[0]}
current_beam.append(BeamState(initial_paths, initial_cells, 0, len(initial_cells)))

X_future = total_shortest_steps_X

for segment in sorted_segments:
    k = segment['k']
    X_k = segment['X_k']
    path_k_fallback = segment['path_k']
    
    if X_k == float('inf'):
        continue
        
    start_node = TARGETS[k]
    goal_node = TARGETS[k+1]
    
    next_beam = [] # 次のステップの候補

    # 現在のビーム (W個の状態) をすべて展開
    for state in current_beam:
        
        # この state における step_limit を計算
        steps_allowed_total = T - state.total_steps
        margin_for_future = X_future - X_k
        step_limit = steps_allowed_total - margin_for_future
        if step_limit < X_k:
            step_limit = X_k
            
        # ダイクストラで N_PATHS_PER_STATE 個の候補パスを取得
        candidate_paths_info = find_path_dijkstra_beam(
            start_node, 
            goal_node, 
            step_limit, 
            state.path_cells # この state の path_cells を使う
        )
        
        # 安全策: もしダイクストラが失敗したら、最短経路でフォールバック
        if candidate_paths_info is None:
            # (cost, steps, path)
            cost = 0
            for cell in path_k_fallback:
                if cell not in state.path_cells:
                    cost += 1
            candidate_paths_info = [(cost, X_k, path_k_fallback)]
            
        # 展開: N_PATHS_PER_STATE 個のパスから新しい状態を生成
        for (cost, steps, path) in candidate_paths_info:
            
            # 既存のパスリストをコピーして更新
            new_paths_list = list(state.paths_list)
            new_paths_list[k] = path
            
            # 既存の path_cells をコピーして更新
            new_path_cells = set(state.path_cells)
            new_path_cells.update(path)
            
            new_total_steps = state.total_steps + steps
            new_total_C = len(new_path_cells)+ 1# Cは path_cells のサイズ
            
            new_state = BeamState(new_paths_list, new_path_cells, new_total_steps, new_total_C)
            next_beam.append(new_state)

    # 4.4. 枝刈り (Pruning)
    # BEAM_WIDTH * N_PATHS_PER_STATE 個の候補をソート
    next_beam.sort()
    
    # 上位 BEAM_WIDTH 個だけを残す
    current_beam = next_beam[:BEAM_WIDTH]
    
    # 処理した X_k を未来から除く
    X_future -= X_k

# --- 5. 最終解の選択 ---
# k=K-2 まで終わった時点で、current_beam には W 個の解候補が残っている
# その中で最も C が小さいものを選ぶ (ソート済みなので先頭)
best_solution = current_beam[0]

all_paths = best_solution.paths_list
path_cells = best_solution.path_cells
C = best_solution.total_C
Q = K

# --- 6. 色の割り当てと遷移規則の生成 (変更なし) ---
color_map = {} 
current_color = 1
for cell in path_cells:
    if cell not in color_map:
        color_map[cell] = current_color
        current_color += 1
# C = current_color # best_solution.total_C を使う

initial_board = [[0] * N for _ in range(N)]
for (r, c), color in color_map.items():
    initial_board[r][c] = color

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