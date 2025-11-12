import sys
from collections import deque

# --- 定数定義 ---
# C=5 (0:U, 1:D, 2:L, 3:R, 4:Target)
C = 5
COLOR_TO_DIR = {0: 'U', 1: 'D', 2: 'L', 3: 'R', 4: '-'}
DIR_TO_COLOR = {'U': 0, 'D': 1, 'L': 2, 'R': 3, '-': 4}
MOVES = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1), '-': (0, 0)}
INV_MOVES = {(-1, 0): 'U', (1, 0): 'D', (0, -1): 'L', (0, 1): 'R', (0, 0): '-'}

# --- 入力読み込み ---
input = sys.stdin.readline
N, K, T = map(int, input().split())
walls_v = [list(map(int, list(input().strip()))) for _ in range(N)]
walls_h = [list(map(int, list(input().strip()))) for _ in range(N - 1)]
targets = []
for _ in range(K):
    targets.append(list(map(int, input().split())))

# --- 1. 事前計算：指示マップの作成 ---
def bfs_calc_shortest_paths(goal_r, goal_c):
    dist = [[-1] * N for _ in range(N)]
    move_dir = [['-'] * N for _ in range(N)]
    q = deque()
    q.append((goal_r, goal_c))
    dist[goal_r][goal_c] = 0
    
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N): continue
            if dist[nr][nc] != -1: continue
            is_wall = False
            if dr == -1: 
                if walls_h[nr][nc] == 1: is_wall = True
            elif dr == 1:
                if walls_h[r][c] == 1: is_wall = True
            elif dc == -1:
                if walls_v[nr][nc] == 1: is_wall = True
            elif dc == 1:
                if walls_v[r][c] == 1: is_wall = True
            if not is_wall:
                dist[nr][nc] = dist[r][c] + 1
                move_dir[nr][nc] = INV_MOVES[(-dr, -dc)] 
                q.append((nr, nc))
    return move_dir

# instruction_maps[k][r][c]: 目的地 k への指示(色 0-3)
instruction_maps = []
for k in range(K): # k=0 から K-1 まで (K個)
    goal_r, goal_c = targets[k] # t[k] への地図を作る
    move_dir_map = bfs_calc_shortest_paths(goal_r, goal_c)
    color_map = [[DIR_TO_COLOR[move_dir_map[r][c]] for c in range(N)] for r in range(N)]
    instruction_maps.append(color_map)

# instruction_maps[K] (q=K-1 が書き込む用)
instruction_maps.append([[4] * N for _ in range(N)]) # 全部 4 (目的地色)


# --- 2. 初期盤面の設定 ---
# 最初の指示 (map[0] = t[0] への指示) を設定
initial_board = [row[:] for row in instruction_maps[0]]

# 目的地 0 から K-1 を「目的地」色(4)で上書き
for k in range(K): # k=0 (スタート地点) も 4 にする
    r, c = targets[k]
    initial_board[r][c] = 4

# --- 3. シミュレーション & 遷移規則の生成 (ロジック修正) ---
Q = K
transitions = {}
current_board = [row[:] for row in initial_board]
robot_r, robot_c = targets[0]
robot_q = 0

for step in range(T):
    if robot_q == K:
        break
        
    c = current_board[robot_r][robot_c]
    q = robot_q
    
    if (c, q) not in transitions:
        A = -1 
        S = -1
        D = '-'
        
        is_at_correct_target = (robot_r == targets[q][0] and robot_c == targets[q][1])

        # ★★★ 3分岐ロジックを復活 ★★★
        if is_at_correct_target:
            # 1. 正しい目的地 (targets[q]) にいる場合
            S = q + 1
            D = '-' # 1ステップ停止
            
            # 次の経路 (map[q+1]) の指示を書き込む
            A = instruction_maps[S][robot_r][robot_c] # (S=K の時は map[K] = 4)
        
        elif c == 4:
            # 2. 間違った目的地を踏んだ場合 (is_at_correct_target=False)
            S = q
            
            # 本来の指示 (map[q]) に従って経路に戻る
            A = instruction_maps[q+1][robot_r][robot_c] # 次のmap[q+1]を書き込む
            D = COLOR_TO_DIR[instruction_maps[q][robot_r][robot_c]] # map[q]に従って動く

        else:
            # 3. 経路上 (c = 0, 1, 2, 3) の場合
            S = q # 状態は維持
            
            # 本来の指示 (map[q]) に従って動く
            D = COLOR_TO_DIR[instruction_maps[q][robot_r][robot_c]] 
            
            # 次の経路 (map[q+1]) の指示を書き込む
            A = instruction_maps[q+1][robot_r][robot_c]
        # ★★★ 修正ここまで ★★★

        transitions[(c, q)] = (A, S, D)
    
    # --- ロボットの動作シミュレーション ---
    A, S, D = transitions[(c, q)]
    
    current_board[robot_r][robot_c] = A
    robot_q = S
    
    dr, dc = MOVES[D]
    nr, nc = robot_r + dr, robot_c + dc
    
    if 0 <= nr < N and 0 <= nc < N:
        is_wall = False
        if dr == -1: 
            if walls_h[nr][nc] == 1: is_wall = True
        elif dr == 1:
            if walls_h[robot_r][robot_c] == 1: is_wall = True
        elif dc == -1:
            if walls_v[nr][nc] == 1: is_wall = True
        elif dc == 1:
            if walls_v[robot_r][robot_c] == 1: is_wall = True
        
        if not is_wall:
            robot_r, robot_c = nr, nc

# --- 4. 出力 ---
M = len(transitions)
print(f"{C} {Q} {M}")

for row in initial_board:
    print("".join(map(str, row)))

for (c, q), (A, S, D) in transitions.items():
    print(f"{c} {q} {A} {S} {D}")