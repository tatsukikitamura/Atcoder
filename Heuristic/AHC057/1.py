import sys
import random

# 定数定義
L = 100000
HALF_L = 50000

def solve():
    # 入力読み込み
    input_data = sys.stdin.read().split()
    if not input_data: return
    iterator = iter(input_data)
    
    try:
        N = int(next(iterator))
        T = int(next(iterator))
        M = int(next(iterator))
        K = int(next(iterator))
        global L, HALF_L
        val = int(next(iterator))
        L = val
        HALF_L = L // 2
        
        x = [0] * N
        y = [0] * N
        vx = [0] * N
        vy = [0] * N
        
        for i in range(N):
            x[i] = int(next(iterator))
            y[i] = int(next(iterator))
            vx[i] = int(next(iterator))
            vy[i] = int(next(iterator))
    except StopIteration:
        return

    # --- ヒルベルト曲線関数 ---
    def get_hilbert_d(tx, ty):
        d = 0
        n = 1 << 17
        s = n // 2
        while s > 0:
            rx = 1 if (tx & s) > 0 else 0
            ry = 1 if (ty & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    tx = n - 1 - tx
                    ty = n - 1 - ty
                tx, ty = ty, tx
            s //= 2
        return d

    # --- 1. t=1000 での予測位置計算 ---
    target_time = T
    pred_x = [0] * N
    pred_y = [0] * N
    for i in range(N):
        pred_x[i] = (x[i] + vx[i] * target_time) % L
        pred_y[i] = (y[i] + vy[i] * target_time) % L

    # --- 2. 初期グループ分け (ヒルベルト) ---
    points_with_idx = []
    for i in range(N):
        h_d = get_hilbert_d(pred_x[i], pred_y[i])
        points_with_idx.append((h_d, i))
    
    points_with_idx.sort()
    
    target_group = [-1] * N
    for i in range(N):
        original_idx = points_with_idx[i][1]
        group_id = i // K
        if group_id >= M: group_id = M - 1
        target_group[original_idx] = group_id

    # --- 3. Swap最適化 (ここが追加点) ---
    # グループ間のメンバーを交換して、各グループの「凝集度」を高める
    # 制限時間ギリギリまで回すと良いが、ここでは回数指定で行う
    
    # 評価関数: グループ内の全点間の距離の二乗和 (小さいほど良い)
    # 高速化のため、「重心との距離」で近似しても良いが、K=30なら総当たりでもいけるか？
    # 重心方式を採用して高速化する。
    
    # グループごとの重心計算用データ
    # トーラス上の重心は難しいが、密集している前提で単純平均をとるか、
    # 厳密には各点ペアの距離和を見るのが安全。
    
    # K=30なので、ペアの距離和の計算コストは 30*29/2 = 435回。
    # これを毎回計算するのは重い。
    # よって、「ランダムに2点選んで、交換して良くなれば採用」を繰り返す山登り法を行う。

    # 事前に各グループのメンバーリストを作成
    group_members = [[] for _ in range(M)]
    for i in range(N):
        group_members[target_group[i]].append(i)

    # 距離計算ヘルパー
    def calc_group_cost(gid):
        members = group_members[gid]
        cost = 0
        for i in range(len(members)):
            p1 = members[i]
            px1, py1 = pred_x[p1], pred_y[p1]
            for j in range(i+1, len(members)):
                p2 = members[j]
                px2, py2 = pred_x[p2], pred_y[p2]
                dx = abs(px1 - px2)
                dy = abs(py1 - py2)
                if dx > HALF_L: dx = L - dx
                if dy > HALF_L: dy = L - dy
                cost += dx*dx + dy*dy
        return cost

    # 初期コスト計算
    current_costs = [calc_group_cost(g) for g in range(M)]
    
    # 最適化ループ (回数は調整可能。多いほど良い)
    # 時間制限に注意。Pythonなら数万回程度か。
    iterations = 15000 
    
    for _ in range(iterations):
        # 異なるグループg1, g2を選ぶ
        g1 = random.randint(0, M-1)
        g2 = random.randint(0, M-1)
        if g1 == g2: continue
        
        # それぞれからランダムに1点選ぶ
        idx1_in_group = random.randint(0, K-1)
        idx2_in_group = random.randint(0, K-1)
        
        p1 = group_members[g1][idx1_in_group]
        p2 = group_members[g2][idx2_in_group]
        
        # 現在のコスト
        cost1_before = current_costs[g1]
        cost2_before = current_costs[g2]
        
        # 仮に交換してみる
        # p1をg2へ、p2をg1へ
        group_members[g1][idx1_in_group] = p2
        group_members[g2][idx2_in_group] = p1
        
        # 新コスト計算
        cost1_after = calc_group_cost(g1)
        cost2_after = calc_group_cost(g2)
        
        # 改善判定 (合計コストが減るか？)
        if (cost1_after + cost2_after) < (cost1_before + cost2_before):
            # 採用 (コスト配列を更新)
            current_costs[g1] = cost1_after
            current_costs[g2] = cost2_after
            # target_group配列も更新
            target_group[p1] = g2
            target_group[p2] = g1
        else:
            # 不採用 (元に戻す)
            group_members[g1][idx1_in_group] = p1
            group_members[g2][idx2_in_group] = p2

    # --- シミュレーション準備 ---
    comp_active = [True] * N
    comp_members = [[i] for i in range(N)]
    comp_vx = list(vx)
    comp_vy = list(vy)
    
    group_cids = [set() for _ in range(M)]
    for i in range(N):
        gid = target_group[i]
        group_cids[gid].add(i)
    
    output_commands = []

    # --- シミュレーション ---
    for t in range(T):
        progress = t / T
        
        is_panic = (progress >= 0.85)
        
        if is_panic:
            p_panic = (progress - 0.85) / 0.15
            current_max = 300 + (L - 300) * p_panic
            threshold_strict = current_max * current_max
            threshold_loose = current_max * current_max
            check_direction = False
        else:
            base_strict = 10 + 30 * progress
            threshold_strict = base_strict * base_strict
            cur_loose = 50 + 250 * (progress ** 2)
            threshold_loose = cur_loose * cur_loose
            check_direction = True

        for gid in range(M):
            cids = list(group_cids[gid])
            if len(cids) < 2: continue
            
            local_merged = set()
            pairs_to_merge = []
            
            for i in range(len(cids)):
                cid1 = cids[i]
                if cid1 in local_merged: continue
                
                mem1 = comp_members[cid1]
                vx1, vy1 = comp_vx[cid1], comp_vy[cid1]
                
                best_target_cid = -1
                best_min_dist_sq = float('inf')
                best_pair_indices = (-1, -1)
                
                for j in range(i+1, len(cids)):
                    cid2 = cids[j]
                    if cid2 in local_merged: continue
                    
                    mem2 = comp_members[cid2]
                    vx2, vy2 = comp_vx[cid2], comp_vy[cid2]
                    
                    if check_direction:
                        dot = vx1 * vx2 + vy1 * vy2
                        if dot < 0 and (vx1 or vy1) and (vx2 or vy2):
                            continue
                    
                    min_d_sq_local = float('inf')
                    best_p1_local = -1
                    best_p2_local = -1
                    
                    for p1_idx in mem1:
                        px1, py1 = x[p1_idx], y[p1_idx]
                        for p2_idx in mem2:
                            px2, py2 = x[p2_idx], y[p2_idx]
                            
                            dx = abs(px1 - px2)
                            dy = abs(py1 - py2)
                            if dx > HALF_L: dx = L - dx
                            if dy > HALF_L: dy = L - dy
                            d_sq = dx*dx + dy*dy
                            
                            if d_sq < min_d_sq_local:
                                min_d_sq_local = d_sq
                                best_p1_local = p1_idx
                                best_p2_local = p2_idx
                                if d_sq == 0: break
                        if min_d_sq_local == 0: break
                    
                    # 未来予測判定
                    p1_idx, p2_idx = best_p1_local, best_p2_local
                    px1, py1 = x[p1_idx], y[p1_idx]
                    px2, py2 = x[p2_idx], y[p2_idx]
                    
                    nx1 = px1 + vx1
                    if nx1 >= L: nx1 -= L
                    elif nx1 < 0: nx1 += L
                    ny1 = py1 + vy1
                    if ny1 >= L: ny1 -= L
                    elif ny1 < 0: ny1 += L

                    nx2 = px2 + vx2
                    if nx2 >= L: nx2 -= L
                    elif nx2 < 0: nx2 += L
                    ny2 = py2 + vy2
                    if ny2 >= L: ny2 -= L
                    elif ny2 < 0: ny2 += L
                    
                    ndx = abs(nx1 - nx2)
                    ndy = abs(ny1 - ny2)
                    if ndx > HALF_L: ndx = L - ndx
                    if ndy > HALF_L: ndy = L - ndy
                    next_d_sq = ndx*ndx + ndy*ndy
                    
                    is_leaving = next_d_sq > min_d_sq_local
                    
                    can_merge = False
                    if is_panic:
                        if min_d_sq_local <= threshold_loose:
                            can_merge = True
                    else:
                        if is_leaving:
                            if min_d_sq_local <= threshold_loose:
                                can_merge = True
                        else:
                            if min_d_sq_local <= threshold_strict:
                                can_merge = True
                    
                    if can_merge:
                        if min_d_sq_local < best_min_dist_sq:
                            best_min_dist_sq = min_d_sq_local
                            best_target_cid = cid2
                            best_pair_indices = (best_p1_local, best_p2_local)
                
                if best_target_cid != -1:
                    pairs_to_merge.append((cid1, best_target_cid, best_pair_indices))
                    local_merged.add(cid1)
                    local_merged.add(best_target_cid)
            
            for cid1, cid2, (p1, p2) in pairs_to_merge:
                output_commands.append(f"{t} {p1} {p2}")
                
                s1 = len(comp_members[cid1])
                s2 = len(comp_members[cid2])
                denom = s1 + s2
                
                new_vx = (s1 * comp_vx[cid1] + s2 * comp_vx[cid2]) / denom
                new_vy = (s1 * comp_vy[cid1] + s2 * comp_vy[cid2]) / denom
                
                comp_vx[cid1] = new_vx
                comp_vy[cid1] = new_vy
                comp_members[cid1].extend(comp_members[cid2])
                
                comp_active[cid2] = False
                group_cids[gid].remove(cid2)

        # 移動フェーズ
        for cid in range(N):
            if comp_active[cid]:
                cvx_val = comp_vx[cid]
                cvy_val = comp_vy[cid]
                for midx in comp_members[cid]:
                    x[midx] = (x[midx] + cvx_val) % L
                    y[midx] = (y[midx] + cvy_val) % L

    for cmd in output_commands:
        print(cmd)

if __name__ == '__main__':
    solve()