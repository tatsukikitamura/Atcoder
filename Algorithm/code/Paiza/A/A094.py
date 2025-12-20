import math
import heapq

N,M = map(int,input().split())
route_list = [[0] * N for i in range(N)]

print(route_list)
for _ in range(M):
    u,v,c = map(int,input().split())
    route_list[u-1][v-1] = c


node_num = N

unsearched_nodes = list(range(node_num)) 
distance = [math.inf] * node_num 
previous_nodes = [-1] * node_num 
distance[0] = 0 # 初期のノードの距離は0とする

def get_target_min_index(min_index, distance, unsearched_nodes):
    start = 0
    while True:
        index = distance.index(min_index, start)
        found = index in unsearched_nodes
        if found:
            return index
        else:
            start = index + 1

while(len(unsearched_nodes) != 0): 

    posible_min_distance = math.inf #
    for node_index in unsearched_nodes: 
        if posible_min_distance > distance[node_index]: 
            posible_min_distance = distance[node_index] # より小さい値が見つかれば更新
    target_min_index = get_target_min_index(posible_min_distance, distance, unsearched_nodes) # 未探索ノードのうちで最小のindex番号を取得
    unsearched_nodes.remove(target_min_index) # ここで探索するので未探索リストから除去

    target_edge = route_list[target_min_index] # ターゲットになるノードからのびるエッジのリスト
    for index, route_dis in enumerate(target_edge):
        if route_dis != 0:
            if distance[index] > (distance[ target_min_index] + route_dis):
                distance[index] = distance[ target_min_index] + route_dis # 過去に設定されたdistanceよりも小さい場合はdistanceを更新
                previous_nodes[index] =  target_min_index #　ひとつ前に到達するノードのリストも更新

# 以下で結果の表示



print("-----経路-----")
previous_node = node_num - 1
while previous_node != -1:
    if previous_node !=0:
        print(str(previous_node + 1) + " <- ", end='')
    else:
        print(str(previous_node + 1))
    previous_node = previous_nodes[previous_node]

print("-----距離-----")
print(distance)
