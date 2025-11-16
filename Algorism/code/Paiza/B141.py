N = int(input())


current_list = []
for _ in range(2 * N):
    current_list.append(int(input()))


target_list = list(range(1, N + 1)) * 2

total_swaps = 0

for i in range(2 * N):
    
   
    target_color = target_list[i]
    

    j = -1 
    for k in range(i, 2 * N):
        if current_list[k] == target_color:
            j = k
            break
            

    swaps_needed = j - i
    total_swaps += swaps_needed
 
    
    element_to_move = current_list.pop(j)
    
    current_list.insert(i, element_to_move)


print(total_swaps)