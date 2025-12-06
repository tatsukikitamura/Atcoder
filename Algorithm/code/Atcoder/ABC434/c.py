
T = int(input())

for x in range(T):
    N, H = map(int, input().split())
    time = []
    t_l_u = []
    answer = True
    for _ in range(N):
        t, l, u = map(int, input().split())
        t_l_u.append([t, l, u])
    list_range = [H-t_l_u[0][0], H+t_l_u[0][0]]
    #print(list_range)
    #print(t_l_u)

    for x in range(N):
        if list_range[0] > t_l_u[x][2] or list_range[1] < t_l_u[x][1]:
            answer = False
            break
        else:
            list_range[0] = max(list_range[0], t_l_u[x][1], 0)
            list_range[1] = min(list_range[1], t_l_u[x][2])
            
        if list_range[0] < 0:
            list_range[0] = 0

        #print(list_range)
        if x != N-1:   
            list_range[0] -= t_l_u[x+1][0] - t_l_u[x][0]
            list_range[1] += t_l_u[x+1][0] - t_l_u[x][0]
    if answer == True:
        print("Yes")
    else:
        print("No")
