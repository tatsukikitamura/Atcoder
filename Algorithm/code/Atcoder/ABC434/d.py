N = int(input())
sum_grid = 2000**2
grid = [[0]*2000 for _ in range(2000)]
count = 0
u_d_l_r = []

for i in range(N):
    U, D, L, R = map(int, input().split())
    u_d_l_r.append([U, D, L, R])
    for x in range(U-1, D):
        for y in range(L-1, R):
            if grid[x][y] == 0:
                grid[x][y] = 1
                count += 1
            else:
                grid[x][y] = "#"
# print(grid)
sum_grid -= count

# print(sum_grid)

for i in range(N):
    U = u_d_l_r[i][0]
    D = u_d_l_r[i][1]
    L = u_d_l_r[i][2]
    R = u_d_l_r[i][3]
    count = 0
    for x in range(U-1, D):
        for y in range(L-1, R):
            if grid[x][y] == 1:
                count += 1

    print(sum_grid + count)
