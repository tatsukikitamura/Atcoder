N = int(input())
sum_grid = 2000**2
grid = [[0]*10 for _ in range(10)]
count = 0
for i in range(N):
    U, D, L, R = map(int, input().split())
    for x in range(U-1, D):
        for y in range(L-1, R):
            if grid[x][y] == 0:
                grid[x][y] = 1
                count += 1
            else:
                grid[x][y] += 1
print(grid)
sum_grid -= count

print(sum_grid)