H,W,N = map(int,input().split())
grid = []

for _ in range(H):
    use = set(map(int,input().split()))
    grid.append(use)

ans_set= set()
for _ in range(N):
    ans_set.add(int(input()))

ans = 0
for x in range(H):
    ans = max(ans,len(grid[x]&ans_set))


print(ans)