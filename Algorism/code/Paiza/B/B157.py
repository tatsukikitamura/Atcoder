N,K = map(int,input().split())
LIST = []
for _ in range(N):
    LIST.append(list(map(int,input().split())))

ans = []

for x in range(K):
    use = []
    for y in range(N):
        use.append(LIST[y][x])

    min_use = min(use)
    ans.append(use.index(min_use)+1)

set = set(tuple(ans))
print(len(set))