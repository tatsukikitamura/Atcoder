N, M = map(int, input().split())
shurui = []
size = []
for _ in range(N):
    A, B = map(int, input().split())

    shurui.append(A)
    size.append(B)

for x in range(1,M+1):
    use = []
    for y in range(N):
        if x == shurui[y]:
            use.append(size[y])
    print(sum(use)/len(use))