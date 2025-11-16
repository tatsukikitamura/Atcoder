N,M = map(int,input().split())
NLI= list(map(int,input().split()))
MLI= list(map(int,input().split()))

for x in range(M):
    B = MLI[x]
    if B in NLI:
        NLI.pop(NLI.index(B))
    else:
        continue

for item in NLI:
    print(item, end=" ")  