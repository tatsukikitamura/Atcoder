N,M,L,S,T = map(int,input().split())
graph = {} #ex 1 -> 2 weight 20 
for _ in range(M):
    U,V,C = map(int,input().split())
    if U in graph:
        graph[U].append((V,C))
    else:
        graph[U] = [(V,C)]
print(graph)
visited = set()
stack = [1]

while stack:
    break