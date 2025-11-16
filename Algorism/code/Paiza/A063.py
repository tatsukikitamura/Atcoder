K = int(input())
LIST = []
for _ in range(K):
    M,X = map(int,input().split())
    for _ in range(M):
        LIST.append(X)

NUM = len(LIST) // 2
count = 0

for x in range(NUM):
    count += abs(LIST[x] - LIST[x+NUM])

print(count)