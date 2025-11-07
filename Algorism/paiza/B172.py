N = int(input())
LIST = list(map(int,input().split()))
answer = []
USE = []

for x in range(N):
    USE.append(LIST[x])
    a = []
    for y in range(x,N):
        a.append(LIST[y])
    USE.append(a)

print(USE)


def check(x,y):
    pass


for x in range(len(USE)):
    for y in range(x,N):
        answer += check(x,y)



print(answer)
print(sum(answer) / len(answer))