N = int(input())
LIST = []
for _ in range(N):
    a = input()
    b = input()
    l = [a,b]
    LIST.append(l)


for x in range(N):
    if set(LIST[x][0]) == set(LIST[x][1]):
        print("Yes")
    else:
        print("No")
        