X = int(input())
N = int(input())
LIST = list(map(int,input().split()))
Q = int(input())
use = []
for _ in range(Q):
    use.append(int(input())-1)
use2 = []
for y in use:
    if y in use2:
        X -= LIST[y]
        use2.remove(y)
    else:
        X += LIST[y]
        use2.append(y)
    print(X)
