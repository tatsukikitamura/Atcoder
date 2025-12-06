N,X,Y = map(int,input().split())

for x in range(1,N+1):
    if x % X == 0 and x % Y == 0:
        print("AB")
    elif x % X == 0:
        print("A")
    elif x % Y == 0:
        print("B")
    else:
        print("N")