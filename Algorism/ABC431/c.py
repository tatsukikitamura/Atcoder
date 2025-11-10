N,M,K = map(int,input().split())
H_N = sorted(list(map(int,input().split())))
B_M = sorted(list(map(int,input().split())))

def check_H_N(H_N,B_M):
    try:
        y = 0
        count = 0
        for x in range(max(N,M)):
            while True:
                if H_N[x] <= B_M[y]:
                    count += 1
                    y += 1
                    break
                else:
                    y += 1
    except:
        return count
    return count



if check_H_N(H_N,B_M) >= K:

    print("Yes")
else:
    print("No")




