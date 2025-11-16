#3/2 < q/p < 2 
#3p < 2q
#q < 2p
#q/2 < p < 2q/3

# 3p - 2q < 0
#2p - q > 0
#を満たす最小のp

#3q/8 < p < 2q/5

#5
#3 2 2 1
#5 2 8 3
#1 2 2 1
#60 191 11 35
#40 191 71 226


T = int(input())

li = []
answer = []

for i in range(T):
    li.append(list(map(int,input().split())))

def keisan(i,M):
    return ((li[i][3]*M // (li[i][2])) + 1) < (li[i][1]*M / li[i][0])

def search(j):
    ans = []
    L = 0
    R = li[j][0]*li[j][2]
    while True:
        M = (L+R) // 2 # 330
        if keisan(j,M) and (not keisan(j,M-1)):
            print(L,R,M)
            for preM in range(L,R+1):
                if keisan(j,preM) and (not keisan(j,preM)):
                    print(preM)
            return(0)
        elif keisan(j,M) and keisan(j,M-1):
            R = M 
        else:
            L = M 


for i in range(T):
    print(search(i))
