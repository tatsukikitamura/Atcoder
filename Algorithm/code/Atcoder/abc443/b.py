N,K = map(int,input().split())


count = 0
N_sum = 0
while True:
    N_sum +=  N
    if N_sum >= K:
        break
    else:
        N  += 1
        count += 1

print(count)