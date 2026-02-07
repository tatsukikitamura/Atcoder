N,K = map(int,input().split())
ans = 0

for x in range(1,N+1):
    su = 0
    for y in range(len(str(x))):
        su += int(str(x)[y])
    
    if su == K:
        ans += 1
print(ans)

