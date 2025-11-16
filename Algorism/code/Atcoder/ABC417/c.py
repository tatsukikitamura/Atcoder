N = int(input())
NLI= list(map(int,input().split()))
count =0
for x in range(N):
    for y in range(x+1,N):
        if (y-x) == NLI[x]+NLI[y]:
            count += 1
        
print(count)
            