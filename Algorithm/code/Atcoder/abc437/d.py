import bisect
import itertools
N,M =map(int,input().split())
A = list(map(int,input().split()))
B = list(map(int,input().split()))

A.sort()
B.sort()

all_list = [0] + list(itertools.accumulate(A))
#print(all_list)
#print(B)

sosuu = 998244353
ans = 0
for x in B:
    y = bisect.bisect_left(A,x)
    ans += -2 * all_list[y] + all_list[N]  + (2*y-N) * x 
    #print(ans)
    #print(y)

print(ans % sosuu)