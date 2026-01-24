from sortedcontainers import SortedList

N,K,X = map(int,input().split())
cups = SortedList(map(int,input().split()))

for _ in range(N-K):
    cups.pop()

ans = N-K
count = 0
while True:
    if count >= X:
        break
    elif len(cups) == 0:
        ans = -1
        break
    else:
        count += cups.pop()
        ans += 1

print(ans)
