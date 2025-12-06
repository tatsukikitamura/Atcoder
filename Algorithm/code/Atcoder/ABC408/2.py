a = int(input())

b = list(map(int,input().split()))

s = set(b)

ans =sorted(s)

print(len(ans))
for i in ans:
    print(i)