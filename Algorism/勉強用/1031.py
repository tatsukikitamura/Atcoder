N,H,W = map(int,input().split())
sy,sx = map(int,input().split())
li = []
ans = []
count = 0
s = str(input())
for x in range(H):
    li.append(list(map(int,input().split())))


for x in range(N):
    if s[x] == "F":
        sy -= 1
        ans.append(li[sy-1][sx-1])
    elif s[x] == "B":
        sy += 1
        ans.append(li[sy-1][sx-1])
    elif s[x] == "L":
        sx -= 1
        ans.append(li[sy-1][sx-1])
    elif s[x] == "R":
        sx += 1
        ans.append(li[sy-1][sx-1])


for a in ans:
    print(a)