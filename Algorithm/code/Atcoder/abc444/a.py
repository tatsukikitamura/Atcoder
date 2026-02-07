N = str(input())
w = N[0]
ans = False
for x in range(len(N)):
    if w != N[x]:
        ans = True
    

if ans:
    print("No")
else:
    print("Yes")
