S = list(map(int,input().split()))

S.sort()

for x in range(3):
    print(S[2-x],end="")
print()