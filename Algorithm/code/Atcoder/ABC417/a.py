N,A,B = map(int,input().split())
S = str(input())
li = []
for x in range(A,N-B):
    li.append(S[x])

result = "".join(li)

print(result)