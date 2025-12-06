a,b = map(int,input().split())

c = list(map(int,input().split()))

answer = True

for x in range(a-1):
    if (c[x+1]-c[x]) > b:
        answer = False
        break
if c[0] > b:
    answer = False

if answer == True:
    print("Yes")
else:
    print("No")
    
    
