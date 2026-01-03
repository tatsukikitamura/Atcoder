import math
ans = []
N = 10**7
li = set()
for z in range(1,N):
    count = 0
    square = math.sqrt(z)
    use = int(square // 1)
    li.add(z**2)
    for x in range(1,use+1):
        if (z**2 - x**2) in li:
            count += 1
    
    if count == 1:
        ans.add(z)
    print(z)

print(li)

