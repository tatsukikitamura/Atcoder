N = int(input())
count = 0
for x in range(1,N+1):
    count += ((-1)**x)*(x**3)
    
print(count)
    