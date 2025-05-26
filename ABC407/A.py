import math
a,b = map(int,input().split())

c = a / b

d = math.floor(c)

if c - d >= 0.5 :
    print (d+1)
else:
    print(d)