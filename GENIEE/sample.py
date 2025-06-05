import sys
a,b,c,d = map(int,input().split())

use = []
num = int(input())
for i in range(num):
    x,y = input().split()
    hours,minutes = map(int,x.split(":"))
    use.append([hours,minutes,int(y)])

answer = 0

def math(i):
    user =  ((use[i][2]-a) // c) 
    if (use[i][2]-a) % c != 0:
        user += 1 
    if use[i][0] >= 22:
        return ((d * user + b) * 1.2)
    else:
        return (d * user + b)
    

for j in range(num):
    
    answer += math(j)
    
print(int(answer))

