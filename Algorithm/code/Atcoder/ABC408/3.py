a,b = map(int,input().split())
use = []
list =[] #１からｂまで
count=[0]*a

for i in range(a):
    list.append(i+1)


for i in range(b):
    x,y = map(int,input().split())
    user = []
    for j in range(x,y+1):
        user.append(j)       
    use.append(user)
    
##print(use)
    
for i in range(1,a+1):
    for j in range(len(use)):
        if i in use[j]:
            count[i-1] += 1

##print(count)
print(min(count))

