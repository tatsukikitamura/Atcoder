N,L = map(int,input().split())
user =0
div = L // 3
user1 =[0]
preans = []
use = list(map(int,input().split()))

for i in range(N-1): 
    user += use[i]
    user2 = user % L
    user1.append(int(user2))
  
user1.sort()
print(user1)
for i in range(div):
  preanswer = [0]*(N+1)
  for j in range(3):
    preanswer[(i+(L//3)*(j))] += 1
  preans.append(preanswer)
  


print(preans)