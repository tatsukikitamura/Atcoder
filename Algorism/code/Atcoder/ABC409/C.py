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
tenkanri = [0]*L

for i in user1:
   tenkanri[i] += 1

lastanswer = 0

for i in range(div):
  answer =1
  for j in range(3):
    answer *= tenkanri[(i+(div)*(j))]
  lastanswer += answer

if L % 3 != 0:
   lastanswer = 0
print(lastanswer)
  
  

