#7
#1 6 2 10 2 3 2
N =int(input())
li =list(map(int,input().split()))
ans =[]

li.sort()

for i in li:
    count = 0
    for j in range(N):
        if i <= li[j]:
                count += 1
    if i <= count:
        continue
    else:
<<<<<<< HEAD:ABC409/B.py
        ans.append(i)
        break
=======
         print(i)
         break
>>>>>>> 42b34f58c5dabbd0e6cf6587a9861b2b758fdd2e:Algorism/ABC409/B.py
    
    
if ans == []:
     print(0)
else:
     print(max(ans))

