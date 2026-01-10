N = int(input())
li = list(map(int,input().split()))
ans_li = []
for _ in range(3):
  ans = 5000
  ind = 0
  for x in range(N):
    if li[x] < ans:
        ans = li[x]
        ind = x
    

    
    #print(ans)
    #print(li)
    
  li[ind] = 10000
  ans_li.append(str(ind+1))
  

  
ans_li=' '.join(ans_li)
print(ans_li)