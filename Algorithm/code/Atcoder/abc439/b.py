N = str(input())

BIGINT = 2000000
ans = False

for x in range(BIGINT):
  count =0
  for y in range(len(N)):
    count += int(N[y])**2


  N = str(count)
  if count == 1:
    ans = True
    break
  
if ans:
  print("Yes")
else:
  print("No")
  
  
