target = str(input())
ans = str(input())
answer = True

for i in range(len(target)):
  if i == 0:
    continue
  elif target[i].isupper():
    if target[i-1] not in ans:
      answer = False
      break
    else:
      answer = True


      
if answer:
  print("Yes")
else:
  print("No")

