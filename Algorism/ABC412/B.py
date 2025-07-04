target = str(input())
ans = str(input())

for i in range(len(target)):
  if i == 0:
    continue
  elif target[i].isupper():
    if target[i-1] in ans:
      print("Yes")
      break
print("No")



























