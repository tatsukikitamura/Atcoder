N = int(input())
LIST = []

for x in range(N):
  a = input()
  LIST.append(a)

X,Y = input().split()

if LIST[int(X)-1] == Y:
  print("Yes")
else:
  print("No")
