T = int(input())

def ans(N,li):
  li.sort()
  j = 0
  while True:
    if len(li) < j+2:
      return len(li)
    elif li[j]*2 >= li[j+1]:
      j += 1
      continue
    else:
      del li[j+1]


for i in range(T):
  N = int(input())
  case = list(map(int,input().split()))
  answer = ans(N,case)
  if answer == 1:
    print("-1")
  else:
    print(answer)




