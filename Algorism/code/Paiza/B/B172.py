N = int(input())
LIST = list(map(int,input().split()))
answer = []


for x in range(N+1):
    for y in range(x+1,N+1):
     
        USE = []
        USE.append(LIST[x:y])
        USE.sort()
        if len(USE) % 2 == 1:
            answer.append(USE[len(USE)//2])
        else:
            answer.append((USE[len(USE)//2-1]+USE[len(USE)//2])/2)


answer_list = []
for x in answer:
    x.sort()
    if len(x) % 2 == 1:
        answer_list.append(x[len(x)//2])
    else:
        answer_list.append((x[len(x)//2-1]+x[len(x)//2])/2)





def format_number(num):
  if isinstance(num, float) and num.is_integer():
    return int(num)
  else:
    return num


answer_list.sort()

length = len(answer) 
if length % 2 == 0:
   print(format_number((answer_list[length//2-1]+answer_list[length//2])/2))
else:
   print(format_number(answer_list[length//2]))
        
