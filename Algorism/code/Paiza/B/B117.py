N = int(input())
a_list = []
for _ in range(N):
    a_list.append(int(input()))

count = 1
ptr = 0
ans = 0
while a_list != []:
    #print(a_list)
    if ptr >= len(a_list):
        ptr = 0
        ans += 1

    if a_list[ptr] == count:
        a_list.remove(count)
        count += 1
  
    else:
        ptr += 1

print(ans)