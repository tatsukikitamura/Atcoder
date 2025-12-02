N, M = map(int, input().split())

gondora = []
gondora_count = []

for _ in range(N):
    gondora.append(int(input()))
    gondora_count.append(0)

group = []
for _ in range(M):
    group.append(int(input()))

#print(gondora)
#print(group)
group_ptr = 0
gondora_ptr = 0
while group_ptr < M:
    #print(f"gondora_ptr:{gondora_ptr},group_ptr:{group_ptr}")
  
    #print(gondora_count)
    if gondora_ptr >= N:
        gondora_ptr = 0


    if gondora[gondora_ptr] >= group[group_ptr]:
        gondora_count[gondora_ptr] += group[group_ptr]
        group_ptr += 1
        gondora_ptr += 1
    


    else:
        group[group_ptr] -= gondora[gondora_ptr]
        gondora_count[gondora_ptr] += gondora[gondora_ptr]
        gondora_ptr += 1

for x in gondora_count:
    print(x)
