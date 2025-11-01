N,M = map(int,input().split())
LIST = []

for _ in range(N):
    LIST.append(str(input()))

list = []
list2 = []

for x in range((N-M+1)):
    for y in range((N-M+1)):
        list = []
        for z in range(M):
            for w in range(M):
                list.append(LIST[x+z][y+w])
        list2.append(list)

unique_types_set = set(tuple(row) for row in list2)  
    
print(len(unique_types_set))
