H,W = map(int,input().split())
LIST = []


def plus(list1,list2):
    
    for x in range(len(list1)):
        list3 = []
        for y in range(-1,2):
            if x+y < 0 or x+y >= len(list1):
                continue
            else:
                list3.append(list1[x+y] + list2[x])
        list2[x] = max(list3)
    return list2

for x in range(H):
    LIST.append(list(map(int,input().split())))

for x in range(H-1):
    LIST[x+1] = plus(LIST[x],LIST[x+1])
    


print(max(LIST[H-1]))






    
