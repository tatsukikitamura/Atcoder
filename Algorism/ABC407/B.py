array = []
answer =[]

a,b = map(int,input().split())
for x in range(6):
    for y in range(6):
        array.append([x+1,y+1])
        
def sum(x):
    return array[x][0]+array[x][1]
        
for x in range(len(array)):
    if sum(x) >= a or abs(array[x][0]-array[x][1]) >= b:
        answer.append(array[x])
    
print(len(answer)/36)
