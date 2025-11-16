N = int(input())
A,B = map(int,input().split())
LIST = []
USE = [1]
for x in range(N):
    USE.append(0)
    LIST.append(x)

def check(NUM:int):
    if USE[NUM-A] == 1 or USE[NUM-B] == 1:
        USE[NUM] = 1
        return True
    else:
        return False

for x in range(1,N):
    if check(x):
        count += 1
        
#print(USE)
print(N-count-1)
#print(USE)


