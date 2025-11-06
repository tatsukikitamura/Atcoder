N,A,B = map(int,input().split())
S = str(input())
# "a" >= A
# "b" < B

list = [0,0]
use_count = 0

def check(x:int,list:list) -> int:
    count = 0
    while True:
        if S[x] == "a":
            list[0] += 1
        elif S[x] == "b":
            list[1] += 1
        elif list[0] >= A and list[1] < B:
            count += 1
        elif list[0] >= A and list[1] >= B:
            break
        x += 1
    return (x,count)

x = 0
while x < N:
    if S[x] == "a":
        y,count = check(x,list)
        use_count += count
        x +=  y
        list[0] -=  1
        
    elif S[x] == "b":
        y,count = check(x,list)
        use_count += count
        x +=  y
        list[1] -= 1

print(use_count)