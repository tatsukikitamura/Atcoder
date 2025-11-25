N,M = map(int,input().split())
win_false = []
for _ in range(M):
    win_false.append(list(map(int,input().split())))

number = []
for x in range(N):
    number.append([x+1])

print(win_false)
print(number)


for x in win_false:
    for y in number:
        if x[0] in y:
            start_ptr = number.index(y)
        elif x[1] in y:
            end_ptr = number.index(y)

    number[start_ptr].extend(number[end_ptr])
    number.remove(number[end_ptr])
    #print(number)
#print(number)

ans_length = 0
ans_ptr = []
for x in range(len(number)):
    ans_length = max(ans_length, len(number[x]))

for x in range(len(number)):
    if len(number[x]) == ans_length:
        ans_ptr.append(number[x][0])


for x in ans_ptr:
    print(x)

