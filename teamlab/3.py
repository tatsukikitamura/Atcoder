

answer = 0

for x in range(7000000):
    count = 0
    for y in range(len(str(x+1))):
        count += int(str(x+1)[y])
        
    if count % 7 ==0:
        answer += 1




print(answer)