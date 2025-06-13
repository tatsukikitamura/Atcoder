99999999

a = []
answer = 0

for x in range(17727):
    a.append(5641*(x+1))

for x in range(len(a)):
    length = len(str(a[x]))     
    answer += int(str(a[x])[length-4])



print(answer)