X = str(input())
LIST = []
for y in range(len(X)):
    LIST.append(int(X[y]))

LIST.sort()

for x in range(len(LIST)):
    if LIST[x] == 0:
        continue
    else:
        a = LIST.pop(x)
        LIST.insert(0,a)
        break

   
for x in range(len(LIST)):
    print(LIST[x],end="")
print()