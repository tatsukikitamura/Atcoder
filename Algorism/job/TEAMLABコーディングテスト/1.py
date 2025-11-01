a = []

a.append(1)
a.append(2)


for x in range(37):
    a.append(a[x]+a[x+1]-1)


print(len(a))
print(a)