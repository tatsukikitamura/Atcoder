m,d = map(str,input().split())

m_list = []
d_list = []

for x in range(len(m)):
    m_list.append(m[x])
for x in range(len(d)):
    d_list.append(d[x])



for x in m_list:
    if x in d_list:
        continue
    else:
        print("No")
        exit()

print("Yes")