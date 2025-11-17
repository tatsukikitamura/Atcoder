N = int(input())
LIST = []


for _ in range(N):
    x,y = input().split()
    LIST.append([x,y])

q,t = input().split()

h,m = t.split(":")
h = int(h)
m = int(m)

if m < 10:
    m = f"0{m}"
else:
    m = m

for x in range(N):
    if LIST[x][0] == q:
        target = LIST[x][1]
        break

for x in range(N):
    JISA = int(LIST[x][1]) - int(target)
    if h + JISA >= 24:
        if h+JISA-24 < 10:
            print(f"0{h+JISA-24}:{m}")
        else:
            print(f"{h+JISA-24}:{m}")
    elif h +JISA < 0:
        if h+JISA+24 < 10:
            print(f"0{h+JISA+24}:{m}")
        else:
            print(f"{h+JISA+24}:{m}")
    elif h +JISA < 10:
        print(f"0{h+JISA}:{m}")
    else:
        print(f"{h+JISA}:{m}")


    
