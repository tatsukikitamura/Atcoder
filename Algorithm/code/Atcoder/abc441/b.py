N,M = map(int,input().split())
S = str(input())
T = str(input())
Q = int(input())
s_set = set()
t_set = set()
for x in range(len(S)):
    s_set.add(S[x])

for x in range(len(T)):
    t_set.add(T[x])


for _ in range(Q):
    w = str(input())
    s_check = True
    t_check = True
    for x in range(len(w)):
        if w[x] not in s_set:
            s_check = False
            break
        elif w[x] not in t_set:
            t_check = False
    
    if s_check and t_check:
        print("Unknown")
    elif s_check:
        print("Takahashi")
    elif t_check:
        print("Aoki")