N = int(input())
a_list = list(map(int,input().split()))

count = 0
for x in range(N):
    for y in range(x+1,N):
        use = 0
        use2 = False
        for z in range(x,y+1):
            use += a_list[z]
        
        for z in range(x,y+1):
            if use % a_list[z] == 0:
                use2 = True
                break

        if not use2:
            count += 1

        

print(count)
