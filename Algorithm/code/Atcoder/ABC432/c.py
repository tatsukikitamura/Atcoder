N,X,Y = map(int,input().split())
LIST = list(map(int,input().split()))

LIST.sort()
Z = Y - X

#N = 3  X = 6 Y = 8
#LIST = [10,11,13]

under_max_number = LIST[0]*Y  #80
upper_min_number = LIST[N-1]*X  #78

count = 0
if (under_max_number - upper_min_number) % Z != 0:
    print(-1)

elif under_max_number >= upper_min_number:
    for x in range(N):
        if int(under_max_number - X* LIST[x]) % Z != 0:
            count = -1
            break 
        elif LIST[x]*Y == under_max_number:
            count += LIST[x]
        else:
            count += int((under_max_number - X*LIST[x]) // Z)
    print(count)

else:
    print(-1)



