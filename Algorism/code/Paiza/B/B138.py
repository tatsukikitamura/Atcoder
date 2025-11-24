H,W = map(int,input().split())
grid = []
for _ in range(H):
    grid.append(str(input()))

count = 0
for x in range(H-2):
    for y in range(W-2):
        check = True
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    if grid[x+i][y+j] == ".":
                        continue
                    else:
                        check = False
                        break
                if grid[x+i][y+j] == "#":
                    continue 
                else:
                    check = False
                    break
        if check:
            count += 1


print(count)    

