N = int(input())
grid = []
for _ in range(N):
    num = str(input())
    use = []
    for x in range(N):
        use.append(int(num[x]))
    grid.append(use)

print(grid)
ans = 0
def check(list:list,N:int) -> int:
    max_ans = 1
    for x in range(N):
        ans = 1
        count = list[x]
        for y in range(x+1,N):
            if list[y] == count+1:
                count += 1
                ans += 1
            else:
                max_ans = max(max_ans,ans)
                count = list[y]
                break
  
        max_ans = max(max_ans,ans)
    for x in range(N):
        ans =  1
        count = list[x]
        for y in range(x+1,N):
            if list[y] == count-1:
                ans += 1
            else:
                max_ans = max(max_ans,ans)
                count = list[y]
                break
        max_ans = max(max_ans,ans)
    return max_ans

#横
for x in range(N):
    ans = max(ans,check(grid[x],N))
#print("横",ans)
#縦
for x in range(N):
    use = []
    for y in range(N):
        use.append(grid[y][x])
    ans = max(ans,check(use,N))
#print("縦",ans)
#左上右下斜め
for i in range(-N+1,N):
    use = []
    for x in range(N):
        for y in range(N):
            if x-y == i:
                use.append(grid[y][x])
    #print(use)
    ans = max(ans,check(use,len(use)))
#print("左上右下斜め",ans)
#右上左下斜め
for i in range(2*N-1):
    use = []
    for x in range(N):
        for y in range(N):
            if x+y == i:
                use.append(grid[y][x])
    ans = max(ans,check(use,len(use)))
#print("右上左下斜め",ans)


print(ans)