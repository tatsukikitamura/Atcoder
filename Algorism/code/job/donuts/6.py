def solution(A):
    count = 0
    for x in range(len(A[0])):
        for y in range(len(A)):
            if A[y][x] != 0:
                count += dfs(x,y,A[y][x],A)




def dfs(x,y,color,A):
    count = 0
    DX = [1,0,-1,0]
    DY = [0,1,0,-1]
    for i in range(4):
        nx = x + DX[i]
        ny = y + DY[i]
        if 0 <= nx < len(A[0]) and 0 <= ny < len(A):
            if A[ny][nx] == color:
                dfs(nx,ny,color,A)
    return 1



print(solution( [[5, 4, 4], [4, 3, 4], [3, 2, 4], [2, 2, 2], [3, 3, 4], [1, 4, 4], [4, 1, 1]]))