def solution(A):
    use = [-1]
    for x in range(len(A)):
        for y in range(x+1,len(A)):
            for z in range(y+1,len(A)):
                if A[x] + A[y] > A[z] and A[x] + A[z] > A[y] and A[y] + A[z] > A[x]:
                    use.append(A[x]+A[y]+A[z])
    return max(use)



