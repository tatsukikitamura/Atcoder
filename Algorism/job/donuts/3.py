def solution(A):
    count = 0
    use = []
    for y in range(len(A)):
        if A[y] == 1:
            A[y] = 0
            for x in range(len(A)-1):
                if A[x] == A[x+1]:
                    count += 1        
                else:
                    use.append(count+1)
                    count = 0
                use.append(count+1)
                count = 0
            A[y] = 1
        elif A[y] == 0:
            A[y] = 1
            for x in range(len(A)-1):
                if A[x] == A[x+1]:
                    count += 1        
                else:
                    use.append(count+1)
                    count = 0
            A[y] = 0

        
    return max(use)
  


