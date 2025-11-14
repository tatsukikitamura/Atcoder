def solution(A):
    use = []
    for x in range(len(A)):
        count2 = []
        count2.append(A[x])
        for y in range(x+1,len(A)):
            if A[x] == A[y] or A[x] == (A[y]-1):
                count2.append(A[y])
        use.append(count2)
    ans = []
    for x in range(len(use)):
        ans.append(max_length(use[x]))  
    return max(ans)

def max_length(list):
    first = list[0]
    count = []
    for x in range(len(list)):
        for y in range(x+1,len(list)):
                max_length = 2
                if list[x] == first and list[y] == first+1:
                    try:
                        for z in range(x):
                            if list[z] == first:
                                max_length += 1
                    except IndexError:
                        break
                    try:
                        for w in range(y+1,len(list)):
                            if list[w] == first+1:
                                max_length += 1
                    except IndexError:
                        break
                count.append(max_length)
            
    if not count:
        return 1
    return max(count)
                




