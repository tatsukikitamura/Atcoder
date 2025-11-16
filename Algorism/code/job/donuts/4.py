import math

def solution(A):
    use = []
    for x in str(A):
        use.append(int(x))
    ans = []
    use.sort()
    count = 1
    for x in range(len(use)):
        if x != len(use)-1:
            if use[x] == use[x+1]:
                count += 1
            else:
                ans.append(count)
                count = 1
        else:
            ans.append(count)
    use2 = 1
    for x in range(len(ans)):
        use2 *= math.factorial(ans[x])
    answer = math.factorial(sum(ans)) // use2
    print(use)
    if 0 in use:
        zero_count = use.count(0)
        remain = use.copy()
        while 0 in remain:
            remain.remove(0)
        if len(remain) == 0:
            return answer
        ans2 = []
        remain.sort()
        count2 = 1
        for x in range(len(remain)):
            if x != len(remain)-1:
                if remain[x] == remain[x+1]:
                    count2 += 1
                else:
                    ans2.append(count2)
                    count2 = 1
            else:
                ans2.append(count2)
        if zero_count > 1:
            ans2.append(zero_count - 1)
        use3 = 1
        for x in range(len(ans2)):
            use3 *= math.factorial(ans2[x])
        answer2 = math.factorial(sum(ans2)) // use3
        answer = answer - answer2
    return answer

print(solution(11120))