n,k = map(int,input().split())
s = str(input())
s_list = list(s)


def first_solve(n,s):
    for x in range(n):
        if s[x] != ".":
            return False
    return True

if first_solve(n,s):
    print(n)
    exit()

if k == 1:
    use = []
    for x in range(n-2):
        count = 0
        for y in range(3):
            if s[x+y] == "S":
                count += 1
        use.append(count)
    index = use.index(max(use))
    for y in range(3):
        s_list[index+y] = "."
    count2 = 0
    for x in range(n):
        if s_list[x] == ".":
            count2 += 1
    print(count2)
    exit()
            
if k == 2:
    max_count = 0
    for i in range(n-2):
        for j in range(n-2):
            temp_list = list(s)
            for y in range(3):
                temp_list[i+y] = "."
            for y in range(3):
                temp_list[j+y] = "."
  
            count = 0
            for x in range(n):
                if temp_list[x] == ".":
                    count += 1
            max_count = max(max_count, count)
    print(max_count)

