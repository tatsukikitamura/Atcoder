from operator import lshift


N = int(input())
len_list = list(map(int, input().split()))
use = list(map(str, input().split()))

use_list = []
for x in range(N):
    use_list.append([use[x][0], use[x][-1]])

print(use_list)

count = 0
for x in range(N):
    for y in range(x+1, N):
        for z in range(y+1, N):
            use = [use_list[x], use_list[y], use_list[z]]
            print(use)
            if use[0][0] == use[1][0] and use[1][1] == use[2][0] and use[0][1] == use[2][1]:
                count += 1
            elif use[0][0] == use[1][0] and use[1][1] == use[2][1] and use[0][1] == use[2][0]:
                count += 1
            elif use[0][0] == use[1][0] and use[1][1] == use[2][0] and use[0][1] == use[2][1]:  
                count += 1
            elif use[0][0] == use[1][1] and use[1][0] == use[2][1] and use[0][1] == use[2][0]:
                count += 1
            elif use[0][0] == use[1][1] and use[1][0] == use[2][0] and use[0][1] == use[2][1]:
                count += 1

print(count)