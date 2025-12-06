n,m,k = map(int, input().split())

str_list = []
b_x_list = []
b_y_list = []
b_list = []

for x in range(n):
    str_input = str(input())
    use = []
    for y in range(m):
        if str_input[y] == 'B':
            b_x_list.append(x)
            b_y_list.append(y)

        use.append(str_input[y])

    str_list.append(use)

set_b_x_list = set(b_x_list)
set_b_y_list = set(b_y_list)

b_x_list = list(set_b_x_list)
b_y_list = list(set_b_y_list)

for x in b_x_list:
    for y in range(m):
        b_list.append([x,y])

for x in b_y_list:
    for y in range(n):
        b_list.append([y,x])

b_list = list(set(tuple(x) for x in b_list))
b_list = sorted(b_list, key=lambda x: (x[0], x[1]))

#print(b_list)
#print(str_list)

danger_list = []
safe_list = []

for x in b_list:
    if str_list[x[0]][x[1]] == 'B':
        pass
    else:
        danger_list.append(int(str_list[x[0]][x[1]]))

use_list = []
for x in range(n):
    for y in range(m):
        use_list.append((x,y))

use_safe_list = [item for item in use_list if item not in b_list]

for x in use_safe_list:
    safe_list.append(int(str_list[x[0]][x[1]]))

danger_list.sort(reverse=True)
safe_list.sort()


#print(danger_list)
#print(safe_list)


for x in range(k):
    if len(danger_list) == 0:
        break
    elif len(safe_list) == 0:
        break
    elif danger_list[0] < safe_list[0]:
        break
    elif danger_list[0] > safe_list[0]:
        safe_list.append(danger_list.pop(0))
        danger_list.append(safe_list.pop(0))
    print(danger_list)
    print(safe_list)





#print(danger_list)
#print(safe_list)

print(sum(safe_list))