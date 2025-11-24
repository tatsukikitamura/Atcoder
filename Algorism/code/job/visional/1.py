num = str(input())

if not num or (num[0] == "0" and num[1] != "."):
    print(-1)
    exit()


for x in range(len(num)):
    if num[x] == "-" or  num[x] == "." or num[x].isdigit():
        continue
    else:
        print(-1)
        exit()
dot = False
num_list = []
dot_point = len(num)-1
for x in range(len(num)):
    if num[x] == ".":   
        dot = True
        dot_point = x-1
        num_list.append(num[x])
        continue
    else:
        num_list.append(int(num[x]))


def henkan(num):
    if num == 0:
        return "zero"
    elif num == 1:
        return "one"
    elif num == 2:
        return "two"
    elif num == 3:
        return "three"
    elif num == 4:
        return "four"
    elif num == 5:
        return "five"
    elif num == 6:
        return "six"
    elif num == 7:
        return "seven"
    elif num == 8:
        return "eight"
    elif num == 9:
        return "nine"

use_list = []
for x in range(len(num_list)):
    use_list.append(dot_point-x)

def check(num,num2,x):
    if num[x] == 4:
        if num2[x] == 1:
            if num2[x+1] == 0:
                ans_list.append("ten")
            elif num2[x+1] == 1:
                ans_list.append("eleven")
            elif num2[x+1] == 2:
                ans_list.append("twelve")
            elif num2[x+1] == 3:
                ans_list.append("thirteen")
            elif num2[x+1] == 4:
                ans_list.append("fourteen")
            elif num2[x+1] == 5:
                ans_list.append("fifteen")
            elif num2[x+1] == 6:
                ans_list.append("sixteen")
            elif num2[x+1] == 7:
                ans_list.append("seventeen")
            elif num2[x+1] == 8:
                ans_list.append("eighteen")
            elif num2[x+1] == 9:
                ans_list.append("nineteen")
        elif num2[x] == 2:
            ans_list.append("twenty")
        elif num2[x] == 3:
            ans_list.append("thirty")
        elif num2[x] == 4:
            ans_list.append("forty")
        elif num2[x] == 5:
            ans_list.append("fifty")
        elif num2[x] == 6:
            ans_list.append("sixty")
        elif num2[x] == 7:
            ans_list.append("seventy")
        elif num2[x] == 8:
            ans_list.append("eighty")
        elif num2[x] == 9:
            ans_list.append("ninety")
    elif num[x] == 5:
        pass
    elif num[x] == 6:
        pass
    elif num[x] == 3:
        if num2[x] != 0:
            ans_list.append(henkan(num2[x]))
        ans_list.append("thousand")
    elif num[x] == 2:
        if num2[x] != 0:
            ans_list.append(henkan(num2[x]))
            ans_list.append("hundred")
    elif num[x] == 1:
        if num2[x] == 1:
            if num2[x+1] == 0:
                ans_list.append("ten")
            elif num2[x+1] == 1:
                ans_list.append("eleven")
            elif num2[x+1] == 2:
                ans_list.append("twelve")
            elif num2[x+1] == 3:
                ans_list.append("thirteen")
            elif num2[x+1] == 4:
                ans_list.append("fourteen")
            elif num2[x+1] == 5:
                ans_list.append("fifteen")
            elif num2[x+1] == 6:
                ans_list.append("sixteen")
            elif num2[x+1] == 7:
                ans_list.append("seventeen")
            elif num2[x+1] == 8:
                ans_list.append("eighteen")
            elif num2[x+1] == 9:
                ans_list.append("nineteen")
        elif num2[x] == 2:
            ans_list.append("twenty")
        elif num2[x] == 3:
            ans_list.append("thirty")
        elif num2[x] == 4:
            ans_list.append("forty")
        elif num2[x] == 5:
            ans_list.append("fifty")
        elif num2[x] == 6:
            ans_list.append("sixty")
        elif num2[x] == 7:
            ans_list.append("seventy")
        elif num2[x] == 8:
            ans_list.append("eighty")
        elif num2[x] == 9:
            ans_list.append("ninety")
    elif num[x] == -1:
        ans_list.append("point")
    elif num[x] == -2:
        ans_list.append(henkan(num2[x]))
    elif num[x] == -3:
        ans_list.append(henkan(num2[x]))
    elif num[x] == -4:
        ans_list.append(henkan(num2[x]))
    if num[x] == 0:
        if num[0] == 0:
            ans_list.append(henkan(num2[x]))
            return
        
        elif num2[x] != 0:
            if x-1 < 0 or num2[x-1] != 1:
                ans_list.append(henkan(num2[x]))
    if num[x] == 9:
        ans_list.append(henkan(num2[x]))
        ans_list.append("billion")



ans_list = []
for x in range(len(use_list)):
    check(use_list,num_list,x)
    
print(use_list)
print(num_list)
print(ans_list)


if not dot:
    print(" ".join(ans_list).capitalize())
else:
    print(" ".join(ans_list).capitalize())