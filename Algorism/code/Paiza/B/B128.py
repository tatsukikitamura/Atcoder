N = str(input())

zero_list = ["...","...","..."]
one_list = ["#..","...","..."]
two_list = ["##.","...","..."]
three_list = ["###","...","..."]
four_list = ["###","#..","..."]
five_list = ["###","##.","..."]
six_list = ["###","###","..."]
seven_list = ["###","###","#.."]
eight_list = ["###","###","##."]
nine_list = ["###","###","###"]

ans = []
for x in range(len(N)):
    if N[x] == "0":
        ans.append(zero_list)
    if N[x] == "1":
        ans.append(one_list)
    elif N[x] == "2":
        ans.append(two_list)
    elif N[x] == "3":
        ans.append(three_list)
    elif N[x] == "4":
        ans.append(four_list)
    elif N[x] == "5":
        ans.append(five_list)
    elif N[x] == "6":
        ans.append(six_list)
    elif N[x] == "7":
        ans.append(seven_list)
    elif N[x] == "8":
        ans.append(eight_list)
    elif N[x] == "9":
        ans.append(nine_list)
    
print(ans)
ans_list = []
for x in range(len(ans)//3):
    for y in range(3):
        use = []
        for z in range(3):
            use.append(ans[x*3+z][y])
        ans_list.append(use)

for x in range(len(ans_list)):
    print("".join(ans_list[x]))