str = str(input())
len = len(str)
use = []
x = 0
print(len)

while x < len:
    try:
        if type(str[x]) == str:
            use.append(str[x])
            x += 1
        
        elif type(str[x]) == int:
            N = int(str[x])
            x += 1
            if str[x] == "(":
             
                l = []
                while str[x] != ")":
                    x += 1
                    l.append(str[x])
                use.append(l)
                x += 1
    except Exception as e:
        print(e)
        break
   

print(use)
