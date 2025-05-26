num = input()
strnum = str(num)
length = len(strnum)
answer = 0
answer += length

def math(x):
    return (x % 10)

count = 0
for x in range(length):
    use =  int(strnum[length-x-1])
    user = math(use-count)
    answer += user
    count += user

print(answer)