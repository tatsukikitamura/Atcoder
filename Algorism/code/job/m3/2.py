def solution(text):
    LIST = text.split(",")
    use = []
    for x in range(len(LIST)):
        if  not x == 0 and (x % 2 == 1 or LIST[x] ==LIST[x-1]):
            use.append(LIST[x])
    return ",".join(use)

 







