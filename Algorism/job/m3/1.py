def solution(text):
    count = []
    LIST = []
    use = []
    for x in text:
        if x == " ":
            LIST.append(use)
            use = []
        elif x == ".":
            LIST.append(use)
            use = []
        else:
            use.append(x)
    LIST.append(use)

    for x in range(len(LIST)):
        if  len(LIST[x]) > 0 and (LIST[x][0].isupper() or LIST[x][0].isdigit()):
            count.append(LIST[x])

    use1 = set(tuple(row) for row in count)

    return len(use1)
    
print(solution("Scala is a general purpose programming language designed to express common programming patterns in a concise, elegant, and type-safe way. It smoothly integrates features of object-oriented and functional languages, enabling Java and other programmers to be more productive. Code sizes are typically reduced by a factor of two to three when compared to an equivalent application by Java."))