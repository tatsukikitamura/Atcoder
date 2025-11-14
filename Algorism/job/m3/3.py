from decimal import Decimal, ROUND_HALF_UP

def solution(text):
    LIST = text.split("\n")
    use = []
    for x in LIST:
        use.append(x.split(","))
    ans = []
    for x in range(4): 
        ans.append(Decimal((int(use[1][x]) + int(use[2][x]) + int(use[3][x]) + int(use[4][x]))/4).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return " ".join(str(int(x)) for x in ans)

print(solution(input()))