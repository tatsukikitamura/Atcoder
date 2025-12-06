STR = input()
len = len(STR)
use = {}
x = 0


def isString(input_data) -> bool:
    return isinstance(input_data, str)

def check(STR:list,N:int) -> int:
    NUMBER = 1
    for y in range(N):
        if STR[y] == '(':
            NUMBER = NUMBER * STR[y-1]
        elif STR[y] == ')':
            NUMBER = 1
        elif STR[y].isdigit():
            NUMBER = int(STR[y])*NUMBER
    return NUMBER


for y in range(len):
    if isString(STR[y]) and STR[y] not in use:
        use[STR[y]] = check(STR[y],y)
    elif isString(STR[y]) and STR[y] in use:
        use[STR[y]].values()[0] += check(STR[y],y)

print(use)

import collections
import sys

sys.setrecursionlimit(1000000)

S = input()
L = len(S)

def parse(x):
    counts = collections.Counter()
    num_str = ""
    while i < L:
        char = S[i]
        if char.isdigit():
            num_str += char
            i += 1
        elif char == "(":
            N = int(num_str) 



