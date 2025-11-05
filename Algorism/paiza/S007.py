STR = input()
len = len(STR)
use = {}
x = 0


def isString(input_data) -> bool:
    return isinstance(input_data, str)

def check(x,N):
    NUMBER = 1
    for x in range(N):
        if STR[x] == '(':
            NUMBER = NUMBER * STR[x-1]
        elif STR[x] == ')':
            NUMBER = 1
        elif STR[x].isdigit():
            NUMBER = int(STR[x])*NUMBER
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



