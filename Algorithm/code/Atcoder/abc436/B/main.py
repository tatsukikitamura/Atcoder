import sys
import math
from collections import deque, defaultdict, Counter
from itertools import accumulate, permutations, combinations, product
from bisect import bisect_left, bisect_right
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def main():
    N = int(input())
    mod_list = [[0]*N for _ in range(N)]

    mod_list[0][(N-1)//2] = 1
    index = [0, (N-1)//2]
    num = 1
    for _ in range(N**2-1):
        if mod_list[(index[0]-1)%N][(index[1]+1)%N] == 0:
            mod_list[(index[0]-1)%N][(index[1]+1)%N] = num + 1
            index = [(index[0]-1)%N, (index[1]+1)%N]
            num += 1
        else:
            mod_list[(index[0]+1)%N][index[1]] = num + 1
            index = [(index[0]+1)%N, index[1]]
            num += 1
        
        #print(mod_list)


    for x in mod_list:
        print(" ".join(map(str,x)))


    

if __name__ == '__main__':
    main()
