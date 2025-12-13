import sys
import math
from collections import deque, defaultdict, Counter
from itertools import accumulate, permutations, combinations, product
from bisect import bisect_left, bisect_right
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def main():
    N,M = map(int,input().split())
    range_list = set()
    neighbor_list = [(0,0),(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]  
    for _ in range(M):
        R,C = map(int,input().split())
        if range_list == set():
            range_list.add((R,C))
        else:
            flag = False
            for x in neighbor_list:
                if (x[0]+R,x[1]+C) in range_list:
                    flag = True
                    break

            if flag == False:
                range_list.add((R,C))
            
        #print(range_list)
    print(len(range_list))


   
if __name__ == '__main__':
    main()
