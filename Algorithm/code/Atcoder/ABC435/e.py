from sortedcontainers import SortedList, SortedSet, SortedDict
import sys
input = sys.stdin.readline

N,M = map(int,input().split())
intervals = SortedList()
for _ in range(M):
    A,B = map(int,input().split())
    