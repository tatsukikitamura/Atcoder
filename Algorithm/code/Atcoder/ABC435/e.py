from sortedcontainers import SortedList, SortedSet, SortedDict
import sys
input = sys.stdin.readline

N,M = map(int,input().split())
intervals = SortedSet()
for _ in range(M):
    A,B = map(int,input().split())
    ptr = intervals.bisect_left((A,B))
    #print(ptr)
    #print(A,B)

    if len(intervals) == 0:
        intervals.add((A,B))
        #print(intervals)
        

    elif ptr == 0:
        if intervals[ptr][1] >= B:
            intervals.add((A, intervals[ptr][1]))
            intervals.discard(intervals[ptr+1])
        else:
            intervals.add((A,B))
    
    elif ptr == len(intervals):
        if intervals[ptr-1][1] >= A:
            intervals.add((intervals[ptr-1][0], B))
            intervals.discard(intervals[ptr-1])
        else:
            intervals.add((A,B))

    elif intervals[ptr-1][1] >= A:
        if intervals[ptr][0] <= B:
            intervals.add((intervals[ptr-1][0], intervals[ptr][1]))
            intervals.discard(intervals[ptr-1])
            intervals.discard(intervals[ptr])
        else:
            intervals.add((A,B))
    #print(intervals)
    count = 0
    for x in intervals:
        count += x[1] - x[0] + 1


    print(N-count)

