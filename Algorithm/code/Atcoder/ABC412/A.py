N = int(input())
count = 0

for x in range(N):
  a,b = map(int,(input().split()))
  if  a < b:
    count += 1

print(count)
