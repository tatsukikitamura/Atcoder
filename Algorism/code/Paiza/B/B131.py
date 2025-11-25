N,M = map(int,input().split())
money =[]
for _ in range(N):
    money.append(list(map(int,input().split())))


X = int(input())
routes = []
for _ in range(X):
    routes.append(list(map(int,input().split())))

#print(money)
#print(routes)
ans = 0
state = 0
for route in routes:
    ans += abs(money[route[0]-1][route[1]-1]-money[route[0]-1][state])
    state = route[1]-1
    
print(ans)