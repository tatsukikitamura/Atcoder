N, A, B = map(int, input().split())
S = list(input())
l = 0
idx_a, idx_b = [0], [0]
res = 0
for r in range(N):
    if S[r] == 'a':
        idx_a.append(r+1)
    else:
        idx_b.append(r+1)
    if len(idx_a)-1 >= A:
        al = idx_a[-A]
        if len(idx_b)-1 >= B:
            bl = idx_b[-B]
        else:
            bl = 0
        res += max(0, al - bl)


print(idx_a)
print(idx_b)
print(res)