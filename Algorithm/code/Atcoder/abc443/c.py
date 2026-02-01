N,T = map(int,input().split())
A = list(map(int,input().split()))
time = 0
ans = 0
if not A:
    ans = T
else:
    for x in A:
        if time > x:
            continue
        else:
            ans += (x - time)
            time = x + 100
    # 最後のブロック終了時刻から T までの区間を足す
    ans += max(0, T - time)

print(ans)


