N = int(input())
A = list(map(int,input().split()))
max_a = max(A)
diff = [0] * (max_a + 100)
for a in A:
    diff[0] += 1
    diff[a] -= 1
    for i in range(max_a):
        diff[i+1] += diff[i]
    ans = []
    use = 0
    for i in range(max_a + 80): 
        current = diff[i] + use
        ans.append(str(current % 10))
        carry = current // 10
    result = "".join(ans[::-1]).lstrip('0')
print(result if result else "0")
