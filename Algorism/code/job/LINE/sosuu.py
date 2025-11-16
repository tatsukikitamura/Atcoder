import sys

def check_sosuu(n):
    for x in range(2,n):
        if n % x == 0:
            return False
    return True

def count_sosuu(N):
    count = 0
    n = 1
    while count <= N:
        if check_sosuu(n):
            count += 1
        n += 1
    return (n-1)


if __name__ == "__main__":
    N = int(sys.stdin.readline())
    print(count_sosuu(N))