
def check_sums():
    # Read target values B from input
    with open('tools/in/0000.txt') as f:
        lines = f.read().split()
    B = [int(x) for x in lines[4:]]
    sum_B = sum(B)
    
    # Read generated values A from output
    try:
        with open('output.txt', 'r') as f:
            a_line = f.readline().strip()
            A = [int(x) for x in a_line.split()]
        sum_A = sum(A)
    except FileNotFoundError:
        print("Error: output.txt not found.")
        return
        
    print(f"Sum B: {sum_B}")
    print(f"Sum A: {sum_A}")
    print(f"Diff (B - A): {sum_B - sum_A}")

if __name__ == "__main__":
    check_sums()
