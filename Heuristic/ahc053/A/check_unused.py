
import collections

def main():
    try:
        with open('output.txt', 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        if len(lines) < 2:
            print("Error: output.txt should have at least 2 lines (A and X).")
            return

        A = list(map(int, lines[0].split()))
        X = list(map(int, lines[1].split()))

        if len(A) != len(X):
            print(f"Error: Length mismatch A({len(A)}) vs X({len(X)})")
            return

        unused_values = []
        for a, x in zip(A, X):
            if x == 0:
                unused_values.append(a)

        unused_values.sort()
        count = collections.Counter(unused_values)

        print(f"Total Unused Cards: {len(unused_values)}")
        print("\n--- Unused Values Histogram ---")
        
        # Determine ranges
        # Small: < 10^9
        # Middle: 10^9 ~ 4*10^12
        # Large: > 4*10^12 (mostly Base)
        
        small_unused = []
        middle_unused = []
        large_unused = []
        
        for v in unused_values:
            if v < 1000000000:
                small_unused.append(v)
            elif v < 4000000000000:
                middle_unused.append(v)
            else:
                large_unused.append(v)
                
        print(f"\nSmall (< 10^9): {len(small_unused)}")
        print(f"Counts: {collections.Counter(small_unused).most_common(10)}")
        
        print(f"\nMiddle (10^9 ~ 4e12): {len(middle_unused)}")
        # Middle values might be unique randoms, so histogram bins might be better
        if middle_unused:
            import statistics
            print(f"Min: {min(middle_unused)}")
            print(f"Max: {max(middle_unused)}")
            print(f"Mean: {statistics.mean(middle_unused):.2f}")
            
        print(f"\nLarge (> 4e12): {len(large_unused)}")
        print(f"Counts: {collections.Counter(large_unused)}")

    except FileNotFoundError:
        print("Error: output.txt not found. Run ./main first.")

if __name__ == "__main__":
    main()
