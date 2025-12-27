import sys

def simulate_binary_fill():
    # Values from 0000.txt
    L = 998000000000000
    B_str = "998048751022181 998097465238075 998160669164017 998231311927150 998239693029031 998356130354587 998446906912252 998555170329840 998587344209968 998743301391436 998760667153740 998947286002826 998981740663966 999002845061505 999059061039867 999128286657558 999297656665943 999358823245113 999383413043563 999640703784442 999733472254114 999790366912377 999873266864151 999911534238464 999985985322908 1000010039870477 1000155426700245 1000265729136158 1000462334004741 1000565907102777 1000908034069731 1000951689976892 1000957188499933 1001083800047313 1001089587866566 1001202738050092 1001213435143294 1001225125980911 1001264019997723 1001404093183272 1001438327970254 1001443079569383 1001470062137717 1001585047564559 1001647141491385 1001664377029572 1001719343473297 1001731992430657 1001856451123514 1001928070662947"
    
    Bs = [int(x) for x in B_str.split()]
    targets = [b - L for b in Bs] # The gaps we need to fill
    
    # Generate powers of 2
    # L gap is up to ~4e12. 
    # 2^42 approx 4.4e12
    # Let's generate a set of powers. 
    # Problem: we only have limited cards (N=500, M=50 used for bases -> 450 left).
    # Can we have enough bits for everyone?
    # 450 cards / 50 piles = 9 cards per pile.
    # 9 bits is NOT enough to cover 10^12 range precisely.
    # BUT, we can share cards!
    
    # Strategy: 
    # Create a pool of powers starting from largest needed (2^41) down to maybe 2^20
    # And Greedily assign them.
    
    # Max gap is 3.9e12. log2(3.9e12) = 41.8. So max bit is 2^41.
    
    # Let's define a card set.
    # We have 450 cards.
    # Let's try to have multiple copies of important bits.
    
    cards = []
    
    # Naive Binary Set: Many large bits, fewer small bits? 
    # Or just a standard set of powers?
    # Let's try a distribution: 
    # 10 copies of 2^41, 10 copies of 2^40 ... down to ...
    
    # Actually, let's simulates the best case:
    # We greedily take the largest power of 2 <= remaining_gap for each pile.
    # And count what cards we WOULD need.
    
    needed_cards = {}
    
    current_gaps = list(targets)
    total_cards_used = 0
    
    # Truncation Analysis
    print("\n--- Truncation Analysis ---")
    print(f"{'Cutoff Power':<15} {'Val':<20} {'Cards Needed':<15} {'Avg Last Err':<15}")
    
    # Check cutoffs from 10 to 40
    for cutoff in range(10, 41):
        needed_count = 0
        total_remaining_err = 0
        
        # Simulate for this cutoff
        temp_gaps = list(targets)
        for i in range(len(temp_gaps)):
            gap = temp_gaps[i]
            # Use powers down to cutoff
            for p in range(41, cutoff - 1, -1):
                val = 1 << p
                if gap >= val:
                    gap -= val
                    needed_count += 1
            total_remaining_err += gap
            
        print(f"2^{cutoff:<13} {1<<cutoff:<20_} {needed_count:<15} {total_remaining_err//50:<15_}")


if __name__ == "__main__":
    simulate_binary_fill()
