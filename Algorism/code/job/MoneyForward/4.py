from collections import defaultdict

s = str(input())
n = len(s)

alphabet_counts = defaultdict(int)
unique_alphabet_count = 0
symbol_count = 0

SYMBOLS = set(['@', '$', '%'])

min_len = n + 1  
left = 0       

for right in range(n):
    char_r = s[right]
    
    if 'a' <= char_r <= 'z':
        if alphabet_counts[char_r] == 0:
            unique_alphabet_count += 1
        alphabet_counts[char_r] += 1
    elif char_r in SYMBOLS:
        symbol_count += 1
        
    while unique_alphabet_count >= 5 and symbol_count >= 1:
        current_len = right - left + 1
        if current_len < min_len:
            min_len = current_len
        
        char_l = s[left]
        if 'a' <= char_l <= 'z':
            alphabet_counts[char_l] -= 1
            if alphabet_counts[char_l] == 0:
                unique_alphabet_count -= 1
        elif char_l in SYMBOLS:
            symbol_count -= 1
        
        left += 1  
print(min_len)

