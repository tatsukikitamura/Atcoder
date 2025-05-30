#a**2 + b**2 = c**2
#a*b <= 12000
#a <= b
#a <= 109
import math
answer = 0
a = []

for x in range(109):
    y = x+1
    while True:
        if y * (x+1) > 12000:
            break
        if math.sqrt((x+1)**2+y**2) % 1 == 0:
                answer += 1
        y += 1
        
print(answer)