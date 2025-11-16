import random
count = 0

for i in range(10000):
  x = random.uniform(0,1)
  y = random.uniform(0,1)
  if x**2 + y**2 <= 1:
    count += 1

print((count/10000 *4))
    

