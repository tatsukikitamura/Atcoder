A,B = map(int,input().split())
X,Y = map(int,input().split())
A_check = False
B_check = False

if (X-A) >= 0  and (X-A) <= 99:
  A_check = True
  
if (Y-B) >= 0 and (Y-B) <= 99:
  B_check = True
  
  

if A_check and B_check:
  print("Yes")
else:
  print("No")