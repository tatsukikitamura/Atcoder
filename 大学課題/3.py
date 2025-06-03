import random as ran
for i  in range(10):
    out = ran.randint(0,2)
    if out % 3 == 0:
        print("ぐー")
    elif out % 3 == 1:
        print("ちょき")
    else:
        print("ぱー")
        