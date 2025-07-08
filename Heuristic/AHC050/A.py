import random
def sample():
    f = open('Heuristic/AHC050/myfile.txt','w')
    for i in range(N**2 -M):
        f.write(" ".join([str(x) for x in ANS[i]]))
        f.write("\n")


GYOURETSU = [[0,1],[1,0],[0,-1],[-1,0]]
N,M = map(int,input().split())
LI = []
USERS = []
FIRST_NO_ROCK = []
ANS = []
NO_ROCK = []

for i in range(N):
    USE = input()
    LI.append(USE) 

for i in range(N):
    USE = []
    for j in range(N):
        if (LI[i])[j] == '.':
            FIRST_NO_ROCK.append([[i,j],0])
            NO_ROCK.append([i,j])
            USE.append(0)
        else:
            FIRST_NO_ROCK.append([[i,j],-1])
            USE.append(1)
    USERS.append(USE)

def search(X,Y):
    for MATRIX in GYOURETSU:
        while True:
            if X*40 + Y > 1600 or X*40 + Y < 0:
                break
            POSITION = FIRST_NO_ROCK[X*40+Y][0]
            X += MATRIX[0]
            Y += MATRIX[1]
            if X*40 + Y > 1600 or X*40 + Y < 0:
                break
            
            if FIRST_NO_ROCK[X*40+Y][1] == -1 or POSITION[0] == -1 or POSITION[1] == -1 or POSITION[0] == 40 or POSITION[1] == 40:
                X -= MATRIX[0]
                Y -= MATRIX[1]
                FIRST_NO_ROCK[X*40+Y][1] += 1
                break


def main():
    while True:
        for LIST in NO_ROCK:
            search(LIST[0],LIST[1])
        for X in range(40):
            for Y in range(40):
                if FIRST_NO_ROCK[X*40+Y][1] == 0:
                    ANS.append([X,Y])
                    NO_ROCK.remove([X,Y])
                    FIRST_NO_ROCK[X*40+Y][1] = -1
        if len(NO_ROCK) == 0:
            break
        USE = NO_ROCK.pop(0)
        ANS.append(USE)
        print(FIRST_NO_ROCK)
main()
sample()

            

    

    


