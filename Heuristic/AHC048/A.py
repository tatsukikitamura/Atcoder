import math
import random 
import time

start_time = time.time() * 1000 
N,K,H,T,D = map(int,input().split())
N = 20
H = 1000
ads = []
ans_want = []
sikiri = []
masu = []
print1 = []

for i in range(K):  ##じぶんがもってるやつ
    x, y, r = map(float,input().split())
    ads.append([x, y, r, i])
    

for i in range(H): ##ほしいもの #機能している
    x,y,r = map(float,input().split())
    ans_want.append([x,y,r])

#3番目が1の時仕切りがある

for i in range(20):
    for j in range(19):
        sikiri.append([i,j,1,0]) #4番目が0の時縦
        sikiri.append([j,i,1,1]) #4番目が1の時横

for i in range(20):
    for j in range(20):
        masu.append([[i,j],1,0,0,0]) #マスに色の要素も持たせた


def use_1(masu1):
    random_color = random.sample(ads,1)[0]
    print1.append([1,masu1[0][0],masu1[0][1],random_color[3]])
    if masu1[1] == 1:
        for i in range(2,5):
            masu1[i] = random_color[i-2]
        masu.append(masu1)
        return 
    else:
        for i in range(2,5):
            masu1[i] = (masu1[i]*masu1[2] + random_color[i-2]) / (masu1[1]+1)
        masu1[1] +=1
        masu.append(masu1)
        return 

def use_2(masu1):
    print1.append([2,masu1[0][0],masu1[0][1]])
    masu1[1] -= 1
    masu.append(masu1)
    return 

def use_3(masu1):
    print1.append([3,masu1[0][0],masu1[0][1]])
    masu1[1] = 0
    masu.append(masu1)
    return

def search(masu1,masu2):
    if masu1[0][0] == masu2[0][0]:
        return 1
    else:
        return 0

def use_4(masu1,masu2):  #masu2を数字の大きい方と仮定
    print1.append([4,masu1[0][0],masu1[0][1],masu2[0][0],masu2[0][1]])
    for x in range(len(sikiri)):
        if sikiri[x][0] == masu2[0][0] and sikiri[x][1] == masu2[0][1] and masu2[1] == 1 and sikiri[x][3]==search(masu1,masu2):
            sikiri[x][2] = 0
            break 
        masu3 = []
        pre = []
        pre.append(masu1[0])
        pre.append(masu2[0])
        masu3.append(pre)
        masu3.append(masu1[1]+masu2[1])
        for i in range(2,4):
            masu3.append((masu1[1]*masu1[i] + masu2[1]*masu2[i])/(masu1[1]+masu2[1])) 
            masu.append(masu3)
        return 

for x in range(10000):
    #if time.time() * 1000 - start_time > 4000:
        #print(1)
        #break
    use1_masu = masu.pop(random.randint(0,len(masu)-1)) 
    if ans_want == []:
        print(2)
        break
        #preans = ans.pop(0)
    use_1(use1_masu)

for i in range(20):
    print(' '.join(['1']*19))

for i in range(19):
    print(' '.join(['1']*20))

for x in range(1000):
    target =masu[random.randint(0,len(masu)-1)]
    use_2(target)
    use_1(target)

for op in print1:
    print(' '.join(map(str, op)))
