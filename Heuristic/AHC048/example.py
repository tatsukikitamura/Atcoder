sikiri =[]
for i in range(20):
    for j in range(19):
        sikiri.append([i,j,0]) #三番目が0の時縦
        
for i in range(20):
    for j in range(19):
        sikiri.append([j,i,1]) #三番目が1の時横
        

        


print(sikiri)