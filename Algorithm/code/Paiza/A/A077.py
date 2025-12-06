class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.nears = []
    def __repr__(self):
        return f"(row:{self.row}, col:{self.col}, nears:{self.nears})"

N,R = map(int,input().split())
LIST = []
for _ in range(N):
    LIST.append(list(map(int,input().split())))

def check(x:list,y:list):
    if ((x[0]-y[0])**2 + (x[1]-y[1])**2 <= 4*R**2) and x != y:
        return True
    else:
        return False


nodes = []

for x in range(N):
    nodes.append(Node(LIST[x][0],LIST[x][1]))

stack = []

for node in nodes:
    for x in range(N):
        if check([node.row,node.col],LIST[x]):
            node.nears.append(LIST[x])

already_used = []
        

stack.append([nodes[0].row,nodes[0].col])


count = 1
while LIST != []:
    if stack == []:
        stack.append(LIST[0])
        count += 1
    use = stack.pop(0)
    already_used.append(use)
    
    LIST.remove(use)
    for node in nodes:
        if node.row == use[0] and node.col == use[1]:
            for near in node.nears:
                if near not in already_used and near not in stack:
                    stack.append(near)
            break
   
    

print(count)


