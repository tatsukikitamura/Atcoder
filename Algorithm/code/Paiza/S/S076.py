from collections import deque

def solve():
        M, N = map(int, input().split())

        grid = []
        start_pos = None
        goal_pos = None

        for y in range(N):
            row = input().split()
            grid.append(row)
            
            for x in range(M):
                if row[x] == 's':
                    start_pos = (x, y) # (列, 行)
                elif row[x] == 'g':
                    goal_pos = (x, y) # (列, 行)

        
        start_x, start_y = start_pos
        goal_x, goal_y = goal_pos

        distances = [[-1] * M for _ in range(N)]

        queue = deque()

        queue.append(start_pos)
        distances[start_y][start_x] = 0

        moves = [
            (-1, 0), # 上
            (1, 0),  # 下
            (0, -1), # 左
            (0, 1)   # 右
        ]

        while queue:
            x, y = queue.popleft() 

            if (x, y) == goal_pos:
                print(distances[y][x]) 
                return

            for dy, dx in moves:
                nx, ny = x + dx, y + dy 

                if 0 <= nx < M and 0 <= ny < N:
                    if distances[ny][nx] == -1 and grid[ny][nx] != '1':    
                        distances[ny][nx] = distances[y][x] + 1        
                        queue.append((nx, ny))

        print("Fail") 



solve()