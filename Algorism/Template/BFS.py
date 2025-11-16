from collections import deque

class GridBFS:
    """
    グリッド上でのBFS（幅優先探索）を行うクラス
    S076.pyをクラス化したもの
    """
    
    def __init__(self, grid, start_char='s', goal_char='g', wall_char='1'):
        """
        Args:
            grid: 2次元リストのグリッド
            start_char: スタート位置を示す文字
            goal_char: ゴール位置を示す文字
            wall_char: 壁を示す文字
        """
        self.grid = grid
        self.M = len(grid[0]) if grid else 0  # 列数
        self.N = len(grid)  # 行数
        self.start_char = start_char
        self.goal_char = goal_char
        self.wall_char = wall_char
        
        # スタートとゴールの位置を探索
        self.start_pos = None
        self.goal_pos = None
        self._find_start_goal()
        
        # 移動方向（上下左右）
        self.moves = [
            (-1, 0),  # 上
            (1, 0),   # 下
            (0, -1),  # 左
            (0, 1)    # 右
        ]
    
    def _find_start_goal(self):
        """グリッドからスタートとゴールの位置を探索"""
        for y in range(self.N):
            for x in range(self.M):
                if self.grid[y][x] == self.start_char:
                    self.start_pos = (x, y)  # (列, 行)
                elif self.grid[y][x] == self.goal_char:
                    self.goal_pos = (x, y)  # (列, 行)
    
    def shortest_path(self):
        """
        スタートからゴールまでの最短距離を返す
        Returns:
            int: 最短距離。見つからない場合は-1
        """
        if self.start_pos is None or self.goal_pos is None:
            return -1
        
        distances = [[-1] * self.M for _ in range(self.N)]
        queue = deque()
        
        queue.append(self.start_pos)
        start_x, start_y = self.start_pos
        distances[start_y][start_x] = 0
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == self.goal_pos:
                return distances[y][x]
            
            for dy, dx in self.moves:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.M and 0 <= ny < self.N:
                    if distances[ny][nx] == -1 and self.grid[ny][nx] != self.wall_char:
                        distances[ny][nx] = distances[y][x] + 1
                        queue.append((nx, ny))
        
        return -1
    
    def solve(self):
        """
        S076.pyのsolve()関数と同じ動作
        最短距離を出力、見つからない場合は"Fail"を出力
        """
        result = self.shortest_path()
        if result == -1:
            print("Fail")
        else:
            print(result)


# 使用例（S076.pyのsolve()関数を再現）
def solve_s076():
    """S076.pyのsolve()関数を再現"""
    M, N = map(int, input().split())
    
    grid = []
    for y in range(N):
        row = input().split()
        grid.append(row)
    
    bfs = GridBFS(grid)
    bfs.solve()


if __name__ == "__main__":
    solve_s076()

