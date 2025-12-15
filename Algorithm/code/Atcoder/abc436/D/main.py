import sys
from collections import deque, defaultdict
input = sys.stdin.readline
def main():
    H, W = map(int, input().split())
    warp_dict = defaultdict(list)
    start_pos = (0, 0)
    goal_pos = (H - 1, W - 1)
    for r in range(H):
        row_str = input().rstrip()
        grid.append(row_str)
        for c, char in enumerate(row_str):
                warp_dict[char].append((r, c))

    dist = [[-1] * W for _ in range(H)]
    dist[0][0] = 0
    queue = deque([start_pos])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal_pos:
            return
        current_char = grid[r][c]
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] != '#' and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    queue.append((nr, nc))
        if current_char.islower():
            for wr, wc in warp_dict[current_char]:
                if dist[wr][wc] == -1:
                    dist[wr][wc] = dist[r][c] + 1
                    queue.append((wr, wc))
            del warp_dict[current_char]
    print(-1)

if __name__ == '__main__':
    main()