def solve():
    try:
        N = int(input())
    except IndexError:
        return

    board = []
    for _ in range(N):
        row = input().strip()
        if len(row) != N:
            return
        board.append(list(row))

    DIRECTIONS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    def is_valid_coord(r, c):
        return 0 <= r < N and 0 <= c < N

    def is_placeable(r, c):

        if board[r][c] != '.':
            return False

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc

            white_count = 0

            while is_valid_coord(nr, nc):
                current_cell = board[nr][nc]

                if current_cell == 'W':
                    white_count += 1
                    nr += dr
                    nc += dc
                elif current_cell == 'B':
                    if white_count > 0:
                        return True
                    else:
                        break
                elif current_cell == '.':
                    break
                else:
                    break

        return False

    placeable_count = 0
    for r in range(N):
        for c in range(N):
            if is_placeable(r, c):
                placeable_count += 1

    print(placeable_count)


solve()
