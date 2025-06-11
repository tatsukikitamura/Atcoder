import time
import random

start_time = time.time() * 1000  # 開始時刻

N = int(input())
MAX_LW = 10000  # スペースの縦と横
ads = [[] for i in range(N)]  # 入力値保存
ans = [[0, 0, 0, 0] for i in range(N)]  # 広告毎の配置座標

for i in range(N):
    x, y, r = map(int, input().split())
    ads[i] = [x, y, r]
    ans[i] = [x, y, x + 1, y + 1]  # 1 × 1を初期値とする

while True:
    # 結果出力のための100msecを余して終了(ギリギリを攻めすぎるとオーバーするので要注意)
    if time.time() * 1000 - start_time > 4900:
        break

    ad = random.randrange(N)  # 広告をランダムに選ぶ
    direction = random.randrange(4)  # 方向をランダムに選ぶ 上右下左

    # 現在の座標(x1,y1,x2,y2)と伸ばした後の座標(x1d,x2d,y1d,y2d)
    x1, y1, x2, y2 = ans[ad][0], ans[ad][1], ans[ad][2], ans[ad][3]
    x1d, y1d, x2d, y2d = x1, y1, x2, y2

    if direction == 0:  # 上
        y1d -= 1
    elif direction == 1:  # 右
        x2d += 1
    elif direction == 2:  # 下
        y2d += 1
    elif direction == 3:  # 左
        x1d -= 1

    # 伸ばす前後の満足度判定
    before_s = (x2 - x1) * (y2 - y1)
    after_s = (x2d - x1d) * (y2d - y1d)
    before_score = 1 - (1 - min([before_s, ads[ad][2]]) / max([before_s, ads[ad][2]])) ** 2
    after_score = 1 - (1 - min([after_s, ads[ad][2]]) / max([after_s, ads[ad][2]])) ** 2
    if before_score >= after_score:
        continue

    # はみ出ていないか判定
    if x1d < 1 or y1d < 1 or x2d > MAX_LW or y2d > MAX_LW:
        continue

    # 重なっていないか判定
    overlap = False
    for i in range(0, N):
        if i == ad:
            continue
        elif (ans[i][0] < x1d and ans[i][2] < x1d) or (ans[i][0] > x2d and ans[i][2] > x2d):
            continue
        elif (ans[i][1] < y1d and ans[i][3] < y1d) or (ans[i][1] > y2d and ans[i][3] > y2d):
            continue
        overlap = True

    # 座標の更新
    if not overlap:
        ans[ad] = [x1d, y1d, x2d, y2d]

# 出力
for i in range(N):
    print(*(ans[i]))
