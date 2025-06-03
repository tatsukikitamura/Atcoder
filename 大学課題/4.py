menu = ["ラーメン","ハンバーグ","パスタ"]
price = [800,1000,900]
menu_list = dict(zip(menu,price))
for key,value in menu_list.items():
    print(key,value)
print(f"ごうけいは{sum(menu_list.values())}円です")