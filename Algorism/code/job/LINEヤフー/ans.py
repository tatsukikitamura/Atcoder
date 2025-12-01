import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

# --- ヘルパー関数 ---


def parse_time(time_str: str) -> datetime:
    """時刻文字列をdatetimeオブジェクトに変換する"""
    return datetime.strptime(time_str, "%Y/%m/%d-%H:%M:%S")


def calculate_margin(sales_sum: int, cost_sum: int) -> float:
    """利益率を計算する (S - C) / S"""
    if sales_sum == 0:
        return 0.0
        # floatなので0.0を返す
    return (sales_sum - cost_sum) / sales_sum


def format_margin(value: float) -> str:
    """
    利益率を出力形式に合わせてフォーマットする
    小数点以下第四位で四捨五入し、小数点以下第三位まで出力
    """
    # Decimalを使って正確に四捨五入を行う
    d = Decimal(str(value))
    rounded = d.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return "{:.3f}".format(rounded)

# --- クラス定義 ---
# SaleRecord,Seller,Item,Permission,Salesystem


class SaleRecord:
    def __init__(self, timestamp: datetime, price: int, cost: int) -> None:
        self.timestamp = timestamp
        self.price = price
        self.cost = cost


class Seller:
    def __init__(self, seller_id: int, name: str) -> None:
        self.id = seller_id
        self.name = name
        self.sales_history = []  # List of SaleRecord
        self.active_permission_id = None  # 直近の有効な許可ID

    def get_stats(self, start_time: datetime | None = None, end_time: datetime | None = None) -> tuple[int, int]:
        total_sales = 0
        total_cost = 0
        for sale in self.sales_history:
            if start_time and end_time:
                # 期間指定がある場合（境界含む）
                if start_time <= sale.timestamp <= end_time:
                    total_sales += sale.price
                    total_cost += sale.cost
            else:
                # 全期間
                total_sales += sale.price
                total_cost += sale.cost
        return total_sales, total_cost


class Item:
    def __init__(self, item_id: int, name: str, cost: int, price: int) -> None:
        self.id = item_id
        self.name = name
        self.current_cost = cost
        self.current_price = price
        self.is_deleted = False
        self.sales_history = []  # List of SaleRecord

    def get_stats(self, start_time: datetime | None = None, end_time: datetime | None = None) -> tuple[int, int]:
        total_sales = 0
        total_cost = 0
        for sale in self.sales_history:
            if start_time and end_time:
                if start_time <= sale.timestamp <= end_time:
                    total_sales += sale.price
                    total_cost += sale.cost
            else:
                total_sales += sale.price
                total_cost += sale.cost
        return total_sales, total_cost


class Permission:
    def __init__(self, perm_id: int, seller_id: int, item_id: int, price: int, cost_snapshot: int, timestamp: datetime) -> None:
        self.id = perm_id
        self.seller_id = seller_id
        self.item_id = item_id
        self.price = price
        self.cost_snapshot = cost_snapshot  # 販売時点での原価を記録しておく
        self.timestamp = timestamp
        self.is_active = True

# --- メインシステム ---


class SalesSystem:
    def __init__(self):
        self.rate_threshold = 0.0
        self.sellers = {}  # id -> Seller
        self.items = {}  # id -> Item
        self.item_name_map = {}  # name -> id
        self.permissions = {}  # id -> Permission

        self.next_item_id = 1
        self.next_permission_id = 1

    def run(self) -> None:
        # 1. 基本設定の読み込み
        try:
            line1 = sys.stdin.readline().strip()
            if not line1:
                return
            self.rate_threshold = float(line1)

            line2 = sys.stdin.readline().strip()
            n = int(line2)

            for i in range(1, n + 1):
                name = sys.stdin.readline().strip()
                self.sellers[i] = Seller(i, name)

            line_m = sys.stdin.readline().strip()
            m = int(line_m)

            # 2. クエリ処理
            for _ in range(m):
                line = sys.stdin.readline().strip()
                if not line:
                    break
                parts = line.split()
                command = parts[0]
                args = parts[1:]
                # それぞれの最初の文字により場合分け
                if command == 'register-item:':
                    self.handle_register_item(args)
                elif command == 'request-sale:':
                    self.handle_request_sale(args)
                elif command == 'complete-sale:':
                    self.handle_complete_sale(args)
                elif command == 'delete-item:':
                    self.handle_delete_item(args)
                elif command == 'update-item:':
                    self.handle_update_item(args)
                elif command == 'get-margin-sellers:':
                    self.handle_get_margin_sellers(args)
                elif command == 'get-margin-items:':
                    self.handle_get_margin_items(args)

        except ValueError as e:
            print(f"Error: Invalid input format - {e}", file=sys.stderr)
            return
        except EOFError:
            # 標準入力が途中で終了した場合、静かに終了
            return

    # --- クエリハンドラ ---

    def handle_register_item(self, args: list[str]) -> None:
        # args: [時刻, 商品名, 商品原価, 商品定価]
        # args[0]は使わない
        name = args[1]
        cost = int(args[2])
        price = int(args[3])

        # 1. 重複チェック
        if name in self.item_name_map:
            existing_id = self.item_name_map[name]
            if not self.items[existing_id].is_deleted:
                print("register-item: duplicated item")
                return

        # 2. 定価販売時の利益率チェック
        margin = calculate_margin(price, cost)
        # 浮動小数点計算の誤差を考慮し、比較には注意が必要だが、
        # 問題文の入力は整数、rateは小数点第2位まで。
        if margin < self.rate_threshold:  # わずかな誤差許容
            print("register-item: too cheap price")
            return

        # 登録処理
        new_id = self.next_item_id
        self.next_item_id += 1

        item = Item(new_id, name, cost, price)
        self.items[new_id] = item
        self.item_name_map[name] = new_id

        print(f"register-item: {new_id}")

    def handle_request_sale(self, args: list[str]) -> None:
        # args: [時刻, 販売員ID, 商品ID, 販売価格]
        timestamp = parse_time(args[0])
        seller_id = int(args[1])
        item_id = int(args[2])
        price = int(args[3])

        # 1. 商品存在チェック
        if item_id not in self.items or self.items[item_id].is_deleted:
            print("request-sale: no such item")
            return

        item = self.items[item_id]

        # 2. 定価より高いかチェック
        if price > item.current_price:
            print("request-sale: too expensive price")
            return
        # 3. 定価未満の場合の条件チェック
        elif price < item.current_price:
            # 原価割れチェック
            if price < item.current_cost:
                print("request-sale: too cheap price")
                return
            # 仮想的な利益率チェック

            # 販売員の見込み利益率
            seller = self.sellers[seller_id]
            s_sales, s_cost = seller.get_stats()  # 現在までの実績
            s_sales += price
            s_cost += item.current_cost
            s_margin = calculate_margin(s_sales, s_cost)

            if s_margin < self.rate_threshold:
                print("request-sale: too cheap price")
                return

            # 商品の見込み利益率
            i_sales, i_cost = item.get_stats()
            i_sales += price
            i_cost += item.current_cost
            i_margin = calculate_margin(i_sales, i_cost)

            if i_margin < self.rate_threshold:
                print("request-sale: too cheap price")
                return
        # 同じ値段の場合は許可
        # 許可発行
        perm_id = self.next_permission_id
        self.next_permission_id += 1

        # 該当販売員の以前の許可を失効させる（上書き）
        # 仕様：「その販売の完了がシステムに登録される前に、同じ販売員から次の販売要求があったとき」
        seller = self.sellers[seller_id]
        if seller.active_permission_id is not None:
            old_perm = self.permissions[seller.active_permission_id]
            old_perm.is_active = False

        # 新しい許可を作成（価格変更の影響を受けないよう、この時点の原価をスナップショットする）
        perm = Permission(perm_id, seller_id, item_id, price, item.current_cost, timestamp)
        self.permissions[perm_id] = perm
        seller.active_permission_id = perm_id

        print(f"request-sale: {perm_id}")

    def handle_complete_sale(self, args: list[str]) -> None:
        # args: [時刻, 販売員ID, 販売許可ID]
        timestamp = parse_time(args[0])
        seller_id = int(args[1])
        perm_id = int(args[2])

        # 1. 許可ID存在チェック
        if perm_id not in self.permissions:
            print("complete-sale: no such sale")
            return

        perm = self.permissions[perm_id]

        # 2. 販売員一致チェック
        if perm.seller_id != seller_id:
            print("complete-sale: unauthorized operation")
            return

        # 3. 失効チェック
        # 条件A: すでに is_active が False (他のリクエストで上書きされた、または完了済み)
        if not perm.is_active:
            print("complete-sale: permission expired")
            return

        # 条件B: 日付が変わっている
        if perm.timestamp.date() != timestamp.date():
            perm.is_active = False  # 失効させる
            print("complete-sale: permission expired")
            return

        # 販売完了登録
        perm.is_active = False  # 使用済み

        # 履歴に追加
        sale_rec = SaleRecord(timestamp, perm.price, perm.cost_snapshot)

        self.sellers[seller_id].sales_history.append(sale_rec)
        self.items[perm.item_id].sales_history.append(sale_rec)

        # 販売員のactive_permをクリア（もしこれが最新なら）
        if self.sellers[seller_id].active_permission_id == perm_id:
            self.sellers[seller_id].active_permission_id = None

        print("complete-sale: ok")

    def handle_delete_item(self, args: list[str]) -> None:
        # args: [時刻, 商品ID]
        timestamp = parse_time(args[0])
        item_id = int(args[1])

        # 1. 存在チェック
        if item_id not in self.items or self.items[item_id].is_deleted:
            print("delete-item: no such item")
            return

        # 2. 進行中の販売許可チェック
        # 全ての有効な許可をチェックし、対象商品IDが含まれるか確認
        # activeなものだけ見ればよい
        has_active_sale = False
        for perm in self.permissions.values():
            if perm.is_active and perm.item_id == item_id:
                # 日付が変わっていないか確認（削除クエリ時点で日付が変わっていれば、その許可は実質無効）
                if perm.timestamp.date() == timestamp.date():
                    has_active_sale = True
                    break

        if has_active_sale:
            print("delete-item: sales in progress")
            return

        # 削除実行
        item = self.items[item_id]
        item.is_deleted = True
        # 名前マップから削除（同名での再登録を許可するため）
        if item.name in self.item_name_map and self.item_name_map[item.name] == item_id:
            del self.item_name_map[item.name]

        print("delete-item: ok")

    def handle_update_item(self, args: list[str]) -> None:
        # args: [時刻, 商品ID, 商品原価, 商品定価]
        timestamp = parse_time(args[0])
        item_id = int(args[1])
        new_cost = int(args[2])
        new_price = int(args[3])

        # 1. 存在チェック
        if item_id not in self.items or self.items[item_id].is_deleted:
            print("update-item: no such item")
            return

        # 2. 進行中の販売許可チェック (Deleteと同じロジック)
        has_active_sale = False
        for perm in self.permissions.values():
            if perm.is_active and perm.item_id == item_id and perm.timestamp.date() == timestamp.date():
                has_active_sale = True
                break

        if has_active_sale:
            print("update-item: sales in progress")
            return

        # 3. 新価格での利益率チェック
        margin = calculate_margin(new_price, new_cost)
        if margin < self.rate_threshold:
            print("update-item: too cheap price")
            return

        # 更新実行
        item = self.items[item_id]
        item.current_cost = new_cost
        item.current_price = new_price

        print("update-item: ok")

    def handle_get_margin_sellers(self, args: list[str]) -> None:
        # args: [時刻, 期間始点, 期間終点]
        # args[0]は使わない
        start_time = parse_time(args[1])
        end_time = parse_time(args[2])

        if start_time > end_time:
            print("get-margin-sellers: invalid time period")
            return

        results = []
        for s_id, seller in self.sellers.items():
            sales, cost = seller.get_stats(start_time, end_time)
            margin = calculate_margin(sales, cost)
            results.append({
                'id': s_id,
                'name': seller.name,
                'margin': margin
            })

        # ソート: 利益率降順 -> ID昇順
        # Pythonのsortは安定ソートなので、第2キーから順にソートするか、タプルで指定する
        # タプル: (-margin, id) ※降順にするためマイナスをつける
        results.sort(key=lambda x: (-x['margin'], x['id']))

        print("get-margin-sellers:")
        for res in results:
            print(f"{res['id']} {res['name']} {format_margin(res['margin'])}")

    def handle_get_margin_items(self, args: list[str]) -> None:
        # args: [時刻, 期間始点, 期間終点]
        # args[0]は使わない
        start_time = parse_time(args[1])
        end_time = parse_time(args[2])

        if start_time > end_time:
            print("get-margin-items: invalid time period")
            return

        results = []
        # クエリが送信された時点で登録されており削除されていない全ての商品
        for i_id, item in self.items.items():
            if not item.is_deleted:
                sales, cost = item.get_stats(start_time, end_time)
                margin = calculate_margin(sales, cost)
                results.append({
                    'id': i_id,
                    'name': item.name,
                    'margin': margin
                })

        # ソート: 利益率降順 -> ID昇順
        results.sort(key=lambda x: (-x['margin'], x['id']))

        print(f"get-margin-items: {len(results)}")
        for res in results:
            print(f"{res['id']} {res['name']} {format_margin(res['margin'])}")


if __name__ == "__main__":
    sales_system = SalesSystem()
    sales_system.run()
