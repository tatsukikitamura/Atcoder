#include <stdio.h>
#include <stdlib.h> // for calloc, free
#include <limits.h> // for INT_MAX

int main() {
    int a, b;
    
    // a (要素の数) と b (区間の数) を読み込む
    if (scanf("%d %d", &a, &b) != 2) {
        // 入力エラー処理 (競技プログラミングでは省略されることもあります)
        return 1;
    }

    // 要素の数が0の場合は、カバーされる要素がないため最小値は0
    if (a == 0) {
        printf("0\n");
        return 0;
    }

    // 1. 差分配列の準備 (imos法のため)
    // diff[i] は、位置 i で始まる区間の数と、位置 i-1 で終わる区間の数の差分を記録します。
    // サイズは a+1 とします。これは、区間が最後の要素 a (1-indexed) で終わる場合に、
    // diff[a] (0-indexed) にアクセスするためです。
    // calloc を使うと、メモリが0で初期化されます。
    int *diff = (int *)calloc(a + 1, sizeof(int));
    if (diff == NULL) {
        // メモリ割り当て失敗処理
        fprintf(stderr, "Memory allocation failed for diff array.\n");
        return 1;
    }

    // 2. 各区間の情報を差分配列に記録
    for (int i = 0; i < b; ++i) {
        int x, y;
        if (scanf("%d %d", &x, &y) != 2) { // x, y は 1-indexed の区間
            // 入力エラー処理
            free(diff);
            return 1;
        }
        
        // 区間の開始点 (0-indexed) に +1
        // x は 1-indexed なので、0-indexed では x-1
        diff[x - 1]++;
        
        // 区間の終了点の「次」の点 (0-indexed) に -1
        // y は 1-indexed の区間の終点。diff[y] は 0-indexed で y番目の要素を指す。
        // y の最大値は a であり、diff 配列のサイズは a+1 (インデックス 0 から a まで有効)。
        // したがって、diff[y] へのアクセスは安全です。
        diff[y]--;
    }

    int min_coverage = INT_MAX; // 各要素のカバー数の最小値を記録 (非常に大きな値で初期化)
    int current_coverage = 0;   // 現在の位置までの累積カバー数

    // 3. 差分配列から実際のカバー数を計算 (累積和) し、最小値を見つける
    // 0-indexed で 0 から a-1 までの各要素について計算
    for (int i = 0; i < a; ++i) {
        current_coverage += diff[i]; // 位置 i の差分を累積
        if (current_coverage < min_coverage) {
            min_coverage = current_coverage; // より小さいカバー数が見つかれば更新
        }
    }
    
    // 4. 結果の出力
    printf("%d\n", min_coverage);

    // 5. 動的に割り当てたメモリを解放
    free(diff);

    return 0;
}