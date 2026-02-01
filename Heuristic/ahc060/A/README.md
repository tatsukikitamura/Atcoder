# Heuristic Contest Template

AtCoder Heuristic Contest (AHC) 用のテンプレートです。

## 🚀 クイックスタート

### 1. ワークスペースのセットアップ

新しいコンテスト用のディレクトリを作成：

```bash
# 例: AHC056 の場合
cp -r /Users/kitamuratatuki/Atcoder/.agent/templates/heuristic Heuristic/AHC056/A
cd Heuristic/AHC056/A
```

または、Gemini に `/setup-heuristic-contest` を実行させてください。

### 2. 公式ツールのダウンロード

1. コンテストページから公式ツール (Rust) をダウンロード
2. `tools/` ディレクトリに展開
3. ビルド：`cd tools && cargo build --release`

### 3. テストケースの生成

```bash
make gen
```

### 4. 解法の実装

`src/main.cpp` を編集して解法を実装してください。

---

## 📁 ディレクトリ構成

```
.
├── src/
│   └── main.cpp           # メイン提出ファイル (Timer/Random ユーティリティ付き)
├── tools/                  # 公式 Rust ツール (ビジュアライザ/ジェネレータ)
│   └── seeds.txt          # テストケース生成用のシード
├── testcases/
│   ├── in/                # 入力ファイル
│   └── out/               # 出力ファイル
├── scripts/
│   ├── test.py            # 並列テスト実行スクリプト
│   └── check.py           # クラッシュ検出スクリプト
├── submissions/           # スコア付きでアーカイブされた提出
├── build/                 # ビルドされたバイナリ
├── tmp/                   # テスト中の一時出力
└── Makefile               # ビルド・テストコマンド
```

---

## 🔧 Make コマンド

| コマンド | 説明 |
|---------|------|
| `make` | メインソルバをビルド |
| `make vis` | ソルバを実行してビジュアライザを開く |
| `make gen` | テストケースを生成 |
| `make test` | 全テストケースを実行（並列処理） |
| `make fast` | シード 0-9 でクイックテスト |
| `make check` | クラッシュ検出 |
| `make submit` | スコア計算、アーカイブ、クリップボードにコピー |
| `make clean` | ビルド成果物を削除 |

---

## 📝 テンプレートの使い方

### main.cpp の構成

```cpp
class Timer {
    // 制限時間内での最適化用タイマー
    bool has_time() const;  // 残り時間があるか確認
    double elapsed() const; // 経過時間を取得
};

class Random {
    // 乱数生成器
    int randint(int lo, int hi);    // [lo, hi] の整数
    double uniform(double lo, hi);   // [lo, hi] の実数
};

class Solver {
    void read_input();  // 入力読み込み
    void solve();       // 解法のメインロジック
    void output();      // 出力
};
```

### 実装の流れ

1. `read_input()` で問題固有の入力を読み込み
2. `solve()` で解法を実装
   - 貪欲法で初期解を作成
   - 局所探索 / 焼きなまし法などで改善
3. `output()` で解を出力

### デバッグ出力

```cpp
cerr << "debug: " << value << endl;  // スコアに影響なし
```

---

## 🧪 テスト・検証

### 基本的なテスト

```bash
# ビルド
make

# サンプル入力でテスト（tmp/input.txt に入力を用意）
make vis

# 全テストケース実行
make test

# 高速テスト（最初の10ケースのみ）
make fast
```

### テストスクリプトのオプション

```bash
# 最初の N ケースだけテスト
python3 scripts/test.py -n 20

# 並列ワーカー数を指定
python3 scripts/test.py -j 8
```

---

## 🏆 提出

```bash
make submit
```

これにより：
1. 全テストケースを実行してスコアを計算
2. `submissions/main_<スコア>.cpp` としてアーカイブ
3. ソースコードをクリップボードにコピー

---

## 💡 Tips

### 焼きなまし法のテンプレート

```cpp
void solve() {
    Timer timer(1.9);
    Random rng(42);
    
    // 初期解を生成
    auto state = initial_solution();
    int best_score = evaluate(state);
    
    while (timer.has_time()) {
        // 温度計算
        double t = timer.elapsed() / timer.time_limit;
        double temp = START_TEMP * pow(END_TEMP / START_TEMP, t);
        
        // 近傍解を生成
        auto new_state = neighbor(state, rng);
        int new_score = evaluate(new_state);
        
        // 遷移判定
        int delta = new_score - best_score;
        if (delta > 0 || rng.uniform() < exp(delta / temp)) {
            state = new_state;
            if (new_score > best_score) {
                best_score = new_score;
            }
        }
    }
}
```

### よく使う定数

```cpp
const int DX[] = {0, 1, 0, -1};
const int DY[] = {1, 0, -1, 0};
const string DIR = "RDLU";
```

---

## 📚 関連ドキュメント

- [焼きなまし法ガイド](../../../.agent/workflows/Annealing_method.md)
- [改善ループ](../../../.agent/workflows/ahc-improvement-loop.md)
