# AHC061 - THIRD Programming Contest 2026

AtCoder Heuristic Contest 061 用のワークスペースです。

## クイックスタート

### 1. 公式ツールのダウンロード

1. [コンテストページ](https://atcoder.jp/contests/ahc061) から公式ツール (Rust) をダウンロード
2. `tools/` ディレクトリに展開
3. ビルド：`cd tools && cargo build --release`

### 2. テストケースの生成

```bash
make gen
```

### 3. 解法の実装

`src/main.cpp` を編集して解法を実装してください。

---

## ディレクトリ構成

```
.
├── src/
│   └── main.cpp           # メイン提出ファイル (Timer/Random ユーティリティ付き)
│   └── experimental/      # 実験的な解法
├── tools/                  # 公式 Rust ツール (ビジュアライザ/ジェネレータ)
│   └── seeds.txt          # テストケース生成用のシード
├── testcases/
│   ├── in/                # 入力ファイル
│   └── out/               # 出力ファイル
├── scripts/
│   ├── test.py            # 並列テスト実行スクリプト
│   └── check.py           # クラッシュ検出スクリプト
├── submissions/           # スコア付きでアーカイブされた提出
├── docs/
│   └── PROBLEM.md         # 問題文
├── build/                 # ビルドされたバイナリ
├── tmp/                   # テスト中の一時出力
└── Makefile               # ビルド・テストコマンド
```

---

## Make コマンド

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
