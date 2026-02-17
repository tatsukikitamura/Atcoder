# AHC Multi-Player Territory Game - Optuna Tuning

## ファイル構成

```
scripts/
  optuna_tune.py     # Optunaチューニングスクリプト
  embed_params.py    # 最適パラメータをC++に埋め込む提出用スクリプト
src/main.cpp         # メインソリューション (configファイル対応)
```

## セットアップ

### 1. Python依存パッケージ
```bash
pip install optuna
```

### 2. ビルド
```bash
make          # src/main.cpp → build/main.exe
```

### 3. テスタ (初回のみ)
```bash
cd tools && cargo build -r    # tester.exe ビルド
```

## ワークフロー

**全コマンドはプロジェクトルート (`A/`) で実行**

### Step 1: デフォルトパラメータで動作確認
```bash
make vis CASE=0000.txt
```

### Step 2: チューニング実行
```bash
# 基本 (逐次実行、100トライアル)
python scripts/optuna_tune.py --tune --n_trials 100

# 高速テスト (テストケース数を制限)
python scripts/optuna_tune.py --tune --n_trials 50 --max_cases 20

# M/Uカテゴリ均等最適化 (推奨)
python scripts/optuna_tune.py --tune --n_trials 200 --strategy stratified
```

### Step 3: 結果確認
```bash
# 最良パラメータを表示
python scripts/optuna_tune.py --show_best

# パラメータファイルとして出力
python scripts/optuna_tune.py --export best_params.cfg

# 特定パラメータで全テストケースを評価
python scripts/optuna_tune.py --eval best_params.cfg
python scripts/optuna_tune.py --eval best_params.cfg --max_cases 10
```

### Step 4: 提出用コード生成
```bash
# パラメータをC++に埋め込み
python scripts/embed_params.py best_params.cfg src/main.cpp > submissions/main_tuned.cpp
```

## チューニング戦略

### `global` (デフォルト)
全テストケースの平均スコアを最大化。

### `stratified` (推奨)
テストケースを (M, U) カテゴリに分類し、カテゴリ毎の平均を均等に扱う。

## パラメータの意味

### フェーズ制御
| パラメータ | 説明 | 範囲 |
|---|---|---|
| `phase1` | 序盤→中盤の切り替え時点 | 0.15-0.45 |
| `phase2` | 中盤→終盤の切り替え時点 | 0.45-0.85 |

### 各フェーズの行動重み
| 接尾辞 | 意味 |
|---|---|
| `wa_*` | 空きマスの占領重み |
| `wb_*` | 自領土強化の重み |
| `wc_*` | 敵レベル1領土の攻撃重み |
| `wd_*` | 敵レベル2+領土の攻撃重み |

### 戦略パラメータ
| パラメータ | 説明 |
|---|---|
| `leader_mult` | 首位AIの領土を攻撃する際の評価倍率 |
| `ucb_c` | MCTS探索のUCB定数 (大=探索、小=活用) |

### M/U適応
| パラメータ | 説明 |
|---|---|
| `u_wb_boost` | U>2の時に強化重みに加算する量 |
| `u_wd_penalty` | U>2の時に高レベル攻撃重みから減算する量 |
| `m_leader_scale` | M増加時のleader_mult加算率 |

## ヒント

- **初回は少ないテストケース(10-20)で素早く探索**：`--max_cases 20 --n_trials 50`
- 良い領域を見つけてからテストケースを増やして精密化
- DBファイル (`optuna_ahc.db`) を消さなければ、前回の結果を引き継いで最適化を継続可能
- Optuna Dashboard: `optuna-dashboard sqlite:///optuna_ahc.db`
