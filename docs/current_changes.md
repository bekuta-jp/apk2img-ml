# Current Changes

このページは、`legacy` から `src/apk2img_ml/` へ移している現行実装の変更点をまとめたものです。

## 対象

- 運用コード: `src/apk2img_ml/`
- 保管コード: `src/apk2img-ml/legacy/`

`legacy` は変更していません。再構成と新規実装はすべて `src/apk2img_ml/` 側に追加しています。

## 今回の変更

### 1. `train_eval_mrun` を新パッケージへ移植

追加したファイル:

- `src/apk2img_ml/cnn/train_eval_mrun.py`
- `src/apk2img_ml/cnn/__init__.py`

CLI 追加:

- `python -m apk2img_ml train-eval-mrun ...`

### 2. 既存互換を維持したまま学習・評価処理を移設

以下の挙動は `legacy/train_eval_mrun.py` と互換です。

- `data_root/dev` を `8:2` で train/val に分割
- `data_root/test` で最終評価
- モデル選択: `tiny`, `alexnet`, `vgg16`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `densenet`, `densenet121`, `mobilenet`, `mobilenet_v2`, `efficientnet_b0`-`efficientnet_b7`, `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l`
- ログ出力:
  - `train_log.json`
  - `lr_curves.png`
  - `loss_curves.png`
  - `val_acc_curves.png`
- 指標出力:
  - Accuracy
  - F1 macro
  - F1 micro
  - F1 weighted
  - binary F1 (`malware`, `benign`)
  - per-class F1
  - classification report
  - confusion matrix

### 3. `--workers` の不整合を修正

`legacy` では `persistent_workers=True` が固定だったため、`--workers 0` で `DataLoader` が失敗する実装でした。

新実装では:

- `workers < 0`: 引数エラー
- `workers == 0`: 有効
- `persistent_workers`: `workers > 0` のときだけ有効

### 4. 任意チャネル数に対応

`legacy` の実装コメントでは任意チャネル対応をうたっていましたが、実際には `1/3/4` のみでした。

新実装では次の一般的な処理にしています。

- `1ch`: PIL `L`
- `2ch` と `3ch`: PIL `RGB`
- `4ch` 以上: PIL `RGBA`
- 変換後にチャンネル数を調整
  - 足りない場合: 先頭から繰り返して埋める
  - 多い場合: 先頭から必要数だけ使う

これにより、`1/3/4` の既存運用は維持しつつ、それ以外の入力も扱えます。

### 5. `tiny` モデルを任意入力サイズ対応に拡張

`legacy` の `tiny` は `256x256` 入力を前提に、最後の全結合層が `128 * 32 * 32` 固定でした。

新実装では、3段目のプーリング後に `AdaptiveAvgPool2d((32, 32))` を追加しています。

効果:

- `256x256` 入力では従来と同じ形状を維持
- それ以外の入力サイズでも全結合層へ接続可能

### 6. ResNet / EfficientNet 系モデルを追加

追加したモデル名:

- `resnet18`
- `resnet34`
- `resnet101`
- `resnet152`
- `efficientnet_b0`
- `efficientnet_b1`
- `efficientnet_b2`
- `efficientnet_b3`
- `efficientnet_b4`
- `efficientnet_b5`
- `efficientnet_b6`
- `efficientnet_b7`
- `efficientnet_v2_s`
- `efficientnet_v2_m`
- `efficientnet_v2_l`

既存互換として `resnet50`, `densenet`, `mobilenet` も引き続き使えます。`densenet121` と `mobilenet_v2` も明示名として使えます。

## 実行に必要なコマンド

### 1. 依存インストール

最小:

```bash
pip install -e .
```

CNN 学習・評価まで使う場合:

```bash
pip install -e ".[cnn]"
```

### 2. CLI ヘルプ確認

```bash
PYTHONPATH=src python3 -m apk2img_ml --help
PYTHONPATH=src python3 -m apk2img_ml train-eval-mrun --help
```

インストール済みなら:

```bash
apk2img-ml --help
apk2img-ml train-eval-mrun --help
```

### 3. CNN 学習・評価の実行例

```bash
PYTHONPATH=src python3 -m apk2img_ml train-eval-mrun \
  --data-root ./images256 \
  --model tiny \
  --epochs 15 \
  --batch 32 \
  --lr 1e-4 \
  --optimizer adam \
  --early-stopping-patience 3 \
  --workers 4 \
  --in-ch 1 \
  --resize 256,256 \
  --seed 3407 \
  --runs 3 \
  --log-dir ./results/train_eval_mrun
```

`./images256` の想定構成:

```text
images256/
  dev/
    benign/
    malware/
  test/
    benign/
    malware/
```

### 4. `--workers 0` を使う例

```bash
PYTHONPATH=src python3 -m apk2img_ml train-eval-mrun \
  --data-root ./images256 \
  --model resnet50 \
  --workers 0 \
  --in-ch 3 \
  --resize 256,256
```

### 5. 任意チャネル数を使う例

```bash
PYTHONPATH=src python3 -m apk2img_ml train-eval-mrun \
  --data-root ./images256 \
  --model tiny \
  --in-ch 5 \
  --resize 320,256
```

### 6. EfficientNet を使う例

```bash
PYTHONPATH=src python3 -m apk2img_ml train-eval-mrun \
  --data-root ./images256 \
  --model efficientnet_b0 \
  --in-ch 3 \
  --resize 224,224
```

### 7. Optuna で CNN ハイパラを探索する例

```bash
PYTHONPATH=src python3 -m apk2img_ml tune-cnn \
  --data-root ./images256 \
  --trials 20 \
  --epochs 15 \
  --models resnet18,resnet50,mobilenet_v2 \
  --batch-candidates 16,32,64 \
  --optimizer-candidates adam,adamw \
  --early-stopping-patience 3 \
  --log-dir ./results/optuna_cnn
```

`tune-cnn` は Optuna/TPE と MedianPruner を使い、`dev/` の train/val 分割で平均 best validation accuracy を最大化します。探索後は既定で最良 trial の設定を `test/` で評価します。

### 8. 結果集約ディレクトリ

実行結果は `results/` にまとめます。

- `results/train_eval_mrun/`: CNN の学習・評価ログ
- `results/optuna_cnn/`: CNN ハイパラ探索ログ
- `results/doc2vec/`: Doc2Vec 学習・推論の結果
- `results/images/`: ベクトル画像化後の確認用出力
- `results/notebooks/`: 実行結果の集計・可視化ノートブック

`train-eval-mrun` の結果を集約する場合は、`--log-dir ./results/train_eval_mrun` を指定してください。

## 注意点

- `--resize none` と `--batch > 1` を同時に使うと、画像サイズが混在するデータセットではバッチ化に失敗することがあります。
- 現在の環境では `torch`, `torchvision`, `matplotlib`, `scikit-learn` が未導入のため、実行確認はヘルプと構文チェックまでです。
- `legacy` と新実装の数値結果の同一性比較はまだ行っていません。
