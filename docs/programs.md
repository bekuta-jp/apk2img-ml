# Programs

このページは `src/apk2img_ml/` の運用コードを対象にしています。

依存関係と環境構築は [setup.md](setup.md) を参照してください。

## CLI 一覧

共通実行形式:

```bash
python -m apk2img_ml <command> ...
```

### `extract`

APK から API トークンを抽出します。

- 主系: Androguard
- フォールバック: baksmali, apktool

主オプション:

- `--backend {hybrid,androguard,baksmali,apktool}`
- `--input-dir`, `--output-dir`
- `--recursive`, `--workers`, `--overwrite`
- `--baksmali-jar`, `--apktool-cmd`
- `--include-prefix`, `--exclude-prefix`, `--include-signature`

### `train-doc2vec`

抽出済みコーパスから Doc2Vec を学習します。

主オプション:

- `--corpus-dir`
- `--model-out`
- `--docvecs-out`
- `--vector-size`, `--window`, `--min-count`, `--epochs`, `--workers`

### `infer-doc2vec`

学習済み Doc2Vec を使って推論ベクトルを出力します。

主オプション:

- `--corpus-dir`
- `--model`
- `--docvecs-out`
- `--infer-epochs`, `--infer-alpha`, `--workers`

### `docvec-to-png`

DocVec TSV をグレースケール画像へ変換します。

主オプション:

- `--docvecs-tsv`
- `--output-dir`
- `--mode {flat,grid,raw32}`
- `--grid-rows`, `--grid-cols`, `--vec-pix-rows`, `--vec-pix-cols`

### `train-eval-mrun`

画像フォルダを使って CNN の学習・検証・テストを複数回実行します。

主オプション:

- `--data-root`
- `--model {tiny,alexnet,vgg16,resnet18,resnet34,resnet50,resnet101,resnet152,densenet,densenet121,mobilenet,mobilenet_v2,efficientnet_b0-b7,efficientnet_v2_s,efficientnet_v2_m,efficientnet_v2_l}`
- `--epochs`, `--batch`, `--workers`
- `--lr`, `--optimizer {adam,adamw,sgd}`, `--weight-decay`
- `--in-ch`
- `--resize`
- `--seed`, `--runs`
- `--log-dir`
- `--early-stopping-patience`, `--early-stopping-min-delta`, `--no-restore-best`

挙動:

- `dev/` を `8:2` で train/val に分割
- `test/` で最終評価
- `train_log.json`, `lr_curves.png`, `loss_curves.png`, `val_acc_curves.png` を保存
- `--early-stopping-patience > 0` の場合は validation accuracy を監視して停止し、既定では best epoch の重みに戻して `test/` 評価

互換性と拡張:

- `legacy/train_eval_mrun.py` の基本挙動を維持
- `--workers 0` を有効化
- 任意チャネル数に対応
- `tiny` は `256x256` 互換を保ちながら任意入力サイズに対応

### `tune-cnn`

Optuna を使って CNN のハイパラを自動調整します。目的関数は `dev/` の train/val 分割における平均 best validation accuracy です。探索後は既定で最良 trial の設定を `test/` で評価します。

主オプション:

- `--data-root`
- `--trials`, `--epochs`, `--runs`
- `--models`
- `--batch-candidates`
- `--optimizer-candidates`
- `--lr-low`, `--lr-high`
- `--weight-decay-candidates`
- `--early-stopping-patience`, `--early-stopping-min-delta`
- `--pruner-startup-trials`, `--pruner-warmup-epochs`
- `--evaluate-best` / `--no-evaluate-best`
- `--study-name`, `--storage`

## モジュール構成

- `apk2img_ml.extraction`
  - `androguard_backend.py`
  - `baksmali_backend.py`
  - `apktool_backend.py`
  - `pipeline.py` (ハイブリッド制御)
- `apk2img_ml.embedding`
  - `corpus.py`
  - `doc2vec.py`
- `apk2img_ml.imaging`
  - `docvec_png.py`
- `apk2img_ml.cnn`
  - `models.py`
  - `train_eval_mrun.py`

## Legacy について

- `src/apk2img-ml/legacy/` は非推奨ではなく「過去実験の保管」です。
- 再現が必要な場合のみ参照し、新規開発は `src/apk2img_ml/` を使用してください。
