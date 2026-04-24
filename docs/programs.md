# Programs

このページは `src/apk2img_ml/` の運用コードを対象にしています。

依存関係と環境構築は [setup.md](setup.md) を参照してください。

## CLI 一覧

共通実行形式:

```bash
python -m apk2img_ml <command> ...
```

## 個別マニュアル

- [`tune-cnn`](manuals/tune-cnn.md): 全オプションと探索動作の詳細

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
- `--pretrained` / `--no-pretrained`
- `--epochs`, `--batch`, `--workers`
- `--lr`, `--optimizer {adam,adamw,sgd}`, `--weight-decay`
- `--lr-scheduler {none,step,multistep,exponential,cosine,plateau,cosine_warm_restarts,onecycle}`
- scheduler 詳細: `--scheduler-step-size`, `--scheduler-milestones`, `--scheduler-gamma`, `--scheduler-exp-gamma`, `--scheduler-patience`, `--scheduler-t-max`, `--scheduler-eta-min`, `--scheduler-t-0`, `--scheduler-t-mult`, `--scheduler-pct-start`, `--scheduler-div-factor`, `--scheduler-final-div-factor`
- `--in-ch`
- `--resize`
- `--seed`, `--runs`
- `--log-dir`
- `--early-stopping-patience`, `--early-stopping-min-delta`, `--no-restore-best`

挙動:

- `dev/` を `8:2` で train/val に分割
- `test/` で最終評価
- `train_log.json`, `lr_curves.png`, `loss_curves.png`, `val_acc_curves.png` を保存
- `--lr-scheduler none` が既定値のため、初期設定では従来通り固定学習率
- `--early-stopping-patience > 0` の場合は validation accuracy を監視して停止し、既定では best epoch の重みに戻して `test/` 評価
- `--pretrained` が既定値で、torchvision の ImageNet 学習済み重みから学習します。`--no-pretrained` の場合はランダム初期化から学習します。

互換性と拡張:

- `legacy/train_eval_mrun.py` の基本挙動を維持
- `--workers 0` を有効化
- 任意チャネル数に対応
- `tiny` は `256x256` 互換を保ちながら任意入力サイズに対応

### `tune-cnn`

Optuna を使って CNN のハイパラを自動調整します。目的関数は `dev/` の train/val 分割における平均 best validation accuracy です。探索後は既定で最良 trial の設定を `test/` で評価します。

詳細は [`manuals/tune-cnn.md`](manuals/tune-cnn.md) を参照してください。

主オプション:

- `--data-root`
- `--trials`, `--epochs`, `--runs`
- `--pretrained` / `--no-pretrained`
- `--models`
- `--batch-candidates`
- `--optimizer-candidates`
- `--lr-low`, `--lr-high`
- `--weight-decay-candidates`
- `--lr-scheduler` と scheduler 詳細オプション
- `--early-stopping-patience`, `--early-stopping-min-delta`
- `--pruner-startup-trials`, `--pruner-warmup-epochs`
- `--evaluate-best` / `--no-evaluate-best`
- `--per-model`
- `--study-name`, `--storage`

探索モード:

- 通常モードでは `--models` の候補を1つの Optuna study 内で探索し、モデル名も trial parameter として扱います。
- `--per-model` 付きでは `--models` の各モデルごとに独立した study を実行し、モデルごとの最良ハイパラと全体の best model を保存します。
- `--pretrained` / `--no-pretrained` は探索対象ではなく、全 trial に共通の固定条件として適用されます。

モデル別探索の例:

```bash
python -m apk2img_ml tune-cnn \
  --data-root ./images256 \
  --trials 20 \
  --epochs 15 \
  --models resnet18,resnet50 \
  --batch-candidates 16,32,64 \
  --optimizer-candidates adam,adamw \
  --early-stopping-patience 3 \
  --per-model \
  --log-dir ./results/optuna_cnn_by_model
```

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
