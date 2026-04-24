# tune-cnn Manual

`tune-cnn` は Optuna を使って CNN のハイパラを探索するコマンドです。

共通実行形式:

```bash
python -m apk2img_ml tune-cnn ...
```

## 概要

- `data-root/dev/` を train/validation に `8:2` 分割して探索します
- 目的関数は各 run の `best validation accuracy` の平均です
- 既定では探索後に最良 trial を `data-root/test/` で再評価します
- 通常モードでは `model` も trial parameter として探索します
- `--per-model` 付きではモデルごとに独立した study を実行します

想定ディレクトリ構成:

```text
<data-root>/
  dev/
    benign/
    malware/
  test/
    benign/
    malware/
```

## 実行フロー

1. `dev/` を `8:2` で train/validation に分割します
2. trial ごとに候補モデルと候補ハイパラを選びます
3. 各 run で学習し、validation accuracy を監視します
4. run ごとの best validation accuracy を平均し、その値を trial の評価値にします
5. Optuna が最良 trial を選択します
6. `--evaluate-best` が有効なら、最良 trial の設定で `test/` を使った再評価を行います

## 探索対象

通常モードで探索対象になるのは次です。

- `model`
- `batch`
- `optimizer`
- `lr`
- `weight_decay`

次は探索対象ではなく、全 trial に共通の固定条件です。

- `--pretrained` / `--no-pretrained`
- `--in-ch`
- `--resize`
- `--lr-scheduler` と scheduler 詳細オプション
- early stopping 関連オプション
- pruning 関連オプション
- `--runs`

`--per-model` を付けた場合は `model` は study ごとに固定され、各モデル内で `batch`, `optimizer`, `lr`, `weight_decay` を探索します。

## 全オプション

### 必須

- `--data-root PATH`
  - `dev/` と `test/` を含む画像データのルートディレクトリです。

### 実行制御

- `--trials INT`
  - 既定値: `20`
  - 実行する trial 数です。
- `--epochs INT`
  - 既定値: `15`
  - 各 trial の最大 epoch 数です。
- `--workers INT`
  - 既定値: `4`
  - DataLoader の worker 数です。`0` も指定できます。
- `--seed INT`
  - 既定値: `3407`
  - 乱数シードです。各 run では `seed + run_idx` が使われます。
- `--runs INT`
  - 既定値: `1`
  - 1 trial あたりの反復回数です。trial の最終評価値は各 run の `best validation accuracy` の平均です。
- `--log-dir PATH`
  - 既定値: `logs/optuna_cnn`
  - ログの出力先ディレクトリです。
- `--study-name STR`
  - Optuna study 名です。
- `--storage STR`
  - Optuna の storage URL です。指定時は `load_if_exists=True` で study を作成します。
- `--timeout INT`
  - 探索の時間上限です。単位は秒です。
- `--n-jobs INT`
  - 既定値: `1`
  - Optuna の trial 並列実行数です。

### 入力条件と固定学習条件

- `--pretrained` / `--no-pretrained`
  - 既定値: `--pretrained`
  - torchvision の ImageNet 学習済み重みを使うかどうかを指定します。
  - `tiny` は独自モデルのため、この指定の影響は実質ありません。
- `--in-ch INT`
  - 既定値: `1`
  - 入力チャネル数です。
- `--resize STR`
  - 既定値: `256,256`
  - 画像サイズ指定です。`256`、`256,256`、`none` を受けます。
  - `none` のとき画像サイズが揃っていないと、`--batch 1` 以外では失敗する場合があります。

### 探索候補

- `--models CSV`
  - 既定値: `resnet18,resnet50,mobilenet_v2`
  - 探索対象のモデル候補です。
- `--batch-candidates CSV`
  - 既定値: `16,32,64`
  - 探索対象のバッチサイズ候補です。
- `--optimizer-candidates CSV`
  - 既定値: `adam,adamw`
  - 探索対象の optimizer 候補です。
- `--lr-low FLOAT`
  - 既定値: `1e-5`
  - 学習率探索範囲の下限です。
- `--lr-high FLOAT`
  - 既定値: `1e-3`
  - 学習率探索範囲の上限です。
- `--weight-decay-candidates CSV`
  - 既定値: `0.0,1e-6,1e-5,1e-4`
  - 探索対象の weight decay 候補です。

### 学習率スケジューラ

- `--lr-scheduler {none,step,multistep,exponential,cosine,plateau,cosine_warm_restarts,onecycle}`
  - 既定値: `none`
  - 全 trial 共通で使う scheduler の種類です。
- `--scheduler-step-size INT`
  - 既定値: `5`
  - `step` の更新間隔です。
  - `cosine_warm_restarts` で `--scheduler-t-0` 未指定時の初期周期としても使われます。
- `--scheduler-milestones CSV`
  - 既定値: `10,20`
  - `multistep` で学習率を下げる epoch 群です。
- `--scheduler-gamma FLOAT`
  - 既定値: `0.1`
  - `step`, `multistep`, `plateau` で使う係数です。
  - `plateau` では `1.0` 未満である必要があります。
- `--scheduler-exp-gamma FLOAT`
  - 既定値: `0.95`
  - `exponential` で使う係数です。
- `--scheduler-patience INT`
  - 既定値: `2`
  - `plateau` の patience です。
- `--scheduler-t-max INT`
  - `cosine` の `T_max` です。未指定時は `epochs` が使われます。
- `--scheduler-eta-min FLOAT`
  - 既定値: `0.0`
  - `cosine` と `cosine_warm_restarts` の最小学習率です。
- `--scheduler-t-0 INT`
  - `cosine_warm_restarts` の `T_0` です。未指定時は `scheduler-step-size` が使われます。
- `--scheduler-t-mult INT`
  - 既定値: `1`
  - `cosine_warm_restarts` の `T_mult` です。
- `--scheduler-pct-start FLOAT`
  - 既定値: `0.3`
  - `onecycle` の `pct_start` です。
- `--scheduler-div-factor FLOAT`
  - 既定値: `25.0`
  - `onecycle` の `div_factor` です。
- `--scheduler-final-div-factor FLOAT`
  - 既定値: `1e4`
  - `onecycle` の `final_div_factor` です。

### Early Stopping と Pruning

- `--early-stopping-patience INT`
  - 既定値: `3`
  - validation accuracy が改善しない epoch がこの回数に達すると学習を止めます。
- `--early-stopping-min-delta FLOAT`
  - 既定値: `0.0`
  - 改善とみなす最小差分です。
- `--restore-best` / `--no-restore-best`
  - 既定値: `--restore-best`
  - 各 run の終了時に best epoch の重みへ戻すかどうかです。
- `--pruner-startup-trials INT`
  - 既定値: `5`
  - MedianPruner が prune 判定を始める前に確保する trial 数です。
- `--pruner-warmup-epochs INT`
  - 既定値: `2`
  - trial 内で prune 判定を始める前に待つ epoch 数です。

### 探索後の評価

- `--evaluate-best` / `--no-evaluate-best`
  - 既定値: `--evaluate-best`
  - 探索終了後、最良 trial の設定で `test/` を使った再評価を行うかどうかです。

### 探索モード

- `--per-model`
  - `--models` の各候補に対して独立した Optuna study を実行します。
  - 例: `resnet18` 用 study、`resnet50` 用 study を別々に実行します。
  - 最終的にモデル別の best trial と全体の best model を集約します。

## 利用できる値

### `--models`

- `tiny`
- `alexnet`
- `vgg16`
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `densenet`
- `densenet121`
- `mobilenet`
- `mobilenet_v2`
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

### `--optimizer-candidates`

- `adam`
- `adamw`
- `sgd`

## 出力

### 通常モード

出力先は概ね次の構成です。

```text
<log-dir>/
  <timestamp>/
    optuna_tuning_log.json
    best_eval/
      <timestamp>/
        train_log.json
        lr_curves.png
        loss_curves.png
        val_acc_curves.png
```

- `optuna_tuning_log.json`
  - 実行引数
  - study 名
  - best trial
  - 全 trial の結果
  - `--evaluate-best` 有効時は `best_eval` の結果要約

### `--per-model` モード

出力先は概ね次の構成です。

```text
<log-dir>/
  <timestamp>_by_model/
    per_model_tuning_log.json
    resnet18/
      <timestamp>/
        optuna_tuning_log.json
        best_eval/
          <timestamp>/
            train_log.json
    resnet50/
      <timestamp>/
        optuna_tuning_log.json
```

- `per_model_tuning_log.json`
  - モデルごとの best trial
  - モデルごとの best evaluation 結果
  - 全体で最良だった `best_model`

## 使用例

### 通常の探索

```bash
python -m apk2img_ml tune-cnn \
  --data-root ./images256 \
  --trials 20 \
  --epochs 15 \
  --models resnet18,resnet50,mobilenet_v2 \
  --batch-candidates 16,32,64 \
  --optimizer-candidates adam,adamw \
  --early-stopping-patience 3 \
  --log-dir ./results/optuna_cnn
```

### モデルごとに独立して探索

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

### 学習済み重みを使わずに探索

```bash
python -m apk2img_ml tune-cnn \
  --data-root ./images256 \
  --trials 20 \
  --epochs 15 \
  --models resnet18,resnet50 \
  --no-pretrained \
  --log-dir ./results/optuna_cnn_scratch
```

### SQLite storage を使って探索を保存

```bash
python -m apk2img_ml tune-cnn \
  --data-root ./images256 \
  --trials 20 \
  --study-name cnn_study \
  --storage sqlite:///./results/optuna.db \
  --log-dir ./results/optuna_cnn
```

## 注意点

- `--resize none` と `--batch > 1` の組み合わせは、画像サイズが不揃いだと失敗する場合があります。
- `--timeout` を指定すると、必ずしも `--trials` 回す前に探索が終了します。
- `--storage` を使う場合、同じ `--study-name` を再利用すると既存 study に追記されます。
- `--n-jobs` は Optuna の trial 並列数であり、DataLoader の `--workers` とは別です。
