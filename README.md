# apk2img-ml

APK から API トークンを抽出し、Doc2Vec 学習・推論・画像化まで行うためのツール群です。

`legacy` は過去の実験コードを保管する領域として残し、現在の運用コードは `src/apk2img_ml/` に再構成しています。

## 現在のコード配置

- `src/apk2img_ml/extraction/`
  - Androguard 抽出（主系）
  - baksmali 抽出（フォールバック）
  - apktool + smali 抽出（フォールバック）
  - ハイブリッド実行パイプライン
- `src/apk2img_ml/embedding/`
  - Doc2Vec コーパス読込
  - 学習
  - 推論
- `src/apk2img_ml/imaging/`
  - DocVec TSV -> PNG 変換
- `src/apk2img_ml/cnn/`
  - CNN モデル定義
  - 学習・検証・テスト実行 (`train_eval_mrun`)
- `src/apk2img-ml/legacy/`
  - 旧実験コード（変更禁止領域）

## CLI

インストール後、または `python -m apk2img_ml` で実行できます。

### 1) API トークン抽出（Androguard 主系 + フォールバック）

```bash
python -m apk2img_ml extract \
  --input-dir ./apks \
  --output-dir ./api_corpus \
  --backend hybrid \
  --recursive \
  --workers 4 \
  --baksmali-jar ./baksmali.jar
```

`hybrid` は `androguard -> baksmali -> apktool` の順で試行します。

### 2) Doc2Vec 学習

```bash
python -m apk2img_ml train-doc2vec \
  --corpus-dir ./api_corpus \
  --model-out ./models/doc2vec.model \
  --docvecs-out ./features/docvecs_train.tsv
```

### 3) 学習済みモデルで推論

```bash
python -m apk2img_ml infer-doc2vec \
  --corpus-dir ./api_corpus_test \
  --model ./models/doc2vec.model \
  --docvecs-out ./features/docvecs_test.tsv
```

### 4) ベクトル画像化

```bash
python -m apk2img_ml docvec-to-png \
  --docvecs-tsv ./features/docvecs_train.tsv \
  --output-dir ./images/train \
  --mode flat
```

### 5) CNN 学習・評価

```bash
python -m apk2img_ml train-eval-mrun \
  --data-root ./images256 \
  --model tiny \
  --epochs 15 \
  --batch 32 \
  --workers 4 \
  --in-ch 1 \
  --resize 256,256 \
  --seed 3407 \
  --runs 3 \
  --log-dir ./results/train_eval_mrun
```

`train-eval-mrun` は次を行います。

- `dev/` を train/val に `8:2` 分割
- `test/` で最終評価
- `train_log.json` と各種学習曲線を保存
- `--workers 0` に対応
- 任意チャネル数に対応
- `tiny` は `256x256` 互換を保ったまま任意入力サイズに対応
- `resnet18/34/50/101/152`, `efficientnet_b0-b7`, `efficientnet_v2_s/m/l` に対応

## 補足

- `legacy` 配下は変更していません。
- baksmali / apktool / sdhash は外部ツールとして別途インストールが必要です。
- 依存関係と導入手順（`pip` / 外部ツール / `conda`）は `docs/setup.md` を参照してください。
- `train_eval_mrun` の移植内容は `docs/current_changes.md` を参照してください。
- 実行結果の集約先は `results/README.md` を参照してください。
- `sdhash` は通常版では最近の環境で問題が出るため、`legacy` 用途では [bekuta-jp/sdhash_for_ubuntu24.04](https://github.com/bekuta-jp/sdhash_for_ubuntu24.04) の利用を推奨します。
