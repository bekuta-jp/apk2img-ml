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
  - CNN モデル定義ユーティリティ
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

## 補足

- `legacy` 配下は変更していません。
- baksmali / apktool / sdhash は外部ツールとして別途インストールが必要です。
