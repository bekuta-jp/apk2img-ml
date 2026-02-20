# 各プログラムの仕様と使い方

このページは `src/apk2img-ml/legacy/` 配下のスクリプトを対象に、用途・入出力・実行方法をまとめたものです。

## 共通前提
- 基本実行形: `python src/apk2img-ml/legacy/<script>.py ...`
- 多くのスクリプトは `*.txt` を「1 APK = 1 ファイル」「1 行にトークン列」で扱います。
- ファイル名に `legacy` がある通り、研究用途の実験コードが中心です。

## 早見表
| スクリプト | 主用途 | 主入力 | 主出力 |
|---|---|---|---|
| `apk2tokens_pipeline.py` | APK→APIトークン抽出（apktool） | APKディレクトリ | `*.txt` トークン列 |
| `apktool_pipeline.py` | APK→smali抽出（baksmaliフォールバック可） | APK/ディレクトリ | `smali*` ディレクトリ |
| `apk2apis.py` | 単一APK→APIトークン（Androguard） | 1 APK | `*.txt` |
| `doc2vec_sample.py` | 再帰APK→APIトークン（Androguard） | 入力ルート | 相対構造を維持した `*.txt` |
| `extract_with_baksmali.py` | baksmali で API トークン抽出 | APKディレクトリ | `*.txt`, `failures.csv` |
| `apk2sdhash.py` | APK→apktool→preproc→sdhash→APIコーパス | APKディレクトリ | `_preproc`, `.sdbf`, `_api_corpus` |
| `apk2sdhash_apicorp.py` | `apk2sdhash` 系の別バリアント | APKディレクトリ | 上と同様 |
| `untitled.py` | `apk2sdhash` 系の簡易バリアント | APKディレクトリ | 上と同様 |
| `train_doc2vec_from_api_tokens.py` | Doc2Vec 学習 | APIトークン `.txt` 群 | `.model`, `docvecs.tsv` |
| `train_doc2vec_from_api_tokens_old.py` | Doc2Vec 学習（旧版） | APIトークン `.txt` 群 | `.model`, `docvecs.tsv` |
| `api2vec-m.py` | 学習済み Doc2Vec 推論 | APIトークン `.txt` 群, `.model` | 推論TSV |
| `train_embed.py.py` | W2V/FastText 学習 | APIトークン `.txt` 群 | `.kv` |
| `embed2img.py` | トークン列＋KV→画像/配列 | `*.txt`, `.kv` | `.npy` / `.png` |
| `vec2png.py` | Doc2Vec TSV→PNG | `docvecs.tsv` | `*.png` |
| `corp2png.py` | APIコーパス→W2V→PNG | `*.txt` 群 | `*.png` |
| `corp2png_debug.py` | `corp2png` のデバッグ/改良版 | `*.txt` 群 | `*.png` |
| `merge_sdhash_doc2vec.py` | sdhash画像 + doc2vec画像 合成 | 2系統のPNG | RGB PNG |
| `models_zoo.py` | 画像分類モデル生成ユーティリティ | モデル名/クラス数/入力ch | `torch.nn.Module` |
| `my_worker.py` | 並列抽出ワーカー補助 | APK path, 出力path | 成否タプル |

## 1. APK 解析・トークン抽出系

### `apk2tokens_pipeline.py`
- 目的: `apktool` の結果から `invoke-*` を正規表現抽出して API トークン列を作る。
- 入力: `--apk-dir`（`*.apk`）
- 出力:
  - `--tokens-root/<apk_stem>.txt`
  - `--log-dir/*.log`
  - `--diagnostics-csv`
- 主なオプション:
  - `--workers`, `--tmp-base`
  - `--include-prefix`, `--exclude-prefix`, `--include-signature`
  - `--no-assets-retry`（assets 内 dex 無視の再試行を無効化）
- 実行例:
```bash
python src/apk2img-ml/legacy/apk2tokens_pipeline.py \
  --apk-dir ./apks \
  --tokens-root ./corp_tokens \
  --log-dir ./logs_apktool \
  --workers 4
```

### `apktool_pipeline.py`
- 目的: APK から `smali*` を抽出。失敗時は assets 除去再試行、必要なら baksmali へフォールバック。
- 入力: `--apk` または `--apk-dir`
- 出力:
  - 単体: `--out-dir`
  - 複数: `--out-root/<apk_stem>/smali*`
  - `--log-dir`, `--diagnostics-csv`
- 主なオプション:
  - `--enable-baksmali`, `--baksmali-jar`, `--java-cmd`
  - `--no-assets-retry`, `--workers`
- 実行例:
```bash
python src/apk2img-ml/legacy/apktool_pipeline.py \
  --apk-dir ./apks \
  --out-root ./corp_apktool \
  --enable-baksmali \
  --baksmali-jar ./baksmali.jar
```
- 備考: ファイル内に同一ロジックが重複定義されています。

### `apk2apis.py`
- 目的: Androguard で単一 APK から API 呼び出し列を抽出。
- 入力: `--apk`
- 出力: `--out/<stem>.txt`（1行トークン列）
- 主なオプション:
  - `--include`, `--include-none`, `--exclude`
  - `--tqdm`, `--log-file`
- 実行例:
```bash
python src/apk2img-ml/legacy/apk2apis.py \
  --apk ./sample.apk \
  --out ./corp_tokens \
  --tqdm
```

### `doc2vec_sample.py`
- 目的: ディレクトリ配下の APK を再帰処理して API トークンを保存。
- 入力: `input_dir`（位置引数）
- 出力: `output_dir/<input_dir相対パス>.txt`
- 主なオプション:
  - `--jobs/-j`（並列数）
  - `--log-file`
- 実行例:
```bash
python src/apk2img-ml/legacy/doc2vec_sample.py \
  ./apks ./api_corpus --jobs 4
```

### `extract_with_baksmali.py`
- 目的: baksmali で DEX を逆アセンブルし、`invoke-*` から API トークン抽出。
- 入力: `--apk-dir`, `--baksmali-jar`
- 出力:
  - `--out-dir/<apk_stem>.txt`
  - 失敗時 `--out-dir/failures.csv`
- 主なオプション:
  - `--include`, `--include-none`, `--exclude`
  - `--include-sig`
- 実行例:
```bash
python src/apk2img-ml/legacy/extract_with_baksmali.py \
  --apk-dir ./apks \
  --out-dir ./api_corpus_baksmali \
  --baksmali-jar ./baksmali.jar
```
- 備考: ファイル内に同一ロジックが重複定義されています。

### `apk2sdhash.py`
- 目的: APK から `apktool`・前処理・`sdhash`・APIコーパス生成までを一括実行。
- 入力: `--input-dir`, `--temp-dir`, `--output-dir`, `--log-dir`
- 出力（オプション有効時）:
  - apktool 出力（all / smali+manifest）
  - `_preproc`（正規化テキスト）
  - `_sdbf/*.sdbf`
  - `_api_corpus/api_sequences.txt`
- 主なオプション:
  - 保存系: `--apktool-save`, `--preproc-save`, `--sdbf-save`, `--api-corpus-save`
  - 保存先分離: `--apktool-out-root`, `--preproc-out-root`, `--sdbf-out-root`, `--api-corpus-out-root`
  - 回復系: `--apktool-retry-strip-assets`, `--overwrite`, `--keep-temp`
- 実行例:
```bash
python src/apk2img-ml/legacy/apk2sdhash.py \
  -i ./apks -o ./out -l ./logs -t ./tmp \
  --apktool-save smali_manifest \
  --sdbf-save --api-corpus-save \
  --workers 4
```
- 備考: `apktool` 完全失敗時に raw DEX から API 抽出するフォールバックがあります。

### `apk2sdhash_apicorp.py`
- 目的: `apk2sdhash.py` と同系統の一括パイプライン。
- 入出力/主オプション: 概ね `apk2sdhash.py` と同様。
- 実行例:
```bash
python src/apk2img-ml/legacy/apk2sdhash_apicorp.py \
  -i ./apks -o ./out -l ./logs -t ./tmp \
  --sdbf-save --api-corpus-save
```
- 備考:
  - ファイル内に同一ロジックが重複定義されています。
  - `apk2sdhash.py` にある raw DEX API フォールバックは入っていません。

### `untitled.py`
- 目的: `apk2sdhash` 系の別バリアント。
- 入出力: `-i/-o/-l/-t` を受け、`apktool`/`preproc`/`sdbf`/`api_corpus` を条件保存。
- 実行例:
```bash
python src/apk2img-ml/legacy/untitled.py \
  -i ./apks -o ./out -l ./logs -t ./tmp \
  --apktool-save smali_manifest --sdbf-save
```
- 備考: 命名が暫定のため、新規利用時は `apk2sdhash.py` を優先推奨。

## 2. 埋め込み学習・推論系

### `train_doc2vec_from_api_tokens.py`
- 目的: API トークン `*.txt` 群で Doc2Vec 学習。
- 入力: `api_corpus_dir`（位置引数）
- 出力: `--output-model`, `--output-docvecs`
- 主なオプション: `--vector-size`, `--window`, `--min-count`, `--epochs`, `--workers`
- 実行例:
```bash
python src/apk2img-ml/legacy/train_doc2vec_from_api_tokens.py \
  ./api_corpus --output-model ./api_doc2vec.model --output-docvecs ./docvecs.tsv
```

### `train_doc2vec_from_api_tokens_old.py`
- 目的: 上記の旧版実装。
- 入出力: `train_doc2vec_from_api_tokens.py` と同様。
- 実行例:
```bash
python src/apk2img-ml/legacy/train_doc2vec_from_api_tokens_old.py \
  ./api_corpus --output-model ./api_doc2vec_old.model
```

### `api2vec-m.py`
- 目的: 学習済み Doc2Vec を使って推論専用 `infer_vector` を実行。
- 入力: `api_corpus_dir`, `pretrained_model`
- 出力: `--output-docvecs`（既定: `docvecs_test.tsv`）
- 主なオプション: `--infer-epochs`, `--infer-alpha`, `--workers`
- 実行例:
```bash
python src/apk2img-ml/legacy/api2vec-m.py \
  ./api_corpus_test ./api_doc2vec.model \
  --output-docvecs ./docvecs_test.tsv --workers 4
```

### `train_embed.py.py`
- 目的: `Word2Vec` または `FastText` を学習し `KeyedVectors(.kv)` を保存。
- 入力: `--corpus`（`*.txt` 群）
- 出力: `--out`（`.kv`）
- 主なオプション: `--algo`, `--dim`, `--window`, `--min-count`, `--epochs`
- 実行例:
```bash
python src/apk2img-ml/legacy/train_embed.py.py \
  --corpus ./corp_tokens --algo fasttext --dim 128 --out ./api_ft.kv
```
- 備考:
  - ファイル名が `train_embed.py.py` です（拡張子が二重）。
  - ファイル内に同一ロジックが重複定義されています。

## 3. ベクトル・画像変換系

### `vec2png.py`
- 目的: Doc2Vec の TSV をグレースケール PNG 化。
- 入力: `docvecs_tsv`（位置引数）
- 出力: `output_dir/<tag相対>/xxx.png`
- エンコードモード:
  - `flat`: ベクトルを最小正方形へ敷き詰め
  - `grid`: セル分割して拡大
  - `raw32`: float32 生ビットを画像化
- 実行例:
```bash
python src/apk2img-ml/legacy/vec2png.py \
  ./docvecs.tsv ./doc_png --encode-mode grid --grid-rows 16 --grid-cols 16
```

### `embed2img.py`
- 目的: API トークン列と `KeyedVectors` から `(L, d)` 行列を作成し画像/配列として保存。
- 入力: `--seq-dir`, `--kv`
- 出力: `--out` に `.npy` または `.png`
- 主なオプション:
  - `--scale raw|uint8`
  - `--fit-stats` / `--stats` / `--clip`
  - `--png`
- 実行例:
```bash
python src/apk2img-ml/legacy/embed2img.py \
  --seq-dir ./corp_tokens --kv ./api_ft.kv --out ./embed_img \
  --scale uint8 --png --fit-stats ./train_stats.npz
```
- 備考: ファイル内に同一ロジックが重複定義されています。

### `corp2png.py`
- 目的: API コーパスから Word2Vec を学習（または読み込み）し、APK単位で PNG 生成。
- 入力: `--corpus-dir`
- 出力: `--out-dir/*.png`（高さ=API数, 幅=ベクトル次元）
- 主なオプション: `--w2v-model`, `--vector-size`, `--window`, `--min-count`, `--epochs`
- 実行例:
```bash
python src/apk2img-ml/legacy/corp2png.py \
  --corpus-dir ./api_corpus_lines --out-dir ./corp_png --vector-size 128
```

### `corp2png_debug.py`
- 目的: `corp2png.py` のメモリ・エラー耐性を強化したデバッグ版。
- 入出力/主オプション: `corp2png.py` とほぼ同等。
- 実行例:
```bash
python src/apk2img-ml/legacy/corp2png_debug.py \
  --corpus-dir ./api_corpus_lines --out-dir ./corp_png_debug
```

### `merge_sdhash_doc2vec.py`
- 目的: sdhash グレースケール画像と doc2vec グレースケール画像を RGB 合成。
- 入力:
  - `--sdhash-dir`（`*_1ch.png` など）
  - `--doc2vec-dir`（`*.png`）
- 出力: `--out-dir/<相対>/xxx.png`（R/G=sdhash上下半分, B=doc2vec）
- 主なオプション: `--prefer-suffix`, `--doc-size`, `--sd-height`, `--nearest`
- 実行例:
```bash
python src/apk2img-ml/legacy/merge_sdhash_doc2vec.py \
  --sdhash-dir ./sd_png --doc2vec-dir ./doc_png --out-dir ./merged_rgb
```

## 4. モデル・補助ユーティリティ

### `models_zoo.py`
- 目的: `alexnet`/`vgg16`/`resnet50`/`densenet`/`mobilenet`/`tiny` を返すヘルパ。
- 特徴:
  - 入力チャネル数 `in_ch` に合わせて最初の Conv を自動調整
  - 出力クラス数 `num_classes` に合わせて最終層を差し替え
- 利用例:
```python
from models_zoo import get_model
model = get_model("resnet50", num_classes=2, pretrained=True, in_ch=1)
```

### `my_worker.py`
- 目的: 並列処理用ワーカー（`apk2apis.extract_api_sequence` 呼び出し）を分離。
- 入力: `extract_and_write(apk_path_str, out_dir_str, include_prefixes, exclude_prefixes)`
- 出力: `(apk_path_str, success, error_message)`
- 利用例:
```python
from my_worker import extract_and_write
res = extract_and_write("/path/app.apk", "/path/out")
```

## 運用上の注意
- 一部ファイルは同一コードがファイル内に重複しています。
  - 確認済み: `apktool_pipeline.py`, `extract_with_baksmali.py`, `embed2img.py`, `train_embed.py.py`, `apk2sdhash_apicorp.py`
- 新規運用の基準としては、以下を優先すると扱いやすいです。
  - APK→トークン: `apk2tokens_pipeline.py`
  - Doc2Vec 学習: `train_doc2vec_from_api_tokens.py`
  - 推論/画像化: `api2vec-m.py`, `vec2png.py`
  - sdhash統合: `apk2sdhash.py`
