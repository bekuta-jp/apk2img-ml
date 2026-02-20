# apk2img-ml

APK から API 呼び出し系列を抽出し、埋め込み学習・ベクトル化・画像化まで行うための実験用ツール群です。  
主なスクリプトは `src/apk2img-ml/legacy/` にまとまっています。

## このリポジトリでできること
- APK から `smali` や API トークン列を抽出
- API トークン列から `Doc2Vec` / `Word2Vec` / `FastText` を学習
- ベクトルを `TSV` / `PNG` / `NPY` に変換
- `sdhash` と `doc2vec` 画像を合成して RGB 画像を生成

## 代表的な処理フロー
1. APK から API トークンを作る（`apk2tokens_pipeline.py` / `apk2apis.py`）
2. トークンで埋め込みを学習する（`train_doc2vec_from_api_tokens.py` など）
3. 推論・可視化する（`api2vec-m.py` / `vec2png.py` / `embed2img.py`）
4. 必要に応じて `sdhash` 系出力と統合する（`merge_sdhash_doc2vec.py`）

## ディレクトリ構成
- `src/apk2img-ml/legacy/`: 実験スクリプト本体
- `docs/`: 仕様・使い方ドキュメント
- `tests/`: テスト用（現状は雛形のみ）

## 依存関係
### Python
- Python 3.10+
- 主なライブラリ: `numpy`, `gensim`, `pillow`, `tqdm`, `androguard`, `loguru`, `torch`, `torchvision`

### 外部ツール（スクリプトに応じて必要）
- `apktool`
- `sdhash`
- `java` + `baksmali.jar`

## ドキュメント
- ドキュメント入口: [docs/index.md](docs/index.md)
- 各プログラム仕様: [docs/programs.md](docs/programs.md)

## 注意
- `legacy` 配下は研究・検証スクリプト群のため、命名や実装に揺れがあります。
- 一部ファイルは同一コードがファイル内で重複しています（仕様をまとめる際は最終定義を基準）。
