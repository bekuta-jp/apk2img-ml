# docs index

`apk2img-ml` のドキュメント入口です。

## 最初に読む
- [../README.md](../README.md): プロジェクト概要
- [programs.md](programs.md): 各スクリプトの仕様と使い方

## 目的別ガイド
- APK から API トークンを作りたい:
  - `apk2tokens_pipeline.py`（apktool ベース）
  - `apk2apis.py`（Androguard ベース）
- Doc2Vec で学習・推論したい:
  - `train_doc2vec_from_api_tokens.py`
  - `api2vec-m.py`
  - `vec2png.py`
- sdhash まで含めた前処理をしたい:
  - `apk2sdhash.py`

## 補足
- 実体のスクリプトは `src/apk2img-ml/legacy/` にあります。
- `legacy` 配下には旧版・重複実装も含まれるため、`programs.md` の「備考」を合わせて参照してください。
