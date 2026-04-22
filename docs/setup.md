# Setup / Dependencies

このページでは、`apk2img-ml` の実行に必要な `pip` パッケージと外部ツール、`conda` を使った導入手順をまとめます。

## 1. Python (`pip`) 依存

`pyproject.toml` の `dependencies` で定義されている必須パッケージ:

- `gensim>=4.3`
- `numpy>=1.24`
- `pillow>=10.0`
- `tqdm>=4.66`
- `androguard>=4.1`

CNN 関連を使う場合のみ追加:

- `matplotlib>=3.8`
- `optuna>=3.6`
- `scikit-learn>=1.3`
- `torch>=2.0`
- `torchvision>=0.15`

インストール例:

```bash
pip install -e .
# CNN も使う場合
pip install -e ".[cnn]"
```

## 2. 外部ツール依存（コマンド別）

- `extract --backend androguard`
  - 外部ツール不要（Python パッケージのみ）
- `extract --backend baksmali` または `hybrid` で baksmali を使う場合
  - `java` コマンド（JRE/JDK）
  - `baksmali.jar`（`--baksmali-jar` で指定）
- `extract --backend apktool` または `hybrid` で apktool を使う場合
  - `apktool` コマンド
  - `java` コマンド（apktool 実行に必要）
- `sdhash`
  - 現行の運用 CLI (`src/apk2img_ml/`) では未使用
  - `src/apk2img-ml/legacy/` の一部スクリプトで必要

## 3. conda での基本セットアップ例

```bash
conda create -n apk2img-ml python=3.11 pip -y
conda activate apk2img-ml

# 外部ツール（Java / apktool）と補助コマンド
conda install -c conda-forge openjdk apktool curl git -y

# Python 側
pip install -e .
```

`baksmali.jar` は `conda` パッケージではなく JAR 配布が一般的です。例:

```bash
mkdir -p tools
curl -L -o tools/baksmali.jar <baksmali-jarのURL>
```

実行時に指定:

```bash
python -m apk2img_ml extract \
  --input-dir ./apks \
  --output-dir ./api_corpus \
  --backend hybrid \
  --baksmali-jar ./tools/baksmali.jar
```

CNN 学習・評価を使う場合の確認:

```bash
python -m apk2img_ml train-eval-mrun --help
```

## 4. sdhash は `sdhash_for_ubuntu24.04` を利用

通常版 `sdhash` は最近の環境でビルド/実行が難しいケースがあるため、以下の fork を使ってください。

- [bekuta-jp/sdhash_for_ubuntu24.04](https://github.com/bekuta-jp/sdhash_for_ubuntu24.04)

`conda` 環境にインストールする例（Ubuntu 24.04 想定）:

※ upstream README は `apt` + `sudo make install` ですが、ここでは同等依存を `conda-forge` で入れて `--prefix="$CONDA_PREFIX"` へインストールする形に置き換えています。

```bash
conda activate apk2img-ml
conda install -c conda-forge \
  git make automake autoconf libtool pkg-config \
  gxx_linux-64 openssl ssdeep -y

git clone https://github.com/bekuta-jp/sdhash_for_ubuntu24.04.git tools/sdhash_for_ubuntu24.04
cd tools/sdhash_for_ubuntu24.04
./bootstrap
./configure --prefix="$CONDA_PREFIX"
make -j"$(nproc)"
make install
```

確認:

```bash
which sdhash
sdhash -h
```

`sdhash` が必要な処理を実行する場合、`PATH` 上で上記 `sdhash` が先に見つかる状態で実行してください。
