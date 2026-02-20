#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APKごとのAPIコーパスを、Word2Vec + グレースケール画像に変換するスクリプト

- 前提:
  - コーパスは「1 APK = 1 テキストファイル」
  - 各ファイルは「1行 = 1 API呼び出し名」とする
  - 例:
      Landroid/app/Activity;->onCreate
      Landroid/widget/Button;->setOnClickListener
      ...

- 処理の流れ:
  1. コーパス全体から Word2Vec を学習（または既存モデルを読み込み）
  2. 学習済み埋め込みから全APIの全次元の min/max を算出
  3. APKごとに (API数, ベクトル次元) の 2D 配列を作成し、0–255へ正規化して PNG で保存
"""

import argparse
import os
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from PIL import Image
from tqdm import tqdm


class CorpusIterator:  # ★追加: ストリーミング読み込み用クラス
    """
    コーパスから Word2Vec 学習用の「文」をストリーミングで提供するイテレータ。

    ここでは「1 APK ファイル = 1 文」として扱い、
    各行(API名)をその文のトークンとみなす。

    list(...) で全読み込みするとメモリを食うので、
    何度でも読み直せるイテレータクラスにしている。
    """

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.paths = sorted(corpus_dir.glob("*.txt"))
        print(f"[INFO] corpus 内の *.txt ファイル数: {len(self.paths)}")  # ★追加

    def __iter__(self):
        for corpus_path in self.paths:
            tokens = []
            # ★変更: encoding エラーで落ちないように errors='ignore' を追加
            with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:  # ★変更
                for line in f:
                    api = line.strip()
                    if not api:
                        continue
                    tokens.append(api)
            if tokens:
                yield tokens
            # tokens が空（全行空白など）のファイルは無視


def train_word2vec(
    corpus_dir: Path,
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    epochs: int = 10,
    model_path: Path = Path("api_word2vec.model"),
):
    """
    コーパスディレクトリから Word2Vec モデルを新規学習して保存する。
    """
    print(f"[INFO] Loading sentences from: {corpus_dir}")
    sentences = CorpusIterator(corpus_dir)  # ★変更: list(iter_sentences) をやめてストリーミングに

    # ★追加: 一度だけ軽く走査して「文の件数」を数える（メモリは食わない）
    print("[INFO] Counting sentences (this may take a while)...")  # ★追加
    num_sentences = 0  # ★追加
    for _ in sentences:  # ★追加
        num_sentences += 1  # ★追加
    print(f"[INFO] Detected {num_sentences} sentences")  # ★追加

    if num_sentences == 0:  # ★追加
        print("[ERROR] コーパスから1件も文が読み込めませんでした。")  # ★追加
        print("        corpus-dir のパスや *.txt の内容を確認してください。")  # ★追加
        return None  # ★追加

    # ★追加: 再度イテレータとして使えるように、新しい CorpusIterator を作り直す
    sentences = CorpusIterator(corpus_dir)  # ★追加

    print("[INFO] Training Word2Vec model...")
    try:  # ★追加: 失敗時にエラー内容を表示
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,  # 1: skip-gram（レア単語に強い）
            epochs=epochs,
        )
    except RuntimeError as e:  # ★追加
        print("[ERROR] Word2Vec 学習中にエラーが発生しました:", repr(e))  # ★追加
        return None  # ★追加

    print(f"[INFO] Saving model to: {model_path}")
    model.save(str(model_path))
    # model.wv.save(str(model_path.with_suffix(".kv")))  # 必要なら単語ベクトルだけ保存

    return model


def load_api_list(corpus_path: Path):
    """
    1つの APK コーパス（テキストファイル）から API 名のリストを取得する。
    """
    apis = []
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:  # ★変更: errors='ignore' 追加
        for line in f:
            api = line.strip()
            if not api:
                continue
            apis.append(api)
    return apis


def compute_global_minmax(corpus_dir: Path, wv) -> tuple[np.ndarray, np.ndarray]:
    """
    全コーパス中のすべての API ベクトルについて、
    各次元ごとの最小値・最大値を求める。
    """
    dim = wv.vector_size
    vmin = np.full(dim, np.inf, dtype=np.float32)
    vmax = np.full(dim, -np.inf, dtype=np.float32)

    print("[INFO] Scanning corpus to compute global min/max of embeddings...")
    for corpus_path in tqdm(sorted(corpus_dir.glob("*.txt")), desc="scan min/max"):
        apis = load_api_list(corpus_path)
        for api in apis:
            if api not in wv:
                continue
            v = wv[api]
            vmin = np.minimum(vmin, v)
            vmax = np.maximum(vmax, v)

    # 全く更新されない次元がある場合の対策（分母0回避）
    same = vmax <= vmin
    vmax[same] = vmin[same] + 1e-6

    return vmin, vmax


def corpus_to_image(
    corpus_path: Path,
    wv,
    vmin: np.ndarray,
    vmax: np.ndarray,
    out_dir: Path,
):
    """
    1つの APK コーパスファイルを、
    (API数, ベクトル次元) のグレースケール画像に変換して保存する。

    - 横方向の長さ = ベクトルの次元数
    - 縦方向の長さ = API の行数
    """
    apis = load_api_list(corpus_path)
    if not apis:
        return  # 空ファイルならスキップ

    dim = wv.vector_size
    vecs = []

    for api in apis:
        if api in wv:
            v = wv[api]
        else:
            # 語彙になかった API はゼロベクトルにする（好みに応じて変更可）
            v = np.zeros(dim, dtype=np.float32)
        vecs.append(v)

    arr = np.stack(vecs, axis=0)  # shape: (H, D) = (API数, ベクトル次元)

    # 次元ごとの global min/max を使って 0–1 に正規化
    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    # 0–255 の uint8 に変換
    img_arr = (norm * 255.0).astype(np.uint8)

    # Pillow は 2D 配列を (高さ, 幅) とみなして L (8bit グレースケール) 画像にできる
    # → 高さ = API数, 幅 = ベクトル次元
    img = Image.fromarray(img_arr, mode="L")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{corpus_path.stem}.png"
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="APKごとのAPIコーパスをWord2Vecベクトル画像に変換する"
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
        type=Path,
        help="APKごとのコーパスファイル（*.txt）が入っているディレクトリ",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="画像(PNG)の出力先ディレクトリ",
    )
    parser.add_argument(
        "--w2v-model",
        type=Path,
        default=None,
        help="既存のWord2Vecモデル(.model)のパス。指定が無い場合は新規学習する",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=128,
        help="Word2Vecのベクトル次元数（新規学習時のみ有効）",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Word2Vecのコンテキストウィンドウ（新規学習時のみ有効）",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,  # ★必要ならここを変えてください（レアAPIも残したいなら1推奨）
        help="出現頻度がこの値未満のAPIは無視（新規学習時のみ有効）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Word2Vecの学習エポック数（新規学習時のみ有効）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Word2Vec学習に使うスレッド数（新規学習時のみ有効）",
    )
    args = parser.parse_args()

    corpus_dir: Path = args.corpus_dir
    out_dir: Path = args.out_dir

    # 1) Word2Vec モデルの用意（読み込み or 新規学習）
    if args.w2v_model is not None and args.w2v_model.exists():
        print(f"[INFO] Loading existing Word2Vec model: {args.w2v_model}")
        model = Word2Vec.load(str(args.w2v_model))
    else:
        print("[INFO] w2v-model が指定されていないか存在しないため、新規学習します")
        model = train_word2vec(
            corpus_dir=corpus_dir,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            epochs=args.epochs,
            model_path=Path("api_word2vec.model"),
        )
        if model is None:  # ★変更: 学習に失敗したらここで終了
            print("[ERROR] Word2Vec モデルの学習に失敗したため、処理を中止します。")  # ★変更
            return  # ★変更

    wv = model.wv

    # 2) 全コーパスから min/max を計算
    vmin, vmax = compute_global_minmax(corpus_dir, wv)

    # 3) APKごとに画像生成
    print("[INFO] Generating images per APK...")
    for corpus_path in tqdm(sorted(corpus_dir.glob("*.txt")), desc="make images"):
        corpus_to_image(corpus_path, wv, vmin, vmax, out_dir)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
