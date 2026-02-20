#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API トークンファイル群から Doc2Vec を学習するスクリプト。

入力:
  - api_corpus_dir: さきほどの抽出スクリプトで出力された .txt 群のルートディレクトリ
    例)
      api_corpus_dir/
        benign/
          app1.txt
          app2.txt
        malware/
          m1.txt
          m2.txt
    各 .txt は 1 行に "トークン1 トークン2 ..." という形式。

出力:
  - Doc2Vec モデル (output_model, デフォルト: doc2vec.model)
  - 文書ベクトル一覧 TSV (output_docvecs, デフォルト: docvecs.tsv)
    1 行目: "tag\tv0\tv1\t... "
"""

import argparse
from pathlib import Path

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm


def load_corpus(api_corpus_dir: Path):
    """
    API トークンの .txt ファイル群から TaggedDocument のリストを作成する。

    各ファイル:
      - 最初の 1 行のみを使用 (APK 1つ = 1 ドキュメント)
      - 行をスペース区切りで分割しトークン列とする
      - タグには「ルートからの相対パス文字列」を使用
        例) benign/app1.txt
    """
    documents = []

    txt_paths = sorted(api_corpus_dir.rglob("*.txt"))
    for txt_path in tqdm(txt_paths, desc="Loading corpus"):
        with txt_path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
        if not line:
            # 空行ならスキップ
            continue
        tokens = line.split()
        tag = str(txt_path.relative_to(api_corpus_dir))
        documents.append(TaggedDocument(tokens, [tag]))

    return documents


def train_doc2vec(
    documents,
    vector_size=128,
    window=5,
    min_count=1,
    workers=4,
    epochs=20,
):
    """
    Doc2Vec モデルを学習して返す。
    """
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )

    model.build_vocab(documents)
    model.train(documents, total_examples=len(documents), epochs=model.epochs)
    return model


def save_doc_vectors(model: Doc2Vec, documents, out_path: Path):
    """
    学習済み Doc2Vec モデルから文書ベクトルを取り出して TSV で保存する。

    出力形式:
        tag  v0  v1  v2  ...  v{vector_size-1}
    """
    vector_size = model.vector_size

    with out_path.open("w", encoding="utf-8") as f:
        # ヘッダ行
        header = ["tag"] + [f"v{i}" for i in range(vector_size)]
        f.write("\t".join(header) + "\n")

        for doc in tqdm(documents, desc="Saving doc vectors"):
            tag = doc.tags[0]
            vec = model.dv[tag]
            vec_str = "\t".join(str(x) for x in vec)
            f.write(f"{tag}\t{vec_str}\n")


def main():
    parser = argparse.ArgumentParser(
        description="API トークンファイル群から Doc2Vec を学習するスクリプト"
    )
    parser.add_argument(
        "api_corpus_dir",
        type=str,
        help="API トークン .txt 群が入っているルートディレクトリ",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="doc2vec.model",
        help="学習済み Doc2Vec モデルの保存先ファイル名 (既定: doc2vec.model)",
    )
    parser.add_argument(
        "--output-docvecs",
        type=str,
        default="docvecs.tsv",
        help="各文書ベクトルを保存する TSV ファイル名 (既定: docvecs.tsv)",
    )

    # 学習ハイパーパラメータ
    parser.add_argument("--vector-size", type=int, default=128, help="ベクトル次元数")
    parser.add_argument("--window", type=int, default=5, help="コンテキスト窓サイズ")
    parser.add_argument("--min-count", type=int, default=1, help="出現頻度の下限")
    parser.add_argument(
        "--epochs", type=int, default=20, help="学習エポック数"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="並列ワーカ数 (コア数に応じて調整)"
    )

    args = parser.parse_args()

    api_corpus_dir = Path(args.api_corpus_dir).resolve()
    if not api_corpus_dir.is_dir():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {api_corpus_dir}")

    # コーパス読み込み
    documents = load_corpus(api_corpus_dir)
    if not documents:
        raise RuntimeError("有効なドキュメントが 1 つもありません。API トークンファイルを確認してください。")

    print(f"[INFO] Total documents: {len(documents)}")

    # Doc2Vec 学習
    model = train_doc2vec(
        documents,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
    )

    # モデル保存
    model_path = Path(args.output_model).resolve()
    model.save(str(model_path))
    print(f"[INFO] Doc2Vec model saved to: {model_path}")

    # 文書ベクトル保存
    docvecs_path = Path(args.output_docvecs).resolve()
    save_doc_vectors(model, documents, docvecs_path)
    print(f"[INFO] Document vectors saved to: {docvecs_path}")


if __name__ == "__main__":
    main()
