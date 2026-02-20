#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
学習済み Doc2Vec モデルを使って、API トークンファイル群 (テスト用) から
文書ベクトルを推論するスクリプト。

※ このスクリプトは「学習は一切行わず」、既存モデルに対して infer_vector だけ行う。

入力:
  - api_corpus_dir_test: テスト用の API トークン .txt 群ルート
    例)
      api_corpus_dir_test/
        benign/
          appA.txt
        malware/
          mB.txt
  - pretrained_model: 学習用コーパスで事前に学習して保存しておいた Doc2Vec モデル

出力:
  - 文書ベクトル一覧 TSV (output_docvecs)
    形式: 1 行目 "tag\tv0\tv1\t... "
          2 行目以降 "benign/appA.txt\t0.123\t-0.045\t..."
"""

import argparse
from pathlib import Path
import sys  # ★ADDED: エラー位置を標準エラーに出すため
from multiprocessing.dummy import Pool as ThreadPool  # ★ADDED: スレッドプールで並列推論

from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm


def load_corpus_for_infer(api_corpus_dir: Path):
    """
    API トークンの .txt ファイル群から (tag, tokens) のリストを作成する。

    各ファイル:
      - 最初の 1 行のみを使用 (APK 1つ = 1 ドキュメント)
      - 行をスペース区切りで分割しトークン列とする
      - タグには「ルートからの相対パス文字列」を使用
        例) benign/appA.txt
    """
    docs = []

    txt_paths = sorted(api_corpus_dir.rglob("*.txt"))
    for txt_path in tqdm(txt_paths, desc="Loading test corpus"):
        with txt_path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
        if not line:
            # 空行ならスキップ
            continue
        tokens = line.split()
        tag = str(txt_path.relative_to(api_corpus_dir))
        docs.append((tag, tokens))

    return docs


def save_inferred_vectors(
    model: Doc2Vec,
    docs,
    out_path: Path,
    infer_epochs: int = 50,
    alpha: float = 0.025,
    workers: int = 1,  # ★ADDED: 推論用ワーカ数
):
    """
    学習済み Doc2Vec モデルから、(tag, tokens) ごとに infer_vector でベクトルを推論し、
    TSV 形式で保存する。

    出力形式:
        tag  v0  v1  v2  ...  v{vector_size-1}
    """
    vector_size = model.vector_size
    total_docs = len(docs)  # ★ADDED: 全件数を保持

    # ★ADDED: 並列推論関数（スレッド用）
    def _infer_one(task):
        idx, tag, tokens = task
        try:
            vec = model.infer_vector(tokens, epochs=infer_epochs, alpha=alpha)
        except Exception as e:
            print(
                f"[ERROR] infer_vector failed at index {idx + 1}/{total_docs}, tag={tag}: {e}",
                file=sys.stderr,
                flush=True,
            )
            raise
        return idx, tag, vec

    with out_path.open("w", encoding="utf-8") as f:
        # ヘッダ行
        header = ["tag"] + [f"v{i}" for i in range(vector_size)]
        f.write("\t".join(header) + "\n")

        if workers is None or workers <= 1:
            # ★元の単一スレッド版（互換性維持）
            for idx, (tag, tokens) in enumerate(
                tqdm(docs, desc="Inferring vectors", total=total_docs)
            ):
                try:
                    # infer_vector で doc ベクトルを推論（学習はしない）
                    vec = model.infer_vector(tokens, epochs=infer_epochs, alpha=alpha)
                except Exception as e:
                    # どのドキュメントで落ちたか詳細表示
                    print(
                        f"[ERROR] infer_vector failed at index {idx + 1}/{total_docs}, tag={tag}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise

                vec_str = "\t".join(str(x) for x in vec)
                f.write(f"{tag}\t{vec_str}\n")
        else:
            # ★ADDED: スレッドプールを使った並列推論版
            tasks = [(idx, tag, tokens) for idx, (tag, tokens) in enumerate(docs)]
            results = [None] * total_docs  # インデックス順に並べ直すためのバッファ

            with ThreadPool(processes=workers) as pool:
                for idx, tag, vec in tqdm(
                    pool.imap_unordered(_infer_one, tasks),
                    total=total_docs,
                    desc=f"Inferring vectors (parallel, workers={workers})",
                ):
                    results[idx] = (tag, vec)

            # 書き出しは順序をそろえて単一スレッドで
            for tag, vec in results:
                vec_str = "\t".join(str(x) for x in vec)
                f.write(f"{tag}\t{vec_str}\n")


def main():
    parser = argparse.ArgumentParser(
        description="学習済み Doc2Vec モデルを用いて、API トークンファイル群 (テスト用) をベクトルに変換するスクリプト"
    )
    parser.add_argument(
        "api_corpus_dir",
        type=str,
        help="テスト用の API トークン .txt 群が入っているルートディレクトリ",
    )
    parser.add_argument(
        "pretrained_model",
        type=str,
        help="事前に学習済みの Doc2Vec モデルファイルパス (例: api_doc2vec.model)",
    )
    parser.add_argument(
        "--output-docvecs",
        type=str,
        default="docvecs_test.tsv",
        help="推論した文書ベクトルを保存する TSV ファイル名 (既定: docvecs_test.tsv)",
    )

    # 推論用ハイパーパラメータ（学習ではなく infer_vector 用）
    parser.add_argument(
        "--infer-epochs",
        type=int,
        default=50,
        help="infer_vector 時の反復回数 (既定: 50)"
    )
    parser.add_argument(
        "--infer-alpha",
        type=float,
        default=0.025,
        help="infer_vector 時の初期学習率 (既定: 0.025)"
    )
    # ★ADDED: 並列推論用ワーカ数
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列推論に使用するワーカ数 (1 以下なら並列化しない)",
    )

    args = parser.parse_args()

    api_corpus_dir = Path(args.api_corpus_dir).resolve()
    if not api_corpus_dir.is_dir():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {api_corpus_dir}")

    model_path = Path(args.pretrained_model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Doc2Vec モデルファイルが見つかりません: {model_path}")

    # テストコーパス読み込み（学習ではなく infer 用）
    docs = load_corpus_for_infer(api_corpus_dir)
    if not docs:
        raise RuntimeError("有効なテストドキュメントが 1 つもありません。API トークンファイルを確認してください。")

    print(f"[INFO] Total test documents: {len(docs)}")

    # モデル読み込み (mmap='r' でメモリ節約)
    print(f"[INFO] Loading Doc2Vec model from: {model_path}", file=sys.stderr, flush=True)
    try:
        model = Doc2Vec.load(str(model_path), mmap='r')
    except Exception as e:
        print(f"[ERROR] Failed to load Doc2Vec model: {e}", file=sys.stderr, flush=True)
        raise
    print(f"[INFO] Model loaded.", file=sys.stderr, flush=True)

    # ベクトル推論＆保存
    docvecs_path = Path(args.output_docvecs).resolve()
    docvecs_path.parent.mkdir(parents=True, exist_ok=True)
    save_inferred_vectors(
        model,
        docs,
        docvecs_path,
        infer_epochs=args.infer_epochs,
        alpha=args.infer_alpha,
        workers=args.workers,  # ★ADDED
    )
    print(f"[INFO] Inferred test document vectors saved to: {docvecs_path}")


if __name__ == "__main__":
    main()
