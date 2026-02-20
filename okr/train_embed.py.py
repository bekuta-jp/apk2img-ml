#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API列コーパス（*.txt, 1 APK = 1 行）から Word2Vec か FastText を学習し、KeyedVectors を保存。
- import 用: train_embeddings(...)
- CLI 用  : python train_embed.py --corpus dir --algo {w2v,fasttext} --dim 64/128/256 --out model.kv
"""
import argparse, glob, os
from typing import Iterable, List
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors

def iter_corpus(dirpath: str) -> Iterable[List[str]]:
    for p in sorted(glob.glob(os.path.join(dirpath, "*.txt"))):
        with open(p, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            toks = line.split()
            if toks:
                yield toks

def train_embeddings(
    corpus_dir: str,
    algo: str = "w2v",
    dim: int = 128,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    negative: int = 10,
    epochs: int = 15,
    ft_min_n: int = 3,
    ft_max_n: int = 6,
) -> KeyedVectors:
    sentences = list(iter_corpus(corpus_dir))
    if algo == "w2v":
        model = Word2Vec(
            sentences=sentences, vector_size=dim, window=window,
            min_count=min_count, sg=sg, negative=negative,
            epochs=epochs, workers=os.cpu_count()
        )
    else:
        model = FastText(
            sentences=sentences, vector_size=dim, window=window,
            min_count=min_count, sg=sg, negative=negative,
            epochs=epochs, workers=os.cpu_count(),
            min_n=ft_min_n, max_n=ft_max_n
        )
    return model.wv  # KeyedVectors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="*.txt が並ぶディレクトリ")
    ap.add_argument("--algo", choices=["w2v", "fasttext"], default="w2v")
    ap.add_argument("--dim", type=int, default=128, choices=[64, 128, 256])
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--sg", type=int, default=1)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-min-n", type=int, default=3)
    ap.add_argument("--ft-max-n", type=int, default=6)
    ap.add_argument("--out", required=True, help="出力 .kv (KeyedVectors) ファイル")
    args = ap.parse_args()

    kv = train_embeddings(
        corpus_dir=args.corpus, algo=args.algo, dim=args.dim, window=args.window,
        min_count=args.min_count, sg=args.sg, negative=args.negative, epochs=args.epochs,
        ft_min_n=args.ft_min_n, ft_max_n=args.ft_max_n
    )
    kv.save(args.out)
    print(f"[train_embed] saved: {args.out} (dim={kv.vector_size}, vocab={len(kv)})")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API列コーパス（*.txt, 1 APK = 1 行）から Word2Vec か FastText を学習し、KeyedVectors を保存。
- import 用: train_embeddings(...)
- CLI 用  : python train_embed.py --corpus dir --algo {w2v,fasttext} --dim 64/128/256 --out model.kv
"""
import argparse, glob, os
from typing import Iterable, List
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors

def iter_corpus(dirpath: str) -> Iterable[List[str]]:
    for p in sorted(glob.glob(os.path.join(dirpath, "*.txt"))):
        with open(p, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            toks = line.split()
            if toks:
                yield toks

def train_embeddings(
    corpus_dir: str,
    algo: str = "w2v",
    dim: int = 128,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    negative: int = 10,
    epochs: int = 15,
    ft_min_n: int = 3,
    ft_max_n: int = 6,
) -> KeyedVectors:
    sentences = list(iter_corpus(corpus_dir))
    if algo == "w2v":
        model = Word2Vec(
            sentences=sentences, vector_size=dim, window=window,
            min_count=min_count, sg=sg, negative=negative,
            epochs=epochs, workers=os.cpu_count()
        )
    else:
        model = FastText(
            sentences=sentences, vector_size=dim, window=window,
            min_count=min_count, sg=sg, negative=negative,
            epochs=epochs, workers=os.cpu_count(),
            min_n=ft_min_n, max_n=ft_max_n
        )
    return model.wv  # KeyedVectors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="*.txt が並ぶディレクトリ")
    ap.add_argument("--algo", choices=["w2v", "fasttext"], default="w2v")
    ap.add_argument("--dim", type=int, default=128, choices=[64, 128, 256])
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--sg", type=int, default=1)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-min-n", type=int, default=3)
    ap.add_argument("--ft-max-n", type=int, default=6)
    ap.add_argument("--out", required=True, help="出力 .kv (KeyedVectors) ファイル")
    args = ap.parse_args()

    kv = train_embeddings(
        corpus_dir=args.corpus, algo=args.algo, dim=args.dim, window=args.window,
        min_count=args.min_count, sg=args.sg, negative=args.negative, epochs=args.epochs,
        ft_min_n=args.ft_min_n, ft_max_n=args.ft_max_n
    )
    kv.save(args.out)
    print(f"[train_embed] saved: {args.out} (dim={kv.vector_size}, vocab={len(kv)})")

if __name__ == "__main__":
    main()
