#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API列（*.txt）と学習済み KeyedVectors(.kv) から行列(L×d)を作り、画像化。
- スケールは2択:
  1) "raw"   : 生の float32 を .npy 保存（L×d）
  2) "uint8" : 0-255 のグレースケール（L×d）を .png or .npy で保存
     - 推奨: train セットで --fit-stats を用いて mean/std を算出・保存し、
             val/test では --stats を指定して同じ統計で z-score+clip→0-255
- import 用: sequence_to_image(...)
- CLI 用  : python embed2img.py --seq-dir seqs --kv model.kv --out imgs --scale {raw,uint8}
"""
import argparse, glob, os, json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from gensim.models import KeyedVectors

try:
    from PIL import Image
except Exception:
    Image = None  # PNG 保存しない場合は不要

def load_sequences(seq_dir: str) -> List[Tuple[str, List[str]]]:
    pairs = []
    for p in sorted(glob.glob(os.path.join(seq_dir, "*.txt"))):
        with open(p, "r", encoding="utf-8") as f:
            toks = f.readline().strip().split()
        pairs.append((Path(p).stem, toks))
    return pairs

def sequence_to_matrix(tokens: List[str], kv: KeyedVectors, oov_mode: str = "ft") -> np.ndarray:
    """
    tokens -> (L, d) float32
    oov_mode:
      - "ft": FastText の subword を使って get_vector を試みる（w2v でも例外時はゼロ）
      - "zero": OOV は 0 ベクトル
    """
    d = kv.vector_size
    rows: List[np.ndarray] = []
    for t in tokens:
        vec = None
        try:
            # FastText なら OOV でも生成可能
            vec = kv.get_vector(t, norm=False)
        except KeyError:
            if oov_mode == "zero":
                vec = np.zeros((d,), np.float32)
            else:
                try:
                    vec = kv[t]  # w2v 既知語
                except KeyError:
                    vec = np.zeros((d,), np.float32)
        rows.append(vec.astype(np.float32))
    if not rows:
        return np.zeros((0, d), np.float32)
    return np.vstack(rows)

def fit_stats_from_dir(seq_dir: str, kv: KeyedVectors, subset: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    学習用（train）シーケンスから mean/std（埋め込み次元ごと）を算出。
    """
    pairs = load_sequences(seq_dir)
    if subset:
        pairs = pairs[:subset]
    bag = []
    for _, toks in pairs:
        M = sequence_to_matrix(toks, kv)
        if M.size > 0:
            bag.append(M)
    if not bag:
        d = kv.vector_size
        return np.zeros((d,), np.float32), np.ones((d,), np.float32)
    X = np.vstack(bag)  # (#tokens_all, d)
    mean = X.mean(axis=0).astype(np.float32)
    std = (X.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std

def to_uint8(M: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """
    Z-score -> クリップ -> 0-255 へ線形写像（uint8）
    """
    if M.size == 0:
        return np.zeros((0, mean.shape[0]), np.uint8)
    Z = (M - mean) / std
    Z = np.clip(Z, -clip, +clip)
    U = ((Z + clip) * (255.0 / (2 * clip))).astype(np.uint8)
    return U

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="*.txt（1 APK = 1 行）")
    ap.add_argument("--kv", required=True, help="学習済み KeyedVectors (.kv)")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--scale", choices=["raw", "uint8"], default="raw",
                    help="raw: float32 .npy / uint8: 0-255 画像（.png または .npy）")
    ap.add_argument("--png", action="store_true", help="--scale uint8 時、PNG で保存（Pillow 必須）")
    ap.add_argument("--fit-stats", help="train 用：このディレクトリの *.txt から mean/std を算出して保存する .npz パス")
    ap.add_argument("--stats", help="mean/std を含む .npz を読み込む（val/test 用）")
    ap.add_argument("--clip", type=float, default=3.0, help="Z-score のクリップ閾値（±clip）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kv = KeyedVectors.load(args.kv, mmap="r")

    # 統計
    if args.stats:
        st = np.load(args.stats)
        mean, std = st["mean"], st["std"]
    elif args.fit_stats:
        mean, std = fit_stats_from_dir(args.seq_dir, kv)
        np.savez(args.fit_stats, mean=mean, std=std)
        with open(args.fit_stats + ".meta.json", "w") as f:
            json.dump({"vector_size": int(kv.vector_size)}, f)
        print(f"[embed2img] fitted stats -> {args.fit_stats}")
    else:
        # 統計なし（raw 保存 or 暫定処理）
        mean = np.zeros((kv.vector_size,), np.float32)
        std = np.ones((kv.vector_size,), np.float32)

    pairs = load_sequences(args.seq_dir)
    for stem, toks in pairs:
        M = sequence_to_matrix(toks, kv)  # (L, d)
        if args.scale == "raw":
            # 生の float32 を .npy（L×d, C=1 相当）
            np.save(Path(args.out) / f"{stem}.npy", M.astype(np.float32))
        else:
            U = to_uint8(M, mean, std, clip=args.clip)  # (L, d) -> uint8
            if args.png:
                if Image is None:
                    raise RuntimeError("Pillow が必要です。 pip install pillow")
                if U.size == 0:
                    U2 = np.zeros((1, kv.vector_size), np.uint8)
                else:
                    U2 = U
                Image.fromarray(U2, mode="L").save(Path(args.out) / f"{stem}.png")
            else:
                np.save(Path(args.out) / f"{stem}.npy", U.astype(np.uint8))
    print(f"[embed2img] done: {args.out}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API列（*.txt）と学習済み KeyedVectors(.kv) から行列(L×d)を作り、画像化。
- スケールは2択:
  1) "raw"   : 生の float32 を .npy 保存（L×d）
  2) "uint8" : 0-255 のグレースケール（L×d）を .png or .npy で保存
     - 推奨: train セットで --fit-stats を用いて mean/std を算出・保存し、
             val/test では --stats を指定して同じ統計で z-score+clip→0-255
- import 用: sequence_to_image(...)
- CLI 用  : python embed2img.py --seq-dir seqs --kv model.kv --out imgs --scale {raw,uint8}
"""
import argparse, glob, os, json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from gensim.models import KeyedVectors

try:
    from PIL import Image
except Exception:
    Image = None  # PNG 保存しない場合は不要

def load_sequences(seq_dir: str) -> List[Tuple[str, List[str]]]:
    pairs = []
    for p in sorted(glob.glob(os.path.join(seq_dir, "*.txt"))):
        with open(p, "r", encoding="utf-8") as f:
            toks = f.readline().strip().split()
        pairs.append((Path(p).stem, toks))
    return pairs

def sequence_to_matrix(tokens: List[str], kv: KeyedVectors, oov_mode: str = "ft") -> np.ndarray:
    """
    tokens -> (L, d) float32
    oov_mode:
      - "ft": FastText の subword を使って get_vector を試みる（w2v でも例外時はゼロ）
      - "zero": OOV は 0 ベクトル
    """
    d = kv.vector_size
    rows: List[np.ndarray] = []
    for t in tokens:
        vec = None
        try:
            # FastText なら OOV でも生成可能
            vec = kv.get_vector(t, norm=False)
        except KeyError:
            if oov_mode == "zero":
                vec = np.zeros((d,), np.float32)
            else:
                try:
                    vec = kv[t]  # w2v 既知語
                except KeyError:
                    vec = np.zeros((d,), np.float32)
        rows.append(vec.astype(np.float32))
    if not rows:
        return np.zeros((0, d), np.float32)
    return np.vstack(rows)

def fit_stats_from_dir(seq_dir: str, kv: KeyedVectors, subset: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    学習用（train）シーケンスから mean/std（埋め込み次元ごと）を算出。
    """
    pairs = load_sequences(seq_dir)
    if subset:
        pairs = pairs[:subset]
    bag = []
    for _, toks in pairs:
        M = sequence_to_matrix(toks, kv)
        if M.size > 0:
            bag.append(M)
    if not bag:
        d = kv.vector_size
        return np.zeros((d,), np.float32), np.ones((d,), np.float32)
    X = np.vstack(bag)  # (#tokens_all, d)
    mean = X.mean(axis=0).astype(np.float32)
    std = (X.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std

def to_uint8(M: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """
    Z-score -> クリップ -> 0-255 へ線形写像（uint8）
    """
    if M.size == 0:
        return np.zeros((0, mean.shape[0]), np.uint8)
    Z = (M - mean) / std
    Z = np.clip(Z, -clip, +clip)
    U = ((Z + clip) * (255.0 / (2 * clip))).astype(np.uint8)
    return U

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="*.txt（1 APK = 1 行）")
    ap.add_argument("--kv", required=True, help="学習済み KeyedVectors (.kv)")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--scale", choices=["raw", "uint8"], default="raw",
                    help="raw: float32 .npy / uint8: 0-255 画像（.png または .npy）")
    ap.add_argument("--png", action="store_true", help="--scale uint8 時、PNG で保存（Pillow 必須）")
    ap.add_argument("--fit-stats", help="train 用：このディレクトリの *.txt から mean/std を算出して保存する .npz パス")
    ap.add_argument("--stats", help="mean/std を含む .npz を読み込む（val/test 用）")
    ap.add_argument("--clip", type=float, default=3.0, help="Z-score のクリップ閾値（±clip）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kv = KeyedVectors.load(args.kv, mmap="r")

    # 統計
    if args.stats:
        st = np.load(args.stats)
        mean, std = st["mean"], st["std"]
    elif args.fit_stats:
        mean, std = fit_stats_from_dir(args.seq_dir, kv)
        np.savez(args.fit_stats, mean=mean, std=std)
        with open(args.fit_stats + ".meta.json", "w") as f:
            json.dump({"vector_size": int(kv.vector_size)}, f)
        print(f"[embed2img] fitted stats -> {args.fit_stats}")
    else:
        # 統計なし（raw 保存 or 暫定処理）
        mean = np.zeros((kv.vector_size,), np.float32)
        std = np.ones((kv.vector_size,), np.float32)

    pairs = load_sequences(args.seq_dir)
    for stem, toks in pairs:
        M = sequence_to_matrix(toks, kv)  # (L, d)
        if args.scale == "raw":
            # 生の float32 を .npy（L×d, C=1 相当）
            np.save(Path(args.out) / f"{stem}.npy", M.astype(np.float32))
        else:
            U = to_uint8(M, mean, std, clip=args.clip)  # (L, d) -> uint8
            if args.png:
                if Image is None:
                    raise RuntimeError("Pillow が必要です。 pip install pillow")
                if U.size == 0:
                    U2 = np.zeros((1, kv.vector_size), np.uint8)
                else:
                    U2 = U
                Image.fromarray(U2, mode="L").save(Path(args.out) / f"{stem}.png")
            else:
                np.save(Path(args.out) / f"{stem}.npy", U.astype(np.uint8))
    print(f"[embed2img] done: {args.out}")

if __name__ == "__main__":
    main()
