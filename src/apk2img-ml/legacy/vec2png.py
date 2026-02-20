#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Doc2Vec の文書ベクトル (TSV) をグレースケール PNG に変換するスクリプト。

入力 TSV 形式 (train_doc2vec_from_api_tokens.py が出力したものを想定):
    tag    v0  v1  v2  ... v{d-1}

処理:
  - 1 行 = 1 APK とみなし、ベクトル値を読み込む

  encode-mode によって挙動が変わる:

  * flat:
      - ベクトル長 d に対して、side = ceil(sqrt(d)) を求める
      - side*side 要素の 1 次元配列を用意し、先頭 d 要素にベクトル値を入れ、残りは 0 で埋める
      - 配列全体を min-max 正規化して 0〜255 にスケール
      - (side, side) に reshape して画像化

  * grid:
      - grid_rows × grid_cols 個のセルにベクトル値を並べる（足りない分は 0 でパディング、超えた分は切り捨て）
      - セル値をベクトル全体で min-max 正規化 → 0〜255 にスケール
      - 各セルを vec_pix_rows × vec_pix_cols のブロックとして、その値で一様に塗る
      - 結果の画像サイズは:
          高さ = grid_rows * vec_pix_rows
          幅   = grid_cols * vec_pix_cols

  * raw32:
      - ベクトルを float32 のままバイト列 (uint8) に変換（4 * 次元数 バイト = 32bit * 次元数）
      - そのバイト列を最小の正方形(side_raw × side_raw)に詰める（不足分は 0）
      - これが「生ビットを完全保存したベース画像」となる
      - grid_rows, grid_cols, vec_pix_rows, vec_pix_cols が全て >0 の場合:
          高さ_final = grid_rows * vec_pix_rows
          幅_final   = grid_cols * vec_pix_cols
        として、ベース画像をそのサイズにリサイズして出力
      - いずれかが 0 の場合は、ベース画像サイズのまま出力

出力:
  - output_dir/ 以下に tag のディレクトリ構造を再現して PNG を保存する
    例)
      tag: benign/app1.txt
      → output_dir/benign/app1.png
"""

import argparse
from pathlib import Path
import math
import csv

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_docvecs_tsv(tsv_path: Path):
    """
    api_docvecs.tsv を読み込み、(tag, vector[np.ndarray]) のリストを返す。
    1 行目がヘッダ (先頭セル 'tag') の場合はスキップする。
    """
    docs = []

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if not row:
                continue
            # 1 列目が "tag" ならヘッダとみなしてスキップ
            if i == 0 and row[0] == "tag":
                continue
            tag = row[0]
            # 残りを float ベクトルとして読み込む
            try:
                vec = np.array([float(x) for x in row[1:]], dtype=np.float32)
            except ValueError:
                # 数値でない行が混ざっていた場合はスキップ
                continue
            docs.append((tag, vec))

    return docs


def vector_to_image_array_flat(vec: np.ndarray, pixels_per_vector: int = 0) -> np.ndarray:
    """
    ★ADDED: encode-mode = 'flat' 用
    ベクトル全体を 0〜255 に min-max 正規化して、
    「最小の正方形」に敷き詰めるシンプルな 8bit 画像。
    """
    d = vec.shape[0]
    if d == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    # pixels_per_vector が指定されていれば、長さをリサンプリング
    if pixels_per_vector is not None and pixels_per_vector > 0:
        target_len = int(pixels_per_vector)
    else:
        target_len = d

    if target_len <= 0:
        target_len = d

    if target_len == d:
        base_vec = vec.astype(np.float32, copy=False)
    else:
        x_old = np.linspace(0, d - 1, num=d, dtype=np.float32)
        x_new = np.linspace(0, d - 1, num=target_len, dtype=np.float32)
        base_vec = np.interp(x_new, x_old, vec.astype(np.float32))

    effective_len = base_vec.shape[0]
    side = int(math.ceil(math.sqrt(effective_len)))

    arr = np.zeros((side * side,), dtype=np.float32)
    arr[:effective_len] = base_vec

    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmax > vmin:
        norm = (arr - vmin) / (vmax - vmin)
        arr_uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr_uint8 = np.zeros_like(arr, dtype=np.uint8)

    img = arr_uint8.reshape((side, side))
    return img


def vector_to_image_array_grid(
    vec: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    vec_pix_rows: int,
    vec_pix_cols: int,
) -> np.ndarray:
    """
    ★ADDED: encode-mode = 'grid' 用
    ベクトルを grid_rows × grid_cols 個のセルに並べ、
    各セル値をベクトル全体で min-max 正規化した 0〜255 に量子化。
    その値で vec_pix_rows × vec_pix_cols のブロックを一様に塗る。
    """
    d = vec.shape[0]
    if d == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if grid_rows <= 0 or grid_cols <= 0:
        # グリッド指定がなければ flat と同じ扱いでもよいが、
        # ここでは安全のため 1x1 にしておく
        grid_rows = 1
        grid_cols = 1

    vpr = max(int(vec_pix_rows), 1)
    vpc = max(int(vec_pix_cols), 1)

    num_cells = grid_rows * grid_cols

    # ベクトルをセル数に合わせる（足りない分は 0、超えた分は切り捨て）
    vals = np.zeros((num_cells,), dtype=np.float32)
    n_copy = min(d, num_cells)
    vals[:n_copy] = vec[:n_copy]

    vmin = float(vals.min())
    vmax = float(vals.max())

    if vmax > vmin:
        norm_vals = (vals - vmin) / (vmax - vmin)  # 0〜1
        cell_uint8 = (norm_vals * 255.0).clip(0, 255).astype(np.uint8)
    else:
        cell_uint8 = np.zeros_like(vals, dtype=np.uint8)

    img_h = grid_rows * vpr
    img_w = grid_cols * vpc
    img = np.zeros((img_h, img_w), dtype=np.uint8)

    for idx in range(num_cells):
        val = int(cell_uint8[idx])
        cell_r = idx // grid_cols
        cell_c = idx % grid_cols

        y0 = cell_r * vpr
        y1 = y0 + vpr
        x0 = cell_c * vpc
        x1 = x0 + vpc

        img[y0:y1, x0:x1] = val

    return img


def vector_to_image_array_raw32(
    vec: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    vec_pix_rows: int,
    vec_pix_cols: int,
) -> np.ndarray:
    """
    ★ADDED: encode-mode = 'raw32' 用

    ベクトルを float32 のままバイト列 (uint8) に変換し、
    それを最小の正方形に詰めて「生ビットを完全保存したベース画像」を作る。
    その後、grid_rows, grid_cols, vec_pix_rows, vec_pix_cols が指定されていれば
    そのサイズにリサイズして返す。
    """
    d = vec.shape[0]
    if d == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    # float32 → 生ビット列 (uint8)
    vec_f32 = vec.astype(np.float32, copy=False)
    bytes_arr = np.frombuffer(vec_f32.tobytes(), dtype=np.uint8)  # 長さ 4*d

    num_bytes = bytes_arr.shape[0]
    side_raw = int(math.ceil(math.sqrt(num_bytes)))

    arr = np.zeros((side_raw * side_raw,), dtype=np.uint8)
    arr[:num_bytes] = bytes_arr

    base_img = arr.reshape((side_raw, side_raw))

    # リサイズ指定があれば、そこで拡大
    if grid_rows > 0 and grid_cols > 0 and vec_pix_rows > 0 and vec_pix_cols > 0:
        final_h = grid_rows * vec_pix_rows
        final_w = grid_cols * vec_pix_cols

        pil_img = Image.fromarray(base_img, mode="L")
        # ビットパターン自体は崩れるが、「16x16の生ビット画像を拡大したもの」として扱う
        resized = pil_img.resize((final_w, final_h), resample=Image.NEAREST)
        return np.array(resized, dtype=np.uint8)
    else:
        # ベース画像サイズのまま返す
        return base_img


def vector_to_image_array(
    vec: np.ndarray,
    encode_mode: str = "flat",        # ★ADDED: エンコードモード
    pixels_per_vector: int = 0,
    grid_rows: int = 0,
    grid_cols: int = 0,
    vec_pix_rows: int = 1,
    vec_pix_cols: int = 1,
) -> np.ndarray:
    """
    ベクトルを encode-mode に応じて画像配列(uint8 2D)に変換するラッパー。
    """
    if encode_mode == "raw32":
        return vector_to_image_array_raw32(
            vec,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            vec_pix_rows=vec_pix_rows,
            vec_pix_cols=vec_pix_cols,
        )
    elif encode_mode == "grid":
        return vector_to_image_array_grid(
            vec,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            vec_pix_rows=vec_pix_rows,
            vec_pix_cols=vec_pix_cols,
        )
    else:
        # "flat" または不正値 → デフォルトは flat として扱う
        return vector_to_image_array_flat(vec, pixels_per_vector=pixels_per_vector)


def save_docvecs_as_png(
    docs,
    output_dir: Path,
    pixels_per_vector: int = 0,
    grid_rows: int = 0,
    grid_cols: int = 0,
    vec_pix_rows: int = 1,
    vec_pix_cols: int = 1,
    encode_mode: str = "flat",  # ★ADDED
):
    """
    (tag, vector) のリストを PNG として output_dir 以下に保存する。

    tag が "benign/app1.txt" の場合:
      - 出力先パス: output_dir / "benign/app1.png"
    """
    for tag, vec in tqdm(docs, desc="Saving PNG images"):
        # tag から拡張子を除き、パスとして扱う
        # 例: benign/app1.txt → benign/app1
        rel_path = Path(tag)
        stem = rel_path.stem  # app1
        parent = rel_path.parent  # benign
        out_dir_for_tag = output_dir / parent
        out_dir_for_tag.mkdir(parents=True, exist_ok=True)

        out_path = out_dir_for_tag / f"{stem}.png"

        img_array = vector_to_image_array(
            vec,
            encode_mode=encode_mode,
            pixels_per_vector=pixels_per_vector,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            vec_pix_rows=vec_pix_rows,
            vec_pix_cols=vec_pix_cols,
        )
        img = Image.fromarray(img_array, mode="L")
        img.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Doc2Vec 文書ベクトル (TSV) からグレースケール PNG を生成するスクリプト"
    )
    parser.add_argument(
        "docvecs_tsv",
        type=str,
        help="Doc2Vec 文書ベクトルの TSV ファイルパス (例: api_docvecs.tsv)",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="PNG 画像の出力先ルートディレクトリ",
    )
    # 従来のベクトルあたりピクセル数指定（flat モードでのみ使用）
    parser.add_argument(
        "--pixels-per-vector",
        type=int,
        default=0,
        help=(
            "[flat モード用] 1 ベクトルを何ピクセルで表現するかを指定 "
            "(0 以下の場合は元のベクトル長を使用して最小の正方形に敷き詰める)"
        ),
    )
    # グリッドレイアウト指定（grid モード、raw32 のリサイズ時に使用）
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=0,
        help="縦方向に並べる要素数 (grid/raw32 モードで使用)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=0,
        help="横方向に並べる要素数 (grid/raw32 モードで使用)",
    )
    parser.add_argument(
        "--vec-pix-rows",
        type=int,
        default=1,
        help="各要素を縦何ピクセルで描画するか (grid/raw32 モードで使用)",
    )
    parser.add_argument(
        "--vec-pix-cols",
        type=int,
        default=1,
        help="各要素を横何ピクセルで描画するか (grid/raw32 モードで使用)",
    )
    # ★ADDED: エンコードモード選択
    parser.add_argument(
        "--encode-mode",
        type=str,
        default="flat",
        choices=["flat", "grid", "raw32"],
        help=(
            "ベクトル→画像のエンコード方法を指定: "
            "'flat' = min-max量子化して最小正方形, "
            "'grid' = min-max量子化した値をグリッド＋ブロックで拡大, "
            "'raw32' = float32 の生ビットを最小正方形に詰めて、任意サイズに拡大"
        ),
    )

    args = parser.parse_args()

    tsv_path = Path(args.docvecs_tsv).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not tsv_path.is_file():
        raise FileNotFoundError(f"TSV ファイルが見つかりません: {tsv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # TSV 読み込み
    docs = load_docvecs_tsv(tsv_path)
    if not docs:
        raise RuntimeError("有効な文書ベクトルが 1 つもありません。TSV の中身を確認してください。")

    print(f"[INFO] Total documents: {len(docs)}")

    # PNG 保存
    save_docvecs_as_png(
        docs,
        output_dir,
        pixels_per_vector=args.pixels_per_vector,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        vec_pix_rows=args.vec_pix_rows,
        vec_pix_cols=args.vec_pix_cols,
        encode_mode=args.encode_mode,  # ★CHANGED
    )

    print(f"[INFO] Finished. PNG images saved under: {output_dir}")


if __name__ == "__main__":
    main()
