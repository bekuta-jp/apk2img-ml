#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from PIL import Image

def find_sdhash_image(sdhash_root: Path, rel_parent: Path, stem: str, prefer_suffix: str) -> Path | None:
    """
    doc2vec: stem.png
    sdhash : stem_1ch.png (prefer), or stem_*ch.png
    """
    cand_dir = sdhash_root / rel_parent

    # 1) まずは指定サフィックスを優先
    p1 = cand_dir / f"{stem}{prefer_suffix}.png"
    if p1.exists():
        return p1

    # 2) なければ *_*ch.png を探す（例: _1ch, _4ch など）
    hits = sorted(cand_dir.glob(f"{stem}_*ch.png"))
    if hits:
        return hits[0]

    return None

def to_gray(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img

def main():
    ap = argparse.ArgumentParser(description="Merge sdhash(L) + doc2vec(L) into RGB image (R/G from sdhash halves, B from doc2vec).")
    ap.add_argument("--sdhash-dir", type=Path, required=True, help="Directory of sdhash images (e.g., xxx_1ch.png)")
    ap.add_argument("--doc2vec-dir", type=Path, required=True, help="Directory of doc2vec images (e.g., xxx.png)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--prefer-suffix", type=str, default="_1ch", help="Preferred suffix for sdhash file (default: _1ch)")
    ap.add_argument("--out-suffix", type=str, default="", help="Optional suffix added to output filename stem (e.g., _rgb)")
    ap.add_argument("--width", type=int, default=344, help="Expected width (default: 344)")
    ap.add_argument("--doc-size", type=int, default=344, help="doc2vec size (default: 344; output is doc-size x doc-size)")
    ap.add_argument("--sd-height", type=int, default=688, help="sdhash resized height (default: 688)")
    ap.add_argument("--nearest", action="store_true", help="Use NEAREST resampling for resizing (recommended to keep 0-255 values sharp)")
    args = ap.parse_args()

    sdhash_dir = args.sdhash_dir
    doc_dir = args.doc2vec_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    resample = Image.Resampling.NEAREST if args.nearest else Image.Resampling.BILINEAR

    doc_pngs = sorted(doc_dir.rglob("*.png"))
    made = 0
    skipped = 0

    for doc_path in doc_pngs:
        rel = doc_path.relative_to(doc_dir)
        rel_parent = rel.parent
        stem = doc_path.stem  # 例: 00c16c....f958e

        sd_path = find_sdhash_image(sdhash_dir, rel_parent, stem, args.prefer_suffix)
        if sd_path is None:
            skipped += 1
            continue

        # --- load doc2vec (B) ---
        doc_img = to_gray(Image.open(doc_path))
        if doc_img.size != (args.doc_size, args.doc_size):
            # 念のため矯正（doc2vecが固定サイズ前提なら不要だが保険）
            doc_img = doc_img.resize((args.doc_size, args.doc_size), resample=resample)

        # --- load sdhash and resize to (344, 688) ---
        sd_img = to_gray(Image.open(sd_path))
        if sd_img.size[0] != args.width:
            # 横が344でない場合は強制的に合わせる（縦も含めてリサイズ）
            sd_img = sd_img.resize((args.width, sd_img.size[1]), resample=resample)
        sd_img = sd_img.resize((args.width, args.sd_height), resample=resample)

        # 上半分/下半分を切り出し（各 344 x 344）
        half = args.doc_size
        r = sd_img.crop((0, 0, args.width, half))
        g = sd_img.crop((0, half, args.width, half * 2))
        b = doc_img  # 344x344

        rgb = Image.merge("RGB", (r, g, b))

        out_path = out_dir / rel_parent / f"{stem}{args.out_suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(out_path, optimize=True)
        made += 1

    print(f"[DONE] made={made}, skipped(no pair)={skipped}, scanned(doc2vec)={len(doc_pngs)}")

if __name__ == "__main__":
    main()
