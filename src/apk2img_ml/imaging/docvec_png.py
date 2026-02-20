from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def load_docvecs_tsv(tsv_path: Path) -> list[tuple[str, np.ndarray]]:
    docs: list[tuple[str, np.ndarray]] = []

    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for i, row in enumerate(reader):
            if not row:
                continue
            if i == 0 and row[0] == "tag":
                continue
            try:
                vector = np.array([float(x) for x in row[1:]], dtype=np.float32)
            except ValueError:
                continue
            docs.append((row[0], vector))

    return docs


def _encode_flat(vec: np.ndarray, pixels_per_vector: int = 0) -> np.ndarray:
    if vec.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if pixels_per_vector and pixels_per_vector > 0:
        target_len = int(pixels_per_vector)
        x_old = np.linspace(0, vec.size - 1, num=vec.size, dtype=np.float32)
        x_new = np.linspace(0, vec.size - 1, num=target_len, dtype=np.float32)
        base = np.interp(x_new, x_old, vec.astype(np.float32))
    else:
        base = vec.astype(np.float32, copy=False)

    side = int(math.ceil(math.sqrt(base.size)))
    padded = np.zeros((side * side,), dtype=np.float32)
    padded[: base.size] = base

    vmin = float(padded.min())
    vmax = float(padded.max())
    if vmax <= vmin:
        return np.zeros((side, side), dtype=np.uint8)

    norm = (padded - vmin) / (vmax - vmin)
    return (norm * 255.0).clip(0, 255).astype(np.uint8).reshape((side, side))


def _encode_grid(
    vec: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
    vec_pix_rows: int,
    vec_pix_cols: int,
) -> np.ndarray:
    if vec.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    rows = max(grid_rows, 1)
    cols = max(grid_cols, 1)
    cell_count = rows * cols

    values = np.zeros((cell_count,), dtype=np.float32)
    copy_count = min(vec.size, cell_count)
    values[:copy_count] = vec[:copy_count]

    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        cell_u8 = np.zeros((cell_count,), dtype=np.uint8)
    else:
        cell_u8 = (((values - vmin) / (vmax - vmin)) * 255.0).clip(0, 255).astype(np.uint8)

    vpr = max(int(vec_pix_rows), 1)
    vpc = max(int(vec_pix_cols), 1)
    image = np.zeros((rows * vpr, cols * vpc), dtype=np.uint8)

    for idx, value in enumerate(cell_u8):
        r = idx // cols
        c = idx % cols
        image[r * vpr : (r + 1) * vpr, c * vpc : (c + 1) * vpc] = int(value)

    return image


def _encode_raw32(
    vec: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
    vec_pix_rows: int,
    vec_pix_cols: int,
) -> np.ndarray:
    if vec.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    raw = np.frombuffer(vec.astype(np.float32, copy=False).tobytes(), dtype=np.uint8)
    side = int(math.ceil(math.sqrt(raw.size)))
    padded = np.zeros((side * side,), dtype=np.uint8)
    padded[: raw.size] = raw
    base = padded.reshape((side, side))

    if grid_rows > 0 and grid_cols > 0 and vec_pix_rows > 0 and vec_pix_cols > 0:
        target_h = grid_rows * vec_pix_rows
        target_w = grid_cols * vec_pix_cols
        return np.array(
            Image.fromarray(base, mode="L").resize((target_w, target_h), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )

    return base


def encode_vector(
    vec: np.ndarray,
    *,
    mode: str = "flat",
    pixels_per_vector: int = 0,
    grid_rows: int = 0,
    grid_cols: int = 0,
    vec_pix_rows: int = 1,
    vec_pix_cols: int = 1,
) -> np.ndarray:
    if mode == "grid":
        return _encode_grid(
            vec,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            vec_pix_rows=vec_pix_rows,
            vec_pix_cols=vec_pix_cols,
        )
    if mode == "raw32":
        return _encode_raw32(
            vec,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            vec_pix_rows=vec_pix_rows,
            vec_pix_cols=vec_pix_cols,
        )
    return _encode_flat(vec, pixels_per_vector=pixels_per_vector)


def save_docvecs_as_png(
    docs: Sequence[tuple[str, np.ndarray]],
    output_dir: Path,
    *,
    mode: str = "flat",
    pixels_per_vector: int = 0,
    grid_rows: int = 0,
    grid_cols: int = 0,
    vec_pix_rows: int = 1,
    vec_pix_cols: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag, vector in docs:
        rel = Path(tag)
        target_dir = output_dir / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        array = encode_vector(
            vector,
            mode=mode,
            pixels_per_vector=pixels_per_vector,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            vec_pix_rows=vec_pix_rows,
            vec_pix_cols=vec_pix_cols,
        )
        Image.fromarray(array, mode="L").save(target_dir / f"{rel.stem}.png")
