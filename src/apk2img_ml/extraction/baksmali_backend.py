from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List
from zipfile import ZipFile

from .config import TokenFilter
from .smali import extract_tokens_from_smali_tree


def _extract_dex_files(apk_path: Path, dex_dir: Path, include_assets_dex: bool) -> List[Path]:
    dex_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []

    with ZipFile(apk_path, "r") as apk_zip:
        for name in apk_zip.namelist():
            if not name.endswith(".dex"):
                continue
            if (not include_assets_dex) and name.startswith("assets/"):
                continue
            target = dex_dir / Path(name).name
            with apk_zip.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)

    return sorted(extracted)


def extract_with_baksmali(
    apk_path: Path,
    token_filter: TokenFilter,
    *,
    baksmali_jar: Path,
    java_cmd: str = "java",
    include_assets_dex: bool = False,
) -> List[str]:
    """Extract API tokens by disassembling DEX with baksmali."""

    if not baksmali_jar.is_file():
        raise RuntimeError(f"baksmali.jar not found: {baksmali_jar}")

    with tempfile.TemporaryDirectory(prefix="apk2img_baksmali_") as tmp:
        temp_root = Path(tmp)
        dex_dir = temp_root / "dex"
        smali_out = temp_root / "smali"
        dex_files = _extract_dex_files(apk_path, dex_dir, include_assets_dex)

        if not dex_files:
            raise RuntimeError("no dex entries found in APK")

        success = 0
        for dex_file in dex_files:
            cmd = [
                java_cmd,
                "-jar",
                str(baksmali_jar),
                "disassemble",
                "--ignore-errors",
                "-o",
                str(smali_out),
                str(dex_file),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode == 0:
                success += 1

        if success == 0:
            raise RuntimeError("baksmali failed for all dex files")

        return extract_tokens_from_smali_tree(smali_out, token_filter)
