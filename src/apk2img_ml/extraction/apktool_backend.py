from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List
from zipfile import ZIP_DEFLATED, ZipFile

from .config import TokenFilter
from .smali import extract_tokens_from_smali_tree


def _build_apktool_cmd(apk_path: Path, out_dir: Path, apktool_cmd: str, no_res: bool) -> List[str]:
    cmd = [apktool_cmd, "d", str(apk_path), "-o", str(out_dir), "-f"]
    if no_res:
        cmd.insert(2, "--no-res")
    return cmd


def _strip_assets_dex(src_apk: Path, dst_apk: Path) -> bool:
    had_assets_dex = False
    with ZipFile(src_apk, "r") as source_zip, ZipFile(dst_apk, "w", ZIP_DEFLATED) as target_zip:
        for info in source_zip.infolist():
            name = info.filename
            if name.startswith("assets/") and name.lower().endswith(".dex"):
                had_assets_dex = True
                continue
            data = source_zip.read(name)
            target_zip.writestr(info, data)
    return had_assets_dex


def _run_apktool(cmd: List[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return proc.returncode, proc.stdout


def extract_with_apktool(
    apk_path: Path,
    token_filter: TokenFilter,
    *,
    apktool_cmd: str = "apktool",
    no_res: bool = True,
    retry_strip_assets: bool = True,
) -> List[str]:
    """Extract API tokens by decoding with apktool then scanning smali."""

    with tempfile.TemporaryDirectory(prefix="apk2img_apktool_") as tmp:
        temp_root = Path(tmp)
        out_dir = temp_root / "apktool_out"

        rc, output = _run_apktool(_build_apktool_cmd(apk_path, out_dir, apktool_cmd, no_res))

        if rc != 0 and retry_strip_assets:
            stripped_apk = temp_root / f"{apk_path.stem}.noassetsdex.apk"
            had_assets_dex = _strip_assets_dex(apk_path, stripped_apk)
            if had_assets_dex:
                shutil.rmtree(out_dir, ignore_errors=True)
                rc, output = _run_apktool(
                    _build_apktool_cmd(stripped_apk, out_dir, apktool_cmd, no_res)
                )

        has_smali = any(path.is_dir() for path in out_dir.glob("smali*"))
        if rc != 0 and not has_smali:
            tail = output[-2000:] if output else ""
            raise RuntimeError(f"apktool failed rc={rc}. output={tail}")
        if not has_smali:
            raise RuntimeError("apktool produced no smali directories")

        return extract_tokens_from_smali_tree(out_dir, token_filter)
