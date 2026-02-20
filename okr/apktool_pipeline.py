#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apktool を用いて APK から smali を抽出するパイプライン。
失敗時は「assets/*.dex を無視した再試行」を自動で行い、
さらに（オプションで）baksmali による per-dex フォールバックも試せる。

【想定する手順（デフォルト）】
1) apktool d を通常実行
2) 失敗 or smali が出ていない場合 → assets/*.dex を除去した一時 APK を作って再試行
3) それでもダメなら（オプション）baksmali で classes*.dex を1本ずつデコード
   ※ assets 内の dex を baksmali 対象に含めるかはオプションで選択可

・Jupyter（import）でも CLI でも実行可
・ログは簡易：標準出力は進捗と結果のみ、詳細は log_dir に出力

Author: you & ChatGPT
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from zipfile import ZipFile, ZIP_DEFLATED

# -------------------------
# 追加: 進捗バー（tqdm は任意）
# CHANGED: 長時間処理の可視化のため（未インストールでも動くフォールバック）
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# 追加: 並列実行（任意）
# CHANGED: ディレクトリ処理を高速化するためオプションで有効化
from concurrent.futures import ProcessPoolExecutor, as_completed


# -------------------------
# オプション（import 用）
# -------------------------
@dataclass
class PipelineOptions:
    apktool_cmd: str = "apktool"           # apktool 実行コマンド
    out_root: Path = Path("./corp_apktool")  # 出力ルート（各APKごとにディレクトリ作成）
    tmp_base: Optional[Path] = None        # 一時作業ルート（未指定なら tempfile.gettempdir()）
    log_dir: Path = Path("./logs_apktool") # 詳細ログ置き場
    diagnostics_csv: Path = Path("./diagnostics.csv")  # 失敗や注意の記録

    # フォールバック挙動
    ignore_assets_on_retry: bool = True    # 失敗した時だけ assets/*.dex を除去して再試行
    enable_baksmali_fallback: bool = False # 上記でも失敗したとき baksmali で per-dex を試す
    include_assets_in_baksmali: bool = False  # baksmali フォールバックに assets/*.dex を含めるか
    baksmali_jar: Path = Path("./baksmali.jar")
    java_cmd: str = "java"
    api_level: Optional[int] = None        # 例: 29（未指定なら付けない）

    # apktool オプション
    no_res: bool = False                   # リソースをデコードしない（必要時のみ）
    force: bool = True                     # 既存出力ディレクトリがあれば上書き

    # 追加: 並列数
    # CHANGED: 既定は 1（従来の挙動を維持）。>1 で並列に。
    workers: int = 1


# -------------------------
# 共通ユーティリティ
# -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _apk_name(apk: Path) -> str:
    return apk.stem


def _has_smali_dirs(out_dir: Path) -> bool:
    return any(d.is_dir() for d in out_dir.glob("smali*"))


def _build_apktool_cmd(apk: Path, out_dir: Path, opt: PipelineOptions) -> List[str]:
    cmd = [opt.apktool_cmd, "d", str(apk), "-o", str(out_dir)]
    if opt.no_res:
        cmd.append("-r")
    if opt.force:
        cmd.append("-f")
    return cmd


def _run(cmd: List[str], log_file: Path) -> int:
    _ensure_dir(log_file.parent)
    with open(log_file, "wb") as lf:
        lf.write((" ".join(cmd) + "\n\n").encode("utf-8"))
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
        return proc.returncode


def _strip_assets_dex(src_apk: Path, dst_apk: Path) -> Tuple[int, int]:
    """
    src_apk から assets/*.dex を除外して dst_apk を作る。
    return: (copied_entries, stripped_entries)
    """
    copied = 0
    stripped = 0
    _ensure_dir(dst_apk.parent)
    with ZipFile(src_apk, "r") as zin, ZipFile(dst_apk, "w", ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            name = info.filename
            # CHANGED: 失敗 APK だけ assets/*.dex を“無視”して再試行するため（情報保全のため通常時は触らない）
            if name.startswith("assets/") and name.lower().endswith(".dex"):
                stripped += 1
                continue
            with zin.open(info, "r") as src:
                data = src.read()
            zout.writestr(info, data)
            copied += 1
    return copied, stripped


def _list_dex_entries(apk: Path, include_assets: bool) -> List[str]:
    """
    APK 内の dex エントリ名を列挙（classes*.dex と、必要なら assets/*.dex）。
    """
    dex_names: List[str] = []
    with ZipFile(apk, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith(".dex"):
                if name.startswith("classes"):
                    dex_names.append(name)
                elif include_assets and name.startswith("assets/"):
                    dex_names.append(name)
    # classes.dex, classes2.dex, ... の順で並ぶよう簡易ソート
    dex_names.sort()
    return dex_names


def _baksmali_one(apk: Path, dex_entry: str, out_dir: Path, opt: PipelineOptions, log_file: Path) -> int:
    """
    baksmali で APK 内の特定 dex エントリだけをデコード
    """
    cmd = [
        opt.java_cmd, "-jar", str(opt.baksmali_jar),
        "disassemble", str(apk),
        "--dex-file", dex_entry,
        "--output", str(out_dir),
    ]
    if opt.api_level is not None:
        cmd += ["--api", str(opt.api_level)]
    return _run(cmd, log_file)


def _decode_with_baksmali_perdex(apk: Path, out_dir: Path, opt: PipelineOptions, log_dir: Path) -> Tuple[int, List[str], List[str]]:
    """
    APK 中の dex を 1本ずつ baksmali でデコード。
    return: (ok_count, failed_entries, skipped_entries)
    """
    dex_entries = _list_dex_entries(apk, include_assets=opt.include_assets_in_baksmali)
    ok = 0
    failed: List[str] = []
    skipped: List[str] = []

    if not dex_entries:
        return 0, failed, skipped

    for i, dex_entry in enumerate(dex_entries, start=1):
        # smali の出力先は classes/, classes2/, ... に合わせる
        if dex_entry.startswith("classes"):
            # classes.dex → smali, classes2.dex → smali_classes2 という慣例に揃える
            suffix = "" if dex_entry == "classes.dex" else "_" + dex_entry.replace(".dex", "")
            dex_out = out_dir / f"smali{suffix}"
        else:
            # assets の dex は区別のためフォルダ名に変換して出力
            safe = dex_entry.replace("/", "_").replace(".dex", "")
            dex_out = out_dir / f"smali_{safe}"

        _ensure_dir(dex_out)
        log_file = log_dir / f"{_apk_name(apk)}__{i:02d}_{dex_entry.replace('/', '_')}.log"
        rc = _baksmali_one(apk, dex_entry, dex_out, opt, log_file)
        if rc == 0:
            ok += 1
        else:
            failed.append(dex_entry)

    return ok, failed, skipped


# -------------------------
# メイン処理（単APK）
# -------------------------
def process_one_apk(apk: Path, out_dir: Path, opt: PipelineOptions) -> Tuple[bool, str]:
    """
    1つの APK を処理。成功/失敗とメッセージを返す。
    成功時 out_dir 配下に smali* が存在する。
    """
    _ensure_dir(out_dir.parent)
    _ensure_dir(opt.log_dir)
    tmp_dir = Path(tempfile.mkdtemp(prefix="apktool_", dir=str(opt.tmp_base) if opt.tmp_base else None))

    try:
        # 1) 通常の apktool d
        log1 = opt.log_dir / f"{_apk_name(apk)}__apktool.log"
        cmd1 = _build_apktool_cmd(apk, out_dir, opt)
        rc1 = _run(cmd1, log1)
        if rc1 == 0 and _has_smali_dirs(out_dir):
            return True, "apktool_ok"

        # 2) 失敗 → assets/*.dex 除去 APK を作成して再試行（※通常は触らない。失敗時のみ）
        if opt.ignore_assets_on_retry:
            sanitized_apk = tmp_dir / f"{apk.name}.noassetsdex.apk"
            copied, stripped = _strip_assets_dex(apk, sanitized_apk)
            log2 = opt.log_dir / f"{_apk_name(apk)}__apktool_noassets.log"
            # 出力ディレクトリを綺麗にしてから再試行（apktool -f でもよいが念のため）
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            rc2 = _run(_build_apktool_cmd(sanitized_apk, out_dir, opt), log2)
            if rc2 == 0 and _has_smali_dirs(out_dir):
                return True, f"apktool_ok_after_ignore_assets (stripped={stripped})"

        # 3) それでもダメ → （オプション）baksmali per-dex フォールバック
        if opt.enable_baksmali_fallback:
            log3_dir = opt.log_dir / f"{_apk_name(apk)}__baksmali"
            _ensure_dir(log3_dir)
            ok_count, failed, _skipped = _decode_with_baksmali_perdex(apk, out_dir, opt, log3_dir)
            if ok_count > 0 and _has_smali_dirs(out_dir):
                return True, f"baksmali_ok (dex_ok={ok_count}, dex_failed={len(failed)})"
            else:
                return False, f"baksmali_failed (dex_ok={ok_count}, dex_failed={len(failed)})"

        return False, "all_failed"

    finally:
        # 一時領域は原則削除（デバッグしたいときはコメントアウト）
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -------------------------
# ディレクトリ処理（複数APK）
# -------------------------
def process_apk_dir(apk_dir: Path, out_root: Path, opt: PipelineOptions) -> Tuple[int, int]:
    """
    ディレクトリ中の *.apk を順に処理。
    return: (success_count, fail_count)
    """
    apks = sorted([p for p in apk_dir.glob("*.apk") if p.is_file()])
    if not apks:
        print(f("[warn] no apk files under: {apk_dir}"))
        return 0, 0

    _ensure_dir(out_root)
    _ensure_dir(opt.log_dir)
    diag_path = opt.diagnostics_csv
    new_file = not diag_path.exists()
    with open(diag_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["apk_name", "status", "note"])

        ok, ng = 0, 0

        # 追加: 進捗バー
        # CHANGED: 視認性向上（tqdm があれば使用）
        pbar = tqdm(total=len(apks), unit="apk", desc="apktool decode", dynamic_ncols=True) if tqdm else None

        if max(1, int(opt.workers)) == 1:
            # 既定: シーケンシャル（従来の挙動）
            for i, apk in enumerate(apks, start=1):
                out_dir = out_root / apk.stem
                msg_head = f"[{i}/{len(apks)}] {apk.name} -> {out_dir.name} ..."
                ok_one, msg = process_one_apk(apk, out_dir, opt)
                if ok_one:
                    ok += 1
                    print(msg_head + " OK")
                    w.writerow([apk.name, "OK", msg])
                else:
                    ng += 1
                    print(msg_head + " FAIL")
                    w.writerow([apk.name, "FAIL", msg])
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"{pbar.n}/{pbar.total}")
        else:
            # 追加: 並列実行
            # CHANGED: 速度向上のため ProcessPoolExecutor をオプションで使用
            with ProcessPoolExecutor(max_workers=int(opt.workers)) as ex:
                futs = {ex.submit(process_one_apk, apk, out_root / apk.stem, opt): apk for apk in apks}
                for i, fut in enumerate(as_completed(futs), start=1):
                    apk = futs[fut]
                    out_dir = out_root / apk.stem
                    msg_head = f"[{i}/{len(apks)}] {apk.name} -> {out_dir.name} ..."
                    try:
                        ok_one, msg = fut.result()
                    except Exception as e:
                        ok_one, msg = False, f"exception:{e}"
                    if ok_one:
                        ok += 1
                        # print(msg_head + " OK")
                        w.writerow([apk.name, "OK", msg])
                    else:
                        ng += 1
                        print(msg_head + " FAIL")
                        w.writerow([apk.name, "FAIL", msg])
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(f"{pbar.n}/{pbar.total}")

        if pbar:
            pbar.close()

    print(f"Completed: {ok} OK, {ng} FAIL (see {diag_path})")
    return ok, ng


# -------------------------
# CLI エントリ
# -------------------------
def _parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser()
    pa.add_argument("--apk", type=Path, help="単一APKを処理（--out-dir と併用）")
    pa.add_argument("--apk-dir", type=Path, help="ディレクトリ中の *.apk を処理（--out-root と併用）")

    # 出力とログ
    pa.add_argument("--out-dir", type=Path, help="単一APKの出力先ディレクトリ")
    pa.add_argument("--out-root", type=Path, default=Path("./corp_apktool"), help="複数APK時の出力ルート")
    pa.add_argument("--log-dir", type=Path, default=Path("./logs_apktool"))
    pa.add_argument("--diagnostics-csv", type=Path, default=Path("./diagnostics.csv"))
    pa.add_argument("--tmp-base", type=Path, help="一時領域（RAMディスク等）")

    # apktool/baksmali
    pa.add_argument("--apktool-cmd", default="apktool")
    pa.add_argument("--no-res", action="store_true", help="apktool: リソースをデコードしない（-r）")
    pa.add_argument("--no-force", action="store_true", help="apktool: -f を使わない（上書き無効）")

    pa.add_argument("--enable-baksmali", action="store_true", help="失敗時の baksmali フォールバックを有効化")
    pa.add_argument("--baksmali-jar", type=Path, default=Path("./baksmali.jar"))
    pa.add_argument("--java-cmd", default="java")
    pa.add_argument("--api-level", type=int, help="baksmali: --api N を付与")
    pa.add_argument("--baksmali-include-assets", action="store_true", help="baksmaliに assets/*.dex も含める")

    # 失敗時 assets 無視の再試行（基本ONの想定だが無効化も可能に）
    pa.add_argument("--no-assets-retry", action="store_true", help="失敗時に assets/*.dex を無視する再試行を行わない")

    # 追加: 並列数
    # CHANGED: 既定1（従来互換）、任意で増やす
    pa.add_argument("--workers", type=int, default=1, help="並列プロセス数（既定1=直列）")

    return pa.parse_args()


def main() -> None:
    args = _parse_args()
    opt = PipelineOptions(
        apktool_cmd=args.apktool_cmd,
        out_root=args.out_root,
        tmp_base=args.tmp_base,
        log_dir=args.log_dir,
        diagnostics_csv=args.diagnostics_csv,
        ignore_assets_on_retry=not args.no_assets_retry,                 # ← デフォルトで有効
        enable_baksmali_fallback=args.enable_baksmali,
        include_assets_in_baksmali=args.baksmali_include_assets,
        baksmali_jar=args.baksmali_jar,
        java_cmd=args.java_cmd,
        api_level=args.api_level,
        no_res=args.no_res,
        force=not args.no_force,
        workers=args.workers,                                           # CHANGED: CLI → Options
    )

    if args.apk and args.out_dir:
        ok, msg = process_one_apk(args.apk, args.out_dir, opt)
        print(f"{args.apk.name}: {'OK' if ok else 'FAIL'} ({msg})")
        return

    if args.apk_dir:
        process_apk_dir(args.apk_dir, opt.out_root, opt)
        return

    print("Specify either --apk + --out-dir (single) or --apk-dir (batch).")
    sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apktool を用いて APK から smali を抽出するパイプライン。
失敗時は「assets/*.dex を無視した再試行」を自動で行い、
さらに（オプションで）baksmali による per-dex フォールバックも試せる。

【想定する手順（デフォルト）】
1) apktool d を通常実行
2) 失敗 or smali が出ていない場合 → assets/*.dex を除去した一時 APK を作って再試行
3) それでもダメなら（オプション）baksmali で classes*.dex を1本ずつデコード
   ※ assets 内の dex を baksmali 対象に含めるかはオプションで選択可

・Jupyter（import）でも CLI でも実行可
・ログは簡易：標準出力は進捗と結果のみ、詳細は log_dir に出力

Author: you & ChatGPT
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from zipfile import ZipFile, ZIP_DEFLATED

# -------------------------
# 追加: 進捗バー（tqdm は任意）
# CHANGED: 長時間処理の可視化のため（未インストールでも動くフォールバック）
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# 追加: 並列実行（任意）
# CHANGED: ディレクトリ処理を高速化するためオプションで有効化
from concurrent.futures import ProcessPoolExecutor, as_completed


# -------------------------
# オプション（import 用）
# -------------------------
@dataclass
class PipelineOptions:
    apktool_cmd: str = "apktool"           # apktool 実行コマンド
    out_root: Path = Path("./corp_apktool")  # 出力ルート（各APKごとにディレクトリ作成）
    tmp_base: Optional[Path] = None        # 一時作業ルート（未指定なら tempfile.gettempdir()）
    log_dir: Path = Path("./logs_apktool") # 詳細ログ置き場
    diagnostics_csv: Path = Path("./diagnostics.csv")  # 失敗や注意の記録

    # フォールバック挙動
    ignore_assets_on_retry: bool = True    # 失敗した時だけ assets/*.dex を除去して再試行
    enable_baksmali_fallback: bool = False # 上記でも失敗したとき baksmali で per-dex を試す
    include_assets_in_baksmali: bool = False  # baksmali フォールバックに assets/*.dex を含めるか
    baksmali_jar: Path = Path("./baksmali.jar")
    java_cmd: str = "java"
    api_level: Optional[int] = None        # 例: 29（未指定なら付けない）

    # apktool オプション
    no_res: bool = False                   # リソースをデコードしない（必要時のみ）
    force: bool = True                     # 既存出力ディレクトリがあれば上書き

    # 追加: 並列数
    # CHANGED: 既定は 1（従来の挙動を維持）。>1 で並列に。
    workers: int = 1


# -------------------------
# 共通ユーティリティ
# -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _apk_name(apk: Path) -> str:
    return apk.stem


def _has_smali_dirs(out_dir: Path) -> bool:
    return any(d.is_dir() for d in out_dir.glob("smali*"))


def _build_apktool_cmd(apk: Path, out_dir: Path, opt: PipelineOptions) -> List[str]:
    cmd = [opt.apktool_cmd, "d", str(apk), "-o", str(out_dir)]
    if opt.no_res:
        cmd.append("-r")
    if opt.force:
        cmd.append("-f")
    return cmd


def _run(cmd: List[str], log_file: Path) -> int:
    _ensure_dir(log_file.parent)
    with open(log_file, "wb") as lf:
        lf.write((" ".join(cmd) + "\n\n").encode("utf-8"))
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
        return proc.returncode


def _strip_assets_dex(src_apk: Path, dst_apk: Path) -> Tuple[int, int]:
    """
    src_apk から assets/*.dex を除外して dst_apk を作る。
    return: (copied_entries, stripped_entries)
    """
    copied = 0
    stripped = 0
    _ensure_dir(dst_apk.parent)
    with ZipFile(src_apk, "r") as zin, ZipFile(dst_apk, "w", ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            name = info.filename
            # CHANGED: 失敗 APK だけ assets/*.dex を“無視”して再試行するため（情報保全のため通常時は触らない）
            if name.startswith("assets/") and name.lower().endswith(".dex"):
                stripped += 1
                continue
            with zin.open(info, "r") as src:
                data = src.read()
            zout.writestr(info, data)
            copied += 1
    return copied, stripped


def _list_dex_entries(apk: Path, include_assets: bool) -> List[str]:
    """
    APK 内の dex エントリ名を列挙（classes*.dex と、必要なら assets/*.dex）。
    """
    dex_names: List[str] = []
    with ZipFile(apk, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith(".dex"):
                if name.startswith("classes"):
                    dex_names.append(name)
                elif include_assets and name.startswith("assets/"):
                    dex_names.append(name)
    # classes.dex, classes2.dex, ... の順で並ぶよう簡易ソート
    dex_names.sort()
    return dex_names


def _baksmali_one(apk: Path, dex_entry: str, out_dir: Path, opt: PipelineOptions, log_file: Path) -> int:
    """
    baksmali で APK 内の特定 dex エントリだけをデコード
    """
    cmd = [
        opt.java_cmd, "-jar", str(opt.baksmali_jar),
        "disassemble", str(apk),
        "--dex-file", dex_entry,
        "--output", str(out_dir),
    ]
    if opt.api_level is not None:
        cmd += ["--api", str(opt.api_level)]
    return _run(cmd, log_file)


def _decode_with_baksmali_perdex(apk: Path, out_dir: Path, opt: PipelineOptions, log_dir: Path) -> Tuple[int, List[str], List[str]]:
    """
    APK 中の dex を 1本ずつ baksmali でデコード。
    return: (ok_count, failed_entries, skipped_entries)
    """
    dex_entries = _list_dex_entries(apk, include_assets=opt.include_assets_in_baksmali)
    ok = 0
    failed: List[str] = []
    skipped: List[str] = []

    if not dex_entries:
        return 0, failed, skipped

    for i, dex_entry in enumerate(dex_entries, start=1):
        # smali の出力先は classes/, classes2/, ... に合わせる
        if dex_entry.startswith("classes"):
            # classes.dex → smali, classes2.dex → smali_classes2 という慣例に揃える
            suffix = "" if dex_entry == "classes.dex" else "_" + dex_entry.replace(".dex", "")
            dex_out = out_dir / f"smali{suffix}"
        else:
            # assets の dex は区別のためフォルダ名に変換して出力
            safe = dex_entry.replace("/", "_").replace(".dex", "")
            dex_out = out_dir / f"smali_{safe}"

        _ensure_dir(dex_out)
        log_file = log_dir / f"{_apk_name(apk)}__{i:02d}_{dex_entry.replace('/', '_')}.log"
        rc = _baksmali_one(apk, dex_entry, dex_out, opt, log_file)
        if rc == 0:
            ok += 1
        else:
            failed.append(dex_entry)

    return ok, failed, skipped


# -------------------------
# メイン処理（単APK）
# -------------------------
def process_one_apk(apk: Path, out_dir: Path, opt: PipelineOptions) -> Tuple[bool, str]:
    """
    1つの APK を処理。成功/失敗とメッセージを返す。
    成功時 out_dir 配下に smali* が存在する。
    """
    _ensure_dir(out_dir.parent)
    _ensure_dir(opt.log_dir)
    tmp_dir = Path(tempfile.mkdtemp(prefix="apktool_", dir=str(opt.tmp_base) if opt.tmp_base else None))

    try:
        # 1) 通常の apktool d
        log1 = opt.log_dir / f"{_apk_name(apk)}__apktool.log"
        cmd1 = _build_apktool_cmd(apk, out_dir, opt)
        rc1 = _run(cmd1, log1)
        if rc1 == 0 and _has_smali_dirs(out_dir):
            return True, "apktool_ok"

        # 2) 失敗 → assets/*.dex 除去 APK を作成して再試行（※通常は触らない。失敗時のみ）
        if opt.ignore_assets_on_retry:
            sanitized_apk = tmp_dir / f"{apk.name}.noassetsdex.apk"
            copied, stripped = _strip_assets_dex(apk, sanitized_apk)
            log2 = opt.log_dir / f"{_apk_name(apk)}__apktool_noassets.log"
            # 出力ディレクトリを綺麗にしてから再試行（apktool -f でもよいが念のため）
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            rc2 = _run(_build_apktool_cmd(sanitized_apk, out_dir, opt), log2)
            if rc2 == 0 and _has_smali_dirs(out_dir):
                return True, f"apktool_ok_after_ignore_assets (stripped={stripped})"

        # 3) それでもダメ → （オプション）baksmali per-dex フォールバック
        if opt.enable_baksmali_fallback:
            log3_dir = opt.log_dir / f"{_apk_name(apk)}__baksmali"
            _ensure_dir(log3_dir)
            ok_count, failed, _skipped = _decode_with_baksmali_perdex(apk, out_dir, opt, log3_dir)
            if ok_count > 0 and _has_smali_dirs(out_dir):
                return True, f"baksmali_ok (dex_ok={ok_count}, dex_failed={len(failed)})"
            else:
                return False, f"baksmali_failed (dex_ok={ok_count}, dex_failed={len(failed)})"

        return False, "all_failed"

    finally:
        # 一時領域は原則削除（デバッグしたいときはコメントアウト）
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -------------------------
# ディレクトリ処理（複数APK）
# -------------------------
def process_apk_dir(apk_dir: Path, out_root: Path, opt: PipelineOptions) -> Tuple[int, int]:
    """
    ディレクトリ中の *.apk を順に処理。
    return: (success_count, fail_count)
    """
    apks = sorted([p for p in apk_dir.glob("*.apk") if p.is_file()])
    if not apks:
        print(f("[warn] no apk files under: {apk_dir}"))
        return 0, 0

    _ensure_dir(out_root)
    _ensure_dir(opt.log_dir)
    diag_path = opt.diagnostics_csv
    new_file = not diag_path.exists()
    with open(diag_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["apk_name", "status", "note"])

        ok, ng = 0, 0

        # 追加: 進捗バー
        # CHANGED: 視認性向上（tqdm があれば使用）
        pbar = tqdm(total=len(apks), unit="apk", desc="apktool decode", dynamic_ncols=True) if tqdm else None

        if max(1, int(opt.workers)) == 1:
            # 既定: シーケンシャル（従来の挙動）
            for i, apk in enumerate(apks, start=1):
                out_dir = out_root / apk.stem
                msg_head = f"[{i}/{len(apks)}] {apk.name} -> {out_dir.name} ..."
                ok_one, msg = process_one_apk(apk, out_dir, opt)
                if ok_one:
                    ok += 1
                    print(msg_head + " OK")
                    w.writerow([apk.name, "OK", msg])
                else:
                    ng += 1
                    print(msg_head + " FAIL")
                    w.writerow([apk.name, "FAIL", msg])
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"{pbar.n}/{pbar.total}")
        else:
            # 追加: 並列実行
            # CHANGED: 速度向上のため ProcessPoolExecutor をオプションで使用
            with ProcessPoolExecutor(max_workers=int(opt.workers)) as ex:
                futs = {ex.submit(process_one_apk, apk, out_root / apk.stem, opt): apk for apk in apks}
                for i, fut in enumerate(as_completed(futs), start=1):
                    apk = futs[fut]
                    out_dir = out_root / apk.stem
                    msg_head = f"[{i}/{len(apks)}] {apk.name} -> {out_dir.name} ..."
                    try:
                        ok_one, msg = fut.result()
                    except Exception as e:
                        ok_one, msg = False, f"exception:{e}"
                    if ok_one:
                        ok += 1
                        # print(msg_head + " OK")
                        w.writerow([apk.name, "OK", msg])
                    else:
                        ng += 1
                        print(msg_head + " FAIL")
                        w.writerow([apk.name, "FAIL", msg])
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(f"{pbar.n}/{pbar.total}")

        if pbar:
            pbar.close()

    print(f"Completed: {ok} OK, {ng} FAIL (see {diag_path})")
    return ok, ng


# -------------------------
# CLI エントリ
# -------------------------
def _parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser()
    pa.add_argument("--apk", type=Path, help="単一APKを処理（--out-dir と併用）")
    pa.add_argument("--apk-dir", type=Path, help="ディレクトリ中の *.apk を処理（--out-root と併用）")

    # 出力とログ
    pa.add_argument("--out-dir", type=Path, help="単一APKの出力先ディレクトリ")
    pa.add_argument("--out-root", type=Path, default=Path("./corp_apktool"), help="複数APK時の出力ルート")
    pa.add_argument("--log-dir", type=Path, default=Path("./logs_apktool"))
    pa.add_argument("--diagnostics-csv", type=Path, default=Path("./diagnostics.csv"))
    pa.add_argument("--tmp-base", type=Path, help="一時領域（RAMディスク等）")

    # apktool/baksmali
    pa.add_argument("--apktool-cmd", default="apktool")
    pa.add_argument("--no-res", action="store_true", help="apktool: リソースをデコードしない（-r）")
    pa.add_argument("--no-force", action="store_true", help="apktool: -f を使わない（上書き無効）")

    pa.add_argument("--enable-baksmali", action="store_true", help="失敗時の baksmali フォールバックを有効化")
    pa.add_argument("--baksmali-jar", type=Path, default=Path("./baksmali.jar"))
    pa.add_argument("--java-cmd", default="java")
    pa.add_argument("--api-level", type=int, help="baksmali: --api N を付与")
    pa.add_argument("--baksmali-include-assets", action="store_true", help="baksmaliに assets/*.dex も含める")

    # 失敗時 assets 無視の再試行（基本ONの想定だが無効化も可能に）
    pa.add_argument("--no-assets-retry", action="store_true", help="失敗時に assets/*.dex を無視する再試行を行わない")

    # 追加: 並列数
    # CHANGED: 既定1（従来互換）、任意で増やす
    pa.add_argument("--workers", type=int, default=1, help="並列プロセス数（既定1=直列）")

    return pa.parse_args()


def main() -> None:
    args = _parse_args()
    opt = PipelineOptions(
        apktool_cmd=args.apktool_cmd,
        out_root=args.out_root,
        tmp_base=args.tmp_base,
        log_dir=args.log_dir,
        diagnostics_csv=args.diagnostics_csv,
        ignore_assets_on_retry=not args.no_assets_retry,                 # ← デフォルトで有効
        enable_baksmali_fallback=args.enable_baksmali,
        include_assets_in_baksmali=args.baksmali_include_assets,
        baksmali_jar=args.baksmali_jar,
        java_cmd=args.java_cmd,
        api_level=args.api_level,
        no_res=args.no_res,
        force=not args.no_force,
        workers=args.workers,                                           # CHANGED: CLI → Options
    )

    if args.apk and args.out_dir:
        ok, msg = process_one_apk(args.apk, args.out_dir, opt)
        print(f"{args.apk.name}: {'OK' if ok else 'FAIL'} ({msg})")
        return

    if args.apk_dir:
        process_apk_dir(args.apk_dir, opt.out_root, opt)
        return

    print("Specify either --apk + --out-dir (single) or --apk-dir (batch).")
    sys.exit(1)


if __name__ == "__main__":
    main()
