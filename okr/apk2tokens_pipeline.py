#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APK → APIトークン抽出パイプライン（並列・再試行・RAMディスク掃除・静かなログ）

フロー:
 1) RAMディスク等の tmp_base に per-APK の作業ディレクトリ作成
 2) apktool d を実行（ログはファイルへ。stderrはERRORのみ）
 3) smali* が無ければ、assets/*.dex を除外した一時APKを作り再試行
 4) 成功したら smali から invoke-* を正規表現で抽出し、APIトークン列(.txt)へ保存
 5) 中間ファイル(作業ディレクトリ)は必ず削除
 6) 全体は ProcessPoolExecutor で並列、tqdmで進捗を表示

ポイント:
 - 標準出力は進捗と最終サマリのみ。詳細ログは log_dir に保存。エラーのみ stderr。
 - include/exclude の接頭辞やシグネチャ付与はオプションで調整可能。
 - Jupyterからは import して run_batch(...) を呼べる。CLIも可。
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from zipfile import ZipFile, ZIP_DEFLATED

# tqdm（進捗）
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # tqdmが無くても動く

# =========================
# 設定用データクラス
# =========================
@dataclass
class Options:
    apktool_cmd: str = "apktool"
    apk_dir: Path = Path(".")                 # 入力APKディレクトリ（*.apk）
    tokens_root: Path = Path("./corp_tokens") # 出力(1APK=1行のトークン.txt)
    log_dir: Path = Path("./logs_apktool")    # 詳細ログ格納
    diagnostics_csv: Path = Path("./diagnostics.csv")

    tmp_base: Optional[Path] = None           # 例) Path("/dev/shm/apkwork")（RAMディスク）
    workers: int = 1                          # 並列度

    include_prefixes: Tuple[str, ...] = ()              # 例) ("Landroid/","Ljava/","Lkotlin/")
    exclude_prefixes: Tuple[str, ...] = ("Landroidx/test/",)
    include_signature: bool = False

    apktool_no_res: bool = False              # -r
    apktool_force: bool = True                # -f

    # 失敗時の再試行: assets/*.dex を除外して再試行する
    retry_ignore_assets: bool = True


# =========================
# ログ＆ユーティリティ
# =========================
def _setup_root_logging() -> None:
    """ルートロガーをERRORのみstderrに出す。詳細は個別ファイルへ。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.ERROR)
    err_h = logging.StreamHandler(sys.stderr)
    err_h.setLevel(logging.ERROR)
    err_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(err_h)


def _logger_for(apk_name: str, log_dir: Path) -> logging.Logger:
    """APKごとのファイルロガー（INFO/DEBUG）を返す（stderrへは流さない）"""
    lg = logging.getLogger(f"apktool.{apk_name}")
    if not lg.handlers:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{apk_name}.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        lg.addHandler(fh)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _apk_stem(p: Path) -> str:
    return p.stem


def _has_smali_dirs(out_dir: Path) -> bool:
    return any(d.is_dir() for d in out_dir.glob("smali*"))


# =========================
# apktool 実行まわり
# =========================
def _build_apktool_cmd(apk: Path, out_dir: Path, opt: Options) -> List[str]:
    cmd = [opt.apktool_cmd, "d", str(apk), "-o", str(out_dir)]
    if opt.apktool_no_res:
        cmd.append("-r")
    if opt.apktool_force:
        cmd.append("-f")
    return cmd


def _run(cmd: List[str], logger: logging.Logger) -> int:
    """サブプロセスを実行。標準出力はロガーのファイルへ。"""
    logger.info("RUN: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False
        )
        out = proc.stdout.decode("utf-8", errors="replace")
        if out:
            logger.info("OUTPUT:\n%s", out)
        return proc.returncode
    except Exception as e:
        logger.error("subprocess failed: %s", e)
        return 99


def _strip_assets_dex(src_apk: Path, dst_apk: Path, logger: logging.Logger) -> Tuple[int, int]:
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
            if name.startswith("assets/") and name.lower().endswith(".dex"):
                stripped += 1
                continue
            with zin.open(info, "r") as src:
                data = src.read()
            zout.writestr(info, data)
            copied += 1
    logger.info("strip assets/*.dex: copied=%d stripped=%d", copied, stripped)
    return copied, stripped


# =========================
# トークン抽出（smali の正規表現）
# =========================
# 例: invoke-virtual {v0,v1}, Landroid/telephony/SmsManager;->sendTextMessage(Ljava/lang/String;...)V
_INVOKE_RE = re.compile(
    r'^\s*invoke-(?:virtual|direct|static|interface|super)[^,]*,\s*'
    r'(L[^;]+;)->([^\(\s;]+)\s*'       # クラス, メソッド
    r'(\([^\)]*\)[^\s;]+)?',           # ()V 等のシグネチャ（任意）
    re.IGNORECASE
)

def _emit_tokens_from_smali(apk_out_dir: Path,
                            include_prefixes: Optional[Tuple[str, ...]],
                            exclude_prefixes: Optional[Tuple[str, ...]],
                            include_signature: bool,
                            logger: logging.Logger) -> List[str]:
    tokens: List[str] = []
    smali_dirs = [p for p in apk_out_dir.glob("smali*") if p.is_dir()]
    if not smali_dirs:
        logger.info("no smali* dirs for token emission")
        return tokens

    for sdir in smali_dirs:
        for smali in sdir.rglob("*.smali"):
            try:
                with open(smali, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        m = _INVOKE_RE.match(line)
                        if not m:
                            continue
                        clazz, method, sig = m.group(1), m.group(2), m.group(3)
                        if include_prefixes is not None and include_prefixes:
                            if not any(clazz.startswith(pref) for pref in include_prefixes):
                                continue
                        if exclude_prefixes is not None and exclude_prefixes:
                            if any(clazz.startswith(pref) for pref in exclude_prefixes):
                                continue
                        if include_signature and sig:
                            tokens.append(f"{clazz}->{method}{sig}")
                        else:
                            tokens.append(f"{clazz}->{method}")
            except Exception as e:
                logger.info("skip broken smali %s: %s", smali, e)
                continue
    logger.info("tokens collected: %d", len(tokens))
    return tokens


# =========================
# 単一APKの処理
# =========================
def process_one_apk(
    apk: Path,
    opt: Options,
) -> Tuple[str, bool, str]:
    """
    1 APK を処理して (apk名, OK/NG, メッセージ) を返す。
    エラーは stderr（親プロセス側）へ最終まとめのみ出す。詳細はファイルログ。
    """
    apk_name = _apk_stem(apk)
    logger = _logger_for(apk_name, opt.log_dir)

    # 作業ディレクトリは RAM ディスク側に作る
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f"apktool_{apk_name}_", dir=str(opt.tmp_base) if opt.tmp_base else None)
    )
    out_dir = tmp_dir / "apktool_out"
    try:
        # 1) 通常 apktool d
        cmd = _build_apktool_cmd(apk, out_dir, opt)
        rc = _run(cmd, logger)
        if rc == 0 and _has_smali_dirs(out_dir):
            logger.info("apktool_ok")

        else:
            # 2) 失敗 → assets/*.dex を除去して再試行
            if opt.retry_ignore_assets:
                sanitized = tmp_dir / f"{apk.name}.noassetsdex.apk"
                _strip_assets_dex(apk, sanitized, logger)
                # 消し直し（-fでもOKだが念のためクリーン）
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                rc2 = _run(_build_apktool_cmd(sanitized, out_dir, opt), logger)
                if not (rc2 == 0 and _has_smali_dirs(out_dir)):
                    return apk.name, False, "apktool_failed_even_after_ignore_assets"
                logger.info("apktool_ok_after_ignore_assets")
            else:
                return apk.name, False, "apktool_failed"

        # 3) smali → トークン抽出 ＆ 保存
        tokens = _emit_tokens_from_smali(
            out_dir,
            opt.include_prefixes or None,
            opt.exclude_prefixes or None,
            opt.include_signature,
            logger
        )
        _ensure_dir(opt.tokens_root)
        out_txt = opt.tokens_root / f"{apk.stem}.txt"
        with open(out_txt, "w", encoding="utf-8") as wf:
            wf.write(" ".join(tokens) + "\n")
        logger.info("wrote tokens: %s (%d)", out_txt, len(tokens))

        return apk.name, True, f"OK tokens={len(tokens)}"

    except Exception as e:
        logging.error("[worker] %s: unexpected error: %s", apk.name, e)
        return apk.name, False, f"exception: {e}"
    finally:
        # 4) 中間ファイルは必ず削除（RAMディスク節約）
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            # ここは最悪残ってもよいのでINFOで落とす
            logger.info("cleanup skip: %s", e)


# =========================
# バッチ（並列）実行
# =========================
def run_batch(opt: Options) -> Tuple[int, int]:
    """
    ディレクトリ内 *.apk を並列処理。
    返り値: (OK数, NG数)
    """
    _setup_root_logging()
    apks = sorted([p for p in opt.apk_dir.glob("*.apk") if p.is_file()])
    if not apks:
        print(f"[warn] no apk files under: {opt.apk_dir}")
        return (0, 0)

    _ensure_dir(opt.log_dir)
    _ensure_dir(opt.tokens_root)
    diag = opt.diagnostics_csv
    new_file = not diag.exists()
    ok, ng = 0, 0

    # 進捗バー
    bar = tqdm(total=len(apks), unit="apk", desc="apk→tokens", dynamic_ncols=True) if tqdm else None

    # 並列（Jupyter対応のため、トップレベル定義の関数を渡している）
    with ProcessPoolExecutor(max_workers=opt.workers) as ex:
        futs = [ex.submit(process_one_apk, apk, opt) for apk in apks]
        for fut in as_completed(futs):
            name, success, msg = fut.result()
            if success:
                ok += 1
            else:
                ng += 1
                # ここだけはユーザに見せたいので stderr へ
                print(f"[ERROR] {name}: {msg}", file=sys.stderr)
            if bar:
                bar.update(1)
                bar.set_postfix_str(f"{ok} OK / {ng} NG")

    if bar:
        bar.close()

    # ダイアグ
    with open(diag, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["apk_name", "status", "note"])
        # 2周目: ログを読み直すのも手だが、ここでは簡潔に集計のみ追記しない（必要なら各APKごとに都度追記に変更可）

    print(f"Completed: {ok} OK, {ng} FAIL (tokens at: {opt.tokens_root}, logs at: {opt.log_dir})")
    return ok, ng


# =========================
# CLI
# =========================
def _parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser()
    pa.add_argument("--apk-dir", type=Path, required=True, help="入力APKディレクトリ（*.apk）")
    pa.add_argument("--tokens-root", type=Path, default=Path("./corp_tokens"), help="トークンtxtの出力先")
    pa.add_argument("--log-dir", type=Path, default=Path("./logs_apktool"), help="詳細ログ出力先")
    pa.add_argument("--diagnostics-csv", type=Path, default=Path("./diagnostics.csv"))
    pa.add_argument("--tmp-base", type=Path, help="RAMディスク等の作業ルート（例: /dev/shm/apkwork）")
    pa.add_argument("--workers", type=int, default=1, help="並列数（CPU/IOに合わせて調整）")

    pa.add_argument("--include-prefix", nargs="*", default=[], help="採用クラス接頭辞（例: Landroid/ Ljava/ Lkotlin/）")
    pa.add_argument("--exclude-prefix", nargs="*", default=["Landroidx/test/"], help="除外クラス接頭辞")
    pa.add_argument("--include-signature", action="store_true", help="メソッドシグネチャも含める")

    pa.add_argument("--apktool-cmd", default="apktool")
    pa.add_argument("--no-res", action="store_true", help="apktool: リソースをデコードしない（-r）")
    pa.add_argument("--no-force", action="store_true", help="apktool: -f を使わない（上書き無効）")

    pa.add_argument("--no-assets-retry", action="store_true",
                    help="失敗時に assets/*.dex を無視する再試行を行わない（デフォルトは再試行する）")
    return pa.parse_args()


def main() -> None:
    args = _parse_args()
    opt = Options(
        apktool_cmd=args.apktool_cmd,
        apk_dir=args.apk_dir,
        tokens_root=args.tokens_root,
        log_dir=args.log_dir,
        diagnostics_csv=args.diagnostics_csv,
        tmp_base=args.tmp_base,
        workers=args.workers,
        include_prefixes=tuple(args.include_prefix) if args.include_prefix else (),
        exclude_prefixes=tuple(args.exclude_prefix) if args.exclude_prefix else (),
        include_signature=args.include_signature,
        apktool_no_res=args.no_res,
        apktool_force=not args.no_force,
        retry_ignore_assets=not args.no_assets_retry,
    )
    run_batch(opt)


if __name__ == "__main__":
    main()
