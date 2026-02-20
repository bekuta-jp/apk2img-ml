#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APK → apktool(RAM) → [前処理(RAM)] → [sdhash(RAM)]
保存オプションを3系統に分離:
  --apktool-save {all,smali_manifest,none}
  --preproc-save    （あり/なし）
  --sdbf-save       （あり/なし）

効率化:
- sdbf-save=False → 前処理までで次へ（preproc-saveがTrueなら保存）
- preproc-save=False かつ sdbf-save=False → apktool終了で次へ（apktool-saveがnoneなら何も保存せず次へ）
- すべて無し（apktool=none & preproc-save=False & sdbf-save=False）→ エラー

tqdm:
- CLIでもimportでもtqdm進捗バーを表示（import時は use_tqdm=False で無効化可）
- バー右側に done/skip/fail を表示

その他:
- --no-res（apktoolのリソース展開スキップ）
- 実行開始フラグ RUNNING.* と per-APK の .inprogress
- 失敗時はRAM中間を削除（keep_temp=False時）、ログに詳細
- sdhashはデフォルト設定。Manifestをハッシュ入力の先頭に固定
"""

from __future__ import annotations
import argparse, concurrent.futures as futures, os, sys, shutil, subprocess
import threading, time, atexit, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

# ========= tqdm（任意） =========
try:
    from tqdm import tqdm
except Exception:  # tqdm未インストールでも動くように
    tqdm = None

# ========= モデル =========
@dataclass
class ItemResult:
    apk_path: Path
    apk_name: str
    ok: bool
    skipped: bool
    message: str
    returncode: Optional[int] = None
    exception: Optional[str] = None
    log_file: Optional[Path] = None
    marker_left: bool = False

@dataclass
class RunSummary:
    total: int
    done: int
    skipped: int
    failed: int
    results: List[ItemResult] = field(default_factory=list)

@dataclass
class Options:
    no_res: bool = True
    apktool_save: str = "none"   # all | smali_manifest | none
    preproc_save: bool = False
    sdbf_save: bool = False
    api_corpus_save: bool = False  # [ADDED] APIコール列コーパスを保存するか
    workers: int = max(1, os.cpu_count() or 1)
    keep_temp: bool = False
    overwrite: bool = False

# ========= ユーティリティ =========
_PRINT_LOCK = threading.Lock()
def _eprint(*a, **k):
    with _PRINT_LOCK:
        print(*a, file=sys.stderr, **k)

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def is_apktool_available() -> bool:
    return shutil.which("apktool") is not None

def ensure_apktool():
    if not is_apktool_available():
        raise RuntimeError("apktool が見つかりません。インストールしてから実行してください。")

def collect_apks(input_dir: Path, recursive: bool=False) -> List[Path]:
    input_dir = input_dir.resolve()
    if recursive:
        return sorted([p for p in input_dir.rglob("*.apk") if p.is_file()])
    return sorted([p for p in input_dir.glob("*.apk") if p.is_file()])

def _build_apktool_cmd(apk: Path, outdir: Path, no_res: bool) -> List[str]:
    cmd = ["apktool", "d", str(apk), "-o", str(outdir)]
    if no_res:
        cmd.insert(2, "--no-res")
    return cmd

def _copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def _copy_smali_manifest(src_root: Path, dst_root: Path):
    ensure_dirs(dst_root)
    smali_src = src_root / "smali"
    if smali_src.exists():
        dst_smali = dst_root / "smali"
        if dst_smali.exists():
            shutil.rmtree(dst_smali)
        shutil.copytree(smali_src, dst_smali)
    mani_src = src_root / "AndroidManifest.xml"
    if mani_src.exists():
        shutil.copy2(mani_src, dst_root / "AndroidManifest.xml")

def _write_running_flag(log_root: Path) -> Path:
    ensure_dirs(log_root)
    flag = log_root / f"RUNNING.{int(time.time())}"
    flag.write_text("RUNNING\n", encoding="utf-8")
    return flag

# ========= 前処理（簡潔版） =========
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")
def clean_text_basic(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def gather_files_for_preproc(apktool_out: Path) -> Tuple[List[Path], Optional[Path]]:
    # [MODIFIED] smali ディレクトリだけでなく smali*（smali_classes2 等）をすべて対象にする
    smali_files: List[Path] = []  # [MODIFIED]
    for smali_dir in apktool_out.glob("smali*"):  # [ADDED]
        if smali_dir.is_dir():                     # [ADDED]
            smali_files.extend(smali_dir.rglob("*.smali"))  # [ADDED]
    smali_files = sorted(set(smali_files))  # [ADDED] 重複除去＆ソート

    mani = apktool_out / "AndroidManifest.xml"
    return smali_files, (mani if mani.exists() else None)

def run_preprocessing_to_temp(apktool_out: Path) -> Path:
    preproc_dir = apktool_out / "_preproc"
    ensure_dirs(preproc_dir)
    smali_list, mani_path = gather_files_for_preproc(apktool_out)

    smali_out = preproc_dir / "smali.txt"
    with smali_out.open("w", encoding="utf-8") as fout:
        for p in smali_list:
            try:
                raw = p.read_text("utf-8", errors="ignore")
                fout.write(clean_text_basic(raw) + "\n")
            except Exception:
                continue

    mani_out = preproc_dir / "manifest.txt"
    if mani_path and mani_path.exists():
        try:
            raw = mani_path.read_text("utf-8", errors="ignore")
            mani_out.write_text(clean_text_basic(raw) + "\n", encoding="utf-8")
        except Exception:
            mani_out.write_text("", encoding="utf-8")
    else:
        mani_out.write_text("", encoding="utf-8")
    return preproc_dir

# ========= APIコーパス抽出（追加） =========
# [ADDED] API呼び出し命令(invoke-*)からメソッドシグネチャを抽出し、
# [ADDED] smaliファイルごとのAPI列を1行のトークン列として保存する。
_INVOKE_RE = re.compile(r"^\s*invoke-\S+\s+{[^}]*},\s+([^\s]+)")  # [ADDED]

def run_api_corpus_to_temp(apktool_out: Path) -> Path:
    """
    [ADDED] smali内のAPI呼び出し列を抽出し、_api_corpus/api_sequences.txt に保存。
    1行 = 1 smaliファイル、トークンはメソッドシグネチャを空白区切り。
    """
    api_dir = apktool_out / "_api_corpus"  # [ADDED]
    ensure_dirs(api_dir)  # [ADDED]
    smali_files, _ = gather_files_for_preproc(apktool_out)  # [ADDED]

    out_path = api_dir / "api_sequences.txt"  # [ADDED]
    with out_path.open("w", encoding="utf-8") as fout:  # [ADDED]
        for smali_path in smali_files:  # [ADDED]
            try:
                text = smali_path.read_text("utf-8", errors="ignore")
            except Exception:
                continue
            tokens: List[str] = []
            for line in text.splitlines():
                m = _INVOKE_RE.match(line)
                if m:
                    tokens.append(m.group(1))
            if tokens:
                fout.write(" ".join(tokens) + "\n")
    return api_dir  # [ADDED]

# ========= sdhash（Manifest先頭固定 / デフォルト設定） =========
def prepare_sdhash_inputs(apktool_out: Path) -> Tuple[Path, Path]:
    """
    _sdbf_in/ に manifest.clean.txt, smali/**/*.clean.txt を順序通りに置き、
    files.lst にリスト化（先頭は Manifest）。
    """
    in_root = apktool_out / "_sdbf_in"
    out_root = apktool_out / "_sdbf"
    if in_root.exists():
        shutil.rmtree(in_root)
    ensure_dirs(in_root, out_root)

    smali_list, mani_path = gather_files_for_preproc(apktool_out)

    txt_paths: List[Path] = []
    if mani_path and mani_path.exists():
        mani_txt = in_root / "manifest.clean.txt"
        try:
            raw = mani_path.read_text("utf-8", errors="ignore")
            mani_txt.write_text(clean_text_basic(raw) + "\n", encoding="utf-8")
            txt_paths.append(mani_txt)
        except Exception:
            mani_txt.write_text("\n", encoding="utf-8")
            txt_paths.append(mani_txt)

    for p in smali_list:
        rel = p.relative_to(apktool_out)  # smali/...
        out_txt = in_root / (str(rel) + ".clean.txt")
        ensure_dirs(out_txt.parent)
        try:
            raw = p.read_text("utf-8", errors="ignore")
            out_txt.write_text(clean_text_basic(raw) + "\n", encoding="utf-8")
            txt_paths.append(out_txt)
        except Exception:
            continue

    list_file = in_root / "files.lst"
    list_file.write_text("\n".join(map(str, txt_paths)) + "\n", encoding="utf-8")
    return list_file, out_root

def run_sdhash_to_temp(apktool_out: Path, log_file: Path) -> Optional[Path]:
    try:
        list_file, out_root = prepare_sdhash_inputs(apktool_out)
        apk_name = apktool_out.name
        sdbf_path = out_root / f"{apk_name}.sdbf"

        proc = subprocess.run(["sdhash", "-f", str(list_file)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(log_file, "a", encoding="utf-8") as lf:
            if proc.stdout:
                lf.write("\n# sdhash stdout (truncated 2KB)\n")
                lf.write(proc.stdout[:2048])
            if proc.stderr:
                lf.write("\n# sdhash stderr\n")
                lf.write(proc.stderr)

        if proc.returncode != 0:
            return None

        sdbf_path.write_text(proc.stdout, encoding="utf-8")
        return sdbf_path
    except Exception as ex:
        try:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n[sdhash EXCEPTION] {ex}\n")
        except Exception:
            pass
        return None

# ========= パイプライン本体 =========
def process_apks(
    apks: Iterable[Path],
    *,
    temp_root: Path,
    output_root: Path,
    log_root: Path,
    opt: Options,
    use_tqdm: bool = True,                         # ← import時もtqdmを出す（無効化可）
    progress_callback: Optional[Callable[[RunSummary], None]] = None
) -> RunSummary:
    ensure_apktool()
    temp_root = temp_root.resolve()
    output_root = output_root.resolve()
    log_root = log_root.resolve()

    # オプション検証
    if (
        opt.apktool_save == "none"
        and not opt.preproc_save
        and not opt.sdbf_save
        and not opt.api_corpus_save  # [MODIFIED] APIコーパス保存も考慮
    ):
        raise ValueError("保存オプションが全て無しです。少なくとも1つは有効にしてください。")

    ensure_dirs(temp_root, output_root, log_root)
    logs_dir = log_root / "apktool_logs"
    mark_dir = log_root / "in_progress"
    ensure_dirs(logs_dir, mark_dir)

    running_flag = _write_running_flag(log_root)
    def _cleanup_running():
        try:
            if running_flag.exists():
                running_flag.unlink()
        except Exception:
            pass
    atexit.register(_cleanup_running)

    apks = list(apks)
    summary = RunSummary(total=len(apks), done=0, skipped=0, failed=0, results=[])
    if not apks:
        return summary

    if opt.workers <= 0:
        opt.workers = max(1, os.cpu_count() or 1)

    lock = threading.Lock()

    def _cleanup_temp(path: Path):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _one(apk: Path) -> ItemResult:
        apk = apk.resolve()
        name = apk.stem
        log_file = logs_dir / f"{name}.log"
        inprog = mark_dir / f"{name}.inprogress"
        final_dir = output_root / name

        # 既存出力 → スキップ
        if final_dir.exists() and not opt.overwrite:
            return ItemResult(apk, name, ok=True, skipped=True, message="既存出力のためスキップ")

        # マーカー
        try:
            inprog.write_text("STARTED\n", encoding="utf-8")
        except Exception as ex:
            return ItemResult(apk, name, ok=False, skipped=False, message=f"マーカー作成失敗: {ex}", exception=str(ex), marker_left=True)

        apktool_out = temp_root / name
        if apktool_out.exists():
            _cleanup_temp(apktool_out)

        # --- apktool（必須） ---
        cmd = _build_apktool_cmd(apk, apktool_out, no_res=opt.no_res)
        try:
            ensure_dirs(apktool_out.parent)
            with open(log_file, "wb") as lf:
                lf.write((" ".join(cmd) + "\n\n").encode("utf-8"))
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False).returncode

            # === 【追加】assets内の壊れDEXを検出して削除 ===
            assets_dir = apktool_out / "assets"
            if assets_dir.exists():
                dex_files = list(assets_dir.glob("*.dex"))
                if dex_files:
                    with open(apktool_out / "_packer_detected.txt", "w", encoding="utf-8") as f:
                        f.write("Detected packer-protected APK (assets/*.dex removed)\n")
                        for df in dex_files:
                            f.write(f"{df.name}\n")
                    for df in dex_files:
                        try:
                            df.unlink()
                        except Exception as e:
                            _eprint(f"[apktool] Warning: failed to remove {df}: {e}")
                    try:
                        if not any(assets_dir.iterdir()):
                            shutil.rmtree(assets_dir)
                    except Exception:
                        pass
            # === 【追加ここまで】===

            if rc != 0:
                smali_exists = any(apktool_out.glob("smali*"))  # [MODIFIED] smali* を確認
                manifest_exists = (apktool_out / "AndroidManifest.xml").exists()
                # [MODIFIED] どちらか一方でもあれば「使える部分あり」とみなして続行
                if smali_exists or manifest_exists:  # [MODIFIED]
                    _eprint(f"[apktool] Non-zero exit but usable files exist: {apk.name}")
                else:
                    if not opt.keep_temp:
                        _cleanup_temp(apktool_out)
                    return ItemResult(apk, name, ok=False, skipped=False,
                                      message=f"apktool 失敗 (rc={rc}) -> {log_file.name}",
                                      returncode=rc, log_file=log_file, marker_left=True)

        except Exception as ex:
            if not opt.keep_temp:
                _cleanup_temp(apktool_out)
            return ItemResult(apk, name, ok=False, skipped=False,
                              message=f"apktool 例外: {ex}", exception=str(ex),
                              log_file=log_file, marker_left=True)

        # --- 前処理が必要？ ---
        preproc_dir: Optional[Path] = None
        need_preproc_for_sdbf = opt.sdbf_save
        need_preproc_for_save = opt.preproc_save
        if need_preproc_for_sdbf or need_preproc_for_save:
            try:
                preproc_dir = run_preprocessing_to_temp(apktool_out)
            except Exception as ex:
                if not opt.keep_temp:
                    _cleanup_temp(apktool_out)
                return ItemResult(apk, name, ok=False, skipped=False,
                                  message=f"前処理失敗: {ex}", exception=str(ex),
                                  log_file=log_file, marker_left=True)

        # --- sdhashが必要？ ---
        sdbf_path: Optional[Path] = None
        if opt.sdbf_save:
            sdbf_path = run_sdhash_to_temp(apktool_out, log_file)
            if sdbf_path is None:
                if not opt.keep_temp:
                    _cleanup_temp(apktool_out)
                return ItemResult(apk, name, ok=False, skipped=False,
                                  message="sdhash 失敗（ログ参照）",
                                  log_file=log_file, marker_left=True)

        # --- APIコーパスが必要？ ---  # [ADDED]
        api_dir: Optional[Path] = None  # [ADDED]
        if opt.api_corpus_save:  # [ADDED]
            try:
                api_dir = run_api_corpus_to_temp(apktool_out)  # [ADDED]
            except Exception as ex:  # [ADDED]
                if not opt.keep_temp:  # [ADDED]
                    _cleanup_temp(apktool_out)  # [ADDED]
                return ItemResult(  # [ADDED]
                    apk, name, ok=False, skipped=False,
                    message=f"APIコーパス生成失敗: {ex}", exception=str(ex),
                    log_file=log_file, marker_left=True
                )

        # --- 保存 ---
        try:
            if final_dir.exists() and opt.overwrite:
                shutil.rmtree(final_dir)
            ensure_dirs(final_dir)

            # 1) apktool保存
            if opt.apktool_save == "all":
                _copy_tree(apktool_out, final_dir)
            elif opt.apktool_save == "smali_manifest":
                _copy_smali_manifest(apktool_out, final_dir)
            elif opt.apktool_save == "none":
                pass

            # 2) 前処理保存
            if opt.preproc_save and preproc_dir and preproc_dir.exists():
                _copy_tree(preproc_dir, final_dir / "_preproc")

            # 3) sdhash保存
            if opt.sdbf_save and sdbf_path and sdbf_path.exists():
                dst_sdbf = final_dir / "_sdbf"
                ensure_dirs(dst_sdbf)
                shutil.copy2(sdbf_path, dst_sdbf / sdbf_path.name)

            # 4) APIコーパス保存  # [ADDED]
            if opt.api_corpus_save and api_dir and api_dir.exists():  # [ADDED]
                _copy_tree(api_dir, final_dir / "_api_corpus")  # [ADDED]

        except Exception as ex:
            if not opt.keep_temp:
                _cleanup_temp(apktool_out)
            return ItemResult(apk, name, ok=False, skipped=False,
                              message=f"コピー失敗: {ex}", exception=str(ex),
                              log_file=log_file, marker_left=True)

        # 正常終了クリーンアップ
        try:
            inprog.unlink(missing_ok=True)
        except Exception:
            pass
        if not opt.keep_temp:
            _cleanup_temp(apktool_out)
        return ItemResult(apk, name, ok=True, skipped=False, message="OK", log_file=log_file)

    # ========= 並列 + tqdm進捗 =========
    try:
        with futures.ThreadPoolExecutor(max_workers=opt.workers) as ex:
            futs = [ex.submit(_one, a) for a in apks]

            use_bar = bool(use_tqdm and tqdm is not None)
            if use_bar:
                # 動的幅・色は環境によっては無視されることがあります
                with tqdm(total=len(apks), desc="APK処理", unit="apk", dynamic_ncols=True) as pbar:
                    for fut in futures.as_completed(futs):
                        res = fut.result()
                        with lock:
                            summary.results.append(res)
                            if res.skipped:
                                summary.skipped += 1
                            elif res.ok:
                                summary.done += 1
                            else:
                                summary.failed += 1
                            pbar.update(1)
                            pbar.set_postfix({
                                "done": summary.done,
                                "skip": summary.skipped,
                                "fail": summary.failed
                            })
                            if progress_callback:
                                progress_callback(summary)
            else:
                for fut in futures.as_completed(futs):
                    res = fut.result()
                    with lock:
                        summary.results.append(res)
                        if res.skipped:
                            summary.skipped += 1
                        elif res.ok:
                            summary.done += 1
                        else:
                            summary.failed += 1
                        if progress_callback:
                            progress_callback(summary)
    except KeyboardInterrupt:
        _eprint("\n[!] 中断されました（Ctrl+C）。RUNNING フラグと .inprogress が残ります。")
        raise

    return summary

# ========= CLI =========
def _cli():
    p = argparse.ArgumentParser(description="APK→apktool(RAM)→[preproc]→[sdhash]（tqdm進捗対応）")
    p.add_argument("-i", "--input-dir", type=Path, required=True)
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("-l", "--log-dir", type=Path, required=True)
    p.add_argument("-t", "--temp-dir", type=Path, required=True)
    p.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--no-res", action="store_true")

    # 分離した3オプション
    p.add_argument("--apktool-save", choices=["all", "smali_manifest", "none"], default="none",
                   help="apktoolの出力を保存（全部 / smali+manifest / 保存しない）")
    p.add_argument("--preproc-save", action="store_true", help="前処理(_preproc)を保存")
    p.add_argument("--sdbf-save", action="store_true", help="sdhash(.sdbf)を保存")
    p.add_argument("--api-corpus-save", action="store_true",
                   help="API呼び出し列のコーパス(_api_corpus)を保存")  # [ADDED]

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-temp", action="store_true")

    # tqdm制御（CLIでも基本ON）
    p.add_argument("--no-tqdm", action="store_true", help="tqdmの進捗バーを無効化")

    args = p.parse_args()

    # 事前バリデーション
    if (
        args.apktool_save == "none"
        and (not args.preproc_save)
        and (not args.sdbf_save)
        and (not args.api_corpus_save)  # [MODIFIED] APIコーパス保存も考慮
    ):
        _eprint("ERROR: すべての保存オプションが無効です。少なくとも1つは有効にしてください。")
        sys.exit(2)

    ensure_apktool()
    apks = collect_apks(args.input_dir, recursive=args.recursive)
    if not apks:
        print("対象のAPKが見つかりません。")
        return

    print(
        f"対象: {len(apks)} / apktool-save={args.apktool_save} "
        f"/ preproc-save={args.preproc_save} / sdbf-save={args.sdbf_save} "
        f"/ api-corpus-save={args.api_corpus_save} "  # [MODIFIED]
        f"/ --no-res={'ON' if args.no_res else 'OFF'} / workers={args.workers}"
    )
    print(f"temp: {args.temp_dir} / out: {args.output_dir} / logs: {args.log_dir}")
    if tqdm is None and not args.no_tqdm:
        print("(tqdm が見つかりませんでした。 pip install tqdm でインストールすると進捗バーが表示されます)")
    print("-" * 60)

    opt = Options(
        no_res=args.no_res,
        apktool_save=args.apktool_save,
        preproc_save=args.preproc_save,
        sdbf_save=args.sdbf_save,
        api_corpus_save=args.api_corpus_save,  # [ADDED]
        workers=args.workers,
        keep_temp=args.keep_temp,
        overwrite=args.overwrite,
    )

    summary = process_apks(
        apks=apks,
        temp_root=args.temp_dir,
        output_root=args.output_dir,
        log_root=args.log_dir,
        opt=opt,
        use_tqdm=not args.no_tqdm
    )

    print("-" * 60)
    print(f"完了: {summary.done} / スキップ: {summary.skipped} / 失敗: {summary.failed} / 合計: {summary.total}")
    if summary.failed:
        print("失敗の詳細はログを確認。in_progress/*.inprogress が残っていれば途中/失敗です。")

if __name__ == "__main__":
    _cli()
