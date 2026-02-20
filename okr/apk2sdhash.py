#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APK → apktool(RAM) → [前処理(RAM)] → [sdhash(RAM)] → [APIコーパス]
- apktool失敗時:
  - オプション指定があれば assets/*.dex を削除した一時APKで再実行
- 再実行後:
  - smali が 1つでもあれば → smali から API コーパス生成
  - smali が 1つも無く、Manifest だけある & rc == 0 → sdhash 専用として扱い、API 抽出はスキップ
  - rc != 0 かつ smali も Manifest も無い → 生の DEX から API 抽出を試みる
      - raw DEX からも抽出できなければエラー終了

保存オプションを分離:
  --apktool-save {all,smali_manifest,none}
  --preproc-save    （あり/なし or preproc-out-root指定）
  --sdbf-save       （あり/なし or sdbf-out-root指定）
  --api-corpus-save （あり/なし or api-corpus-out-root指定）

tqdm:
- CLIでもimportでもtqdm進捗バーを表示（import時は use_tqdp=False で無効化可）
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
import zipfile  # apktool失敗時の一時APK作成用（assets/*.dex除去）

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
    api_corpus_save: bool = False  # APIコール列コーパスを保存するか
    apktool_retry_strip_assets: bool = False  # apktool失敗時に assets/*.dex 除去して再実行

    # 各出力の保存先ルート（指定がなければ従来通り output_root 配下）
    apktool_out_root: Optional[Path] = None
    preproc_out_root: Optional[Path] = None
    sdbf_out_root: Optional[Path] = None
    api_corpus_out_root: Optional[Path] = None

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

def _make_stripped_apk(src_apk: Path, temp_root: Path) -> Optional[Path]:
    """
    apktoolエラー時のフォールバック用:
    src_apk を一時展開し、assets/*.dex を削除してから再zipしたAPKパスを返す。
    assets/*.dex が無い、または処理失敗時は None を返す。
    """
    work_dir = temp_root / f"{src_apk.stem}_ziptmp"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    ensure_dirs(work_dir)

    has_assets_dex = False
    try:
        with zipfile.ZipFile(src_apk, "r") as zf:
            for info in zf.infolist():
                if info.filename.startswith("assets/") and info.filename.endswith(".dex"):
                    has_assets_dex = True
                    break
            if not has_assets_dex:
                shutil.rmtree(work_dir, ignore_errors=True)
                return None
            zf.extractall(work_dir)
    except Exception as ex:
        _eprint(f"[strip_apk] zip処理に失敗しました: {src_apk} ({ex})")
        shutil.rmtree(work_dir, ignore_errors=True)
        return None

    assets_dir = work_dir / "assets"
    if assets_dir.exists():
        for p in assets_dir.glob("*.dex"):
            try:
                p.unlink()
            except Exception as e:
                _eprint(f"[strip_apk] failed to remove {p}: {e}")

    stripped_apk = temp_root / f"{src_apk.stem}_stripped.apk"
    if stripped_apk.exists():
        stripped_apk.unlink()

    try:
        with zipfile.ZipFile(stripped_apk, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in work_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(work_dir))
    except Exception as ex:
        _eprint(f"[strip_apk] 再圧縮に失敗しました: {src_apk} ({ex})")
        shutil.rmtree(work_dir, ignore_errors=True)
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return stripped_apk

# ========= 前処理（簡潔版） =========
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")
def clean_text_basic(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def gather_files_for_preproc(apktool_out: Path) -> Tuple[List[Path], Optional[Path]]:
    """
    smali ディレクトリだけでなく smali*（smali_classes2 等）をすべて対象にする
    """
    smali_files: List[Path] = []
    for smali_dir in apktool_out.glob("smali*"):
        if smali_dir.is_dir():
            smali_files.extend(smali_dir.rglob("*.smali"))
    smali_files = sorted(set(smali_files))

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

# ========= APIコーパス抽出（smaliベース） =========
_INVOKE_RE = re.compile(r"^\s*invoke-\S+\s+{[^}]*},\s+([^\s]+)")

def run_api_corpus_to_temp(apktool_out: Path) -> Path:
    """
    smali内のAPI呼び出し列を抽出し、_api_corpus/api_sequences.txt に保存。
    1行 = 1 smaliファイル、トークンはメソッドシグネチャを空白区切り。
    """
    api_dir = apktool_out / "_api_corpus"
    ensure_dirs(api_dir)
    smali_files, _ = gather_files_for_preproc(apktool_out)

    out_path = api_dir / "api_sequences.txt"
    with out_path.open("w", encoding="utf-8") as fout:
        for smali_path in smali_files:
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
    return api_dir

# ========= APIコーパス抽出（raw DEX フォールバック） =========
# ざっくり「Lxxx/yyy;->methodName(" のようなシグネチャをバイナリから拾う
_RAW_API_RE = re.compile(r"L[\w/$-]{1,200};->[\w$<>]{1,100}\(")

def extract_api_from_raw_dex(apk: Path,
                             max_dex_files: int = 5,
                             max_bytes_per_dex: int = 8 * 1024 * 1024) -> List[str]:
    """
    apk から DEX ファイルを直接読み、バイナリ中からメソッドシグネチャっぽい文字列を正規表現で抜く。
    - assets/*.dex は無視（packer用を避けるため）
    - 上限 max_dex_files, max_bytes_per_dex でメモリ使いすぎを防止
    """
    tokens: List[str] = []
    try:
        with zipfile.ZipFile(apk, "r") as zf:
            dex_names = [
                name for name in zf.namelist()
                if name.endswith(".dex") and not name.startswith("assets/")
            ]
            dex_names = sorted(dex_names)[:max_dex_files]
            for name in dex_names:
                try:
                    data = zf.read(name)[:max_bytes_per_dex]
                except Exception:
                    continue
                # バイナリだが、latin-1 でそのまま 1byte→1文字 として扱う
                text = data.decode("latin-1", errors="ignore")
                for m in _RAW_API_RE.finditer(text):
                    tokens.append(m.group(0))
    except Exception as ex:
        _eprint(f"[rawdex] {apk.name}: raw DEX 読み出しに失敗 ({ex})")
        return []

    return tokens

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
    use_tqdm: bool = True,
    progress_callback: Optional[Callable[[RunSummary], None]] = None
) -> RunSummary:
    ensure_apktool()
    temp_root = temp_root.resolve()
    output_root = output_root.resolve()
    log_root = log_root.resolve()

    # どの項目を「保存したいか」を、出力先指定も加味して判定
    want_preproc = opt.preproc_save or (opt.preproc_out_root is not None)
    want_sdbf    = opt.sdbf_save    or (opt.sdbf_out_root is not None)
    want_api     = opt.api_corpus_save or (opt.api_corpus_out_root is not None)

    # オプション検証
    if (
        opt.apktool_save == "none"
        and not want_preproc
        and not want_sdbf
        and not want_api
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
        final_dir = output_root / name  # 従来の出力ルート（互換用）

        # 既存出力 → スキップ（従来挙動維持）
        if final_dir.exists() and not opt.overwrite:
            return ItemResult(apk, name, ok=True, skipped=True, message="既存出力のためスキップ")

        # マーカー
        try:
            inprog.write_text("STARTED\n", encoding="utf-8")
        except Exception as ex:
            return ItemResult(apk, name, ok=False, skipped=False,
                              message=f"マーカー作成失敗: {ex}", exception=str(ex), marker_left=True)

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

            # apktool 成功/失敗に関わらず、一旦 assets/ の packer 用 dex を削除
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

            # 失敗時、必要なら「assets/*.dex削除済みAPK」で再実行
            if rc != 0 and opt.apktool_retry_strip_assets:
                stripped_apk = _make_stripped_apk(apk, temp_root)
                if stripped_apk is not None:
                    _cleanup_temp(apktool_out)
                    ensure_dirs(apktool_out.parent)
                    with open(log_file, "ab") as lf2:
                        lf2.write(b"\n# Retrying apktool with stripped APK (assets/*.dex removed)\n")
                        cmd2 = _build_apktool_cmd(stripped_apk, apktool_out, no_res=opt.no_res)
                        lf2.write((" ".join(cmd2) + "\n\n").encode("utf-8"))
                        rc = subprocess.run(cmd2, stdout=lf2, stderr=subprocess.STDOUT,
                                            check=False).returncode

            # 再実行後の smali/Manifest 有無をここで確定させておく
            smali_exists = any(apktool_out.glob("smali*"))
            manifest_exists = (apktool_out / "AndroidManifest.xml").exists()

            # === ここから apktool 成否フラグの導入 / 完全失敗時の扱い調整 ===
            apktool_ok = (rc == 0)  # [ADDED] apktool が正常終了したか
            apktool_fail = (rc != 0 and (not smali_exists) and (not manifest_exists))  # [ADDED] 出力が何も無い完全失敗
            apktool_partial = (rc != 0 and (smali_exists or manifest_exists))  # [ADDED] 一部の成果物だけある部分成功

            if apktool_partial:  # [ADDED]
                _eprint(f"[apktool] Non-zero exit but usable files exist: {apk.name}")

            # [MODIFIED] 以前は rc!=0 かつ smali/Manifest 無しで常にここで return していた。
            #            その結果、API コーパスが欲しい場合でも raw DEX フォールバックまで到達していなかった。
            #            今回は「APIが不要」の場合だけここで失敗確定し、APIが欲しい場合は後段で raw DEX を試す。
            if apktool_fail and not want_api:  # [MODIFIED]
                if not opt.keep_temp:
                    _cleanup_temp(apktool_out)
                return ItemResult(
                    apk, name, ok=False, skipped=False,
                    message=f"apktool 失敗 (rc={rc}) -> {log_file.name}",
                    returncode=rc, log_file=log_file, marker_left=True
                )
            # === 変更ここまで ===

        except Exception as ex:
            if not opt.keep_temp:
                _cleanup_temp(apktool_out)
            return ItemResult(apk, name, ok=False, skipped=False,
                              message=f"apktool 例外: {ex}", exception=str(ex),
                              log_file=log_file, marker_left=True)

        # --- 前処理が必要？ ---
        preproc_dir: Optional[Path] = None
        need_preproc_for_sdbf = want_sdbf
        need_preproc_for_save = want_preproc
        if need_preproc_for_sdbf or need_preproc_for_save:
            # [ADDED] apktool が完全に失敗している場合は preproc/sdhash は実行できない
            if 'apktool_fail' in locals() and apktool_fail:  # [ADDED]
                if not opt.keep_temp:  # [ADDED]
                    _cleanup_temp(apktool_out)  # [ADDED]
                return ItemResult(  # [ADDED]
                    apk, name, ok=False, skipped=False,
                    message="apktool 完全失敗のため preproc/sdhash を実行できません",
                    log_file=log_file, marker_left=True
                )
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
        if want_sdbf:
            # [ADDED] apktool 完全失敗時のガード（上と重複だが念のため）
            if 'apktool_fail' in locals() and apktool_fail:  # [ADDED]
                if not opt.keep_temp:  # [ADDED]
                    _cleanup_temp(apktool_out)  # [ADDED]
                return ItemResult(  # [ADDED]
                    apk, name, ok=False, skipped=False,
                    message="apktool 完全失敗のため sdhash を実行できません",
                    log_file=log_file, marker_left=True
                )

            sdbf_path = run_sdhash_to_temp(apktool_out, log_file)
            if sdbf_path is None:
                if not opt.keep_temp:
                    _cleanup_temp(apktool_out)
                return ItemResult(apk, name, ok=False, skipped=False,
                                  message="sdhash 失敗（ログ参照）",
                                  log_file=log_file, marker_left=True)

        # --- APIコーパスが必要？ ---
        api_dir: Optional[Path] = None
        if want_api:
            try:
                if smali_exists:
                    # smali から API 抽出（apktool の rc は問わない）
                    api_dir = run_api_corpus_to_temp(apktool_out)
                else:
                    if apktool_ok and manifest_exists:  # [MODIFIED] rc==0 のときは「manifest only → APIスキップ」
                        # 再実行は成功しているが smali が無く Manifest だけ:
                        # → 指定通り「sdhash専用」とみなし、API 抽出はスキップ
                        with open(log_file, "a", encoding="utf-8") as lf:
                            lf.write(
                                "\n[api_corpus] skipped: rc==0 & no smali "
                                "(manifest only / sdhash only)\n"
                            )
                        api_dir = None
                    else:
                        # rc != 0 で smali が無いケース:
                        # - Manifest があってもなくても、smali が無い以上は raw DEX からの抽出を試みる
                        api_tokens = extract_api_from_raw_dex(apk)
                        if not api_tokens:
                            if not opt.keep_temp:
                                _cleanup_temp(apktool_out)
                            return ItemResult(
                                apk, name, ok=False, skipped=False,
                                message="API抽出失敗: apktool失敗 & raw DEX からも抽出不可",
                                log_file=log_file, marker_left=True
                            )

                        api_dir = apktool_out / "_api_corpus_rawdex"
                        ensure_dirs(api_dir)
                        out_path = api_dir / "api_sequences.txt"
                        with out_path.open("w", encoding="utf-8") as f:
                            f.write(" ".join(api_tokens) + "\n")

                        with open(log_file, "a", encoding="utf-8") as lf:
                            lf.write("\n[api_corpus] generated from raw DEX fallback (rc!=0 or no smali)\n")  # [MODIFIED] ログ文言少し拡張
            except Exception as ex:
                if not opt.keep_temp:
                    _cleanup_temp(apktool_out)
                return ItemResult(
                    apk, name, ok=False, skipped=False,
                    message=f"APIコーパス生成失敗: {ex}",
                    exception=str(ex), log_file=log_file, marker_left=True
                )

        # --- 保存 ---
        try:
            # 従来の final_dir を使うパスがある場合のみ作成
            use_final_dir = (
                (opt.apktool_out_root is None and opt.apktool_save != "none")
                or (opt.preproc_out_root is None and want_preproc)
                or (opt.sdbf_out_root is None and want_sdbf)
                or (opt.api_corpus_out_root is None and want_api)
            )

            if final_dir.exists() and opt.overwrite and use_final_dir:
                shutil.rmtree(final_dir)
            if use_final_dir:
                ensure_dirs(final_dir)

            # 1) apktool保存
            if opt.apktool_save == "all":
                if opt.apktool_out_root is not None:
                    dst_dir = opt.apktool_out_root / name
                    ensure_dirs(dst_dir)
                    _copy_tree(apktool_out, dst_dir)
                else:
                    _copy_tree(apktool_out, final_dir)
            elif opt.apktool_save == "smali_manifest":
                if opt.apktool_out_root is not None:
                    dst_dir = opt.apktool_out_root / name
                    ensure_dirs(dst_dir)
                    _copy_smali_manifest(apktool_out, dst_dir)
                else:
                    _copy_smali_manifest(apktool_out, final_dir)

            # 2) 前処理保存
            if want_preproc and preproc_dir and preproc_dir.exists():
                if opt.preproc_out_root is not None:
                    dst_pre = opt.preproc_out_root / name
                    _copy_tree(preproc_dir, dst_pre)
                else:
                    _copy_tree(preproc_dir, final_dir / "_preproc")

            # 3) sdhash保存
            if want_sdbf and sdbf_path and sdbf_path.exists():
                if opt.sdbf_out_root is not None:
                    ensure_dirs(opt.sdbf_out_root)
                    dst_file = opt.sdbf_out_root / f"{name}.sdbf"
                    shutil.copy2(sdbf_path, dst_file)
                else:
                    dst_sdbf = final_dir / "_sdbf"
                    ensure_dirs(dst_sdbf)
                    shutil.copy2(sdbf_path, dst_sdbf / sdbf_path.name)

            # 4) APIコーパス保存
            if want_api and api_dir and api_dir.exists():
                tmp_api_file = api_dir / "api_sequences.txt"
                if opt.api_corpus_out_root is not None:
                    ensure_dirs(opt.api_corpus_out_root)
                    dst_api = opt.api_corpus_out_root / f"{name}_api_sequences.txt"
                    if tmp_api_file.exists():
                        shutil.copy2(tmp_api_file, dst_api)
                    else:
                        # 念のためディレクトリごとコピー
                        _copy_tree(api_dir, opt.api_corpus_out_root / name)
                else:
                    _copy_tree(api_dir, final_dir / "_api_corpus")

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
        return ItemResult(apk, name, ok=True, skipped=False,
                          message="OK", log_file=log_file)

    # ========= 並列 + tqdm進捗 =========
    try:
        with futures.ThreadPoolExecutor(max_workers=opt.workers) as ex:
            futs = [ex.submit(_one, a) for a in apks]

            use_bar = bool(use_tqdm and tqdm is not None)
            if use_bar:
                with tqdm(total=len(apks), desc="APK処理", unit="apk",
                          dynamic_ncols=True) as pbar:
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
    p = argparse.ArgumentParser(description="APK→apktool(RAM)→[preproc]→[sdhash]→[APIコーパス]（tqdm進捗対応）")
    p.add_argument("-i", "--input-dir", type=Path, required=True)
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("-l", "--log-dir", type=Path, required=True)
    p.add_argument("-t", "--temp-dir", type=Path, required=True)
    p.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--no-res", action="store_true")

    # 分離した3オプション + API
    p.add_argument("--apktool-save", choices=["all", "smali_manifest", "none"], default="none",
                   help="apktoolの出力を保存（全部 / smali+manifest / 保存しない）")
    p.add_argument("--preproc-save", action="store_true", help="前処理(_preproc)を保存")
    p.add_argument("--sdbf-save", action="store_true", help="sdhash(.sdbf)を保存")
    p.add_argument("--api-corpus-save", action="store_true",
                   help="API呼び出し列のコーパス(_api_corpus)を保存")
    p.add_argument("--apktool-retry-strip-assets", action="store_true",
                   help="apktool失敗時に assets/*.dex を削除した一時APKで再実行する")

    # 各出力の保存先ルート
    p.add_argument("--apktool-out-root", type=Path, default=None,
                   help="apktool出力を保存するルートディレクトリ（apk名配下に展開）")
    p.add_argument("--preproc-out-root", type=Path, default=None,
                   help="前処理結果を保存するルートディレクトリ（apk名配下に展開）")
    p.add_argument("--sdbf-out-root", type=Path, default=None,
                   help="sdhash(.sdbf)を保存するルートディレクトリ（apk名.sdbf）")
    p.add_argument("--api-corpus-out-root", type=Path, default=None,
                   help="APIコーパスを保存するルートディレクトリ（apk名_api_sequences.txt）")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-temp", action="store_true")

    # tqdm制御（CLIでも基本ON）
    p.add_argument("--no-tqdm", action="store_true", help="tqdmの進捗バーを無効化")

    args = p.parse_args()

    want_preproc = args.preproc_save or (args.preproc_out_root is not None)
    want_sdbf    = args.sdbf_save    or (args.sdbf_out_root is not None)
    want_api     = args.api_corpus_save or (args.api_corpus_out_root is not None)

    if (
        args.apktool_save == "none"
        and (not want_preproc)
        and (not want_sdbf)
        and (not want_api)
    ):
        _eprint("ERROR: すべての保存オプションが無効です。少なくとも1つは有効にしてください。")
        sys.exit(2)

    ensure_apktool()
    apks = collect_apks(args.input_dir, recursive=args.recursive)  # [MODIFIED] input-dir → input_dir に修正
    if not apks:
        print("対象のAPKが見つかりません。")
        return

    print(
        f"対象: {len(apks)} / apktool-save={args.apktool_save} "
        f"/ preproc-save={args.preproc_save} / sdbf-save={args.sdbf_save} "
        f"/ api-corpus-save={args.api_corpus_save} "
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
        api_corpus_save=args.api_corpus_save,
        apktool_retry_strip_assets=args.apktool_retry_strip_assets,
        apktool_out_root=args.apktool_out_root,
        preproc_out_root=args.preproc_out_root,
        sdbf_out_root=args.sdbf_out_root,
        api_corpus_out_root=args.api_corpus_out_root,
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
