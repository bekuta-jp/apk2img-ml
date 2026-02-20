#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APK群から baksmali で smali を起こし、invoke行から API トークンを抽出して *.txt 出力。
- トークン形式（既定）: Lpkg/Class;->method
- オプション: --include-sig を付けると (args)return を含む完全シグネチャに
- 失敗しても処理継続、失敗リストは failures.csv に保存
"""

from __future__ import annotations
import argparse, csv, re, subprocess, tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from zipfile import ZipFile

INVOKE_RE = re.compile(
    r"^\s*invoke-(?:virtual|static|direct|interface|super)(?:/range)?\s+.*?,\s*(L[^;]+;)->([^\(]+)\((.*?)\)(\S)")

def disassemble_all_dex(apk_path: Path, baksmali_jar: Path, smali_out: Path) -> None:
    smali_out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dexdir = Path(tmp) / "dex"; dexdir.mkdir()
        with ZipFile(apk_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".dex"):
                    zf.extract(name, dexdir)
        for dex in sorted(dexdir.rglob("*.dex")):
            cmd = ["java","-jar",str(baksmali_jar),"disassemble","--ignore-errors","-o",str(smali_out),str(dex)]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_tokens_from_smali(smali_root: Path,
                              include: Optional[Tuple[str, ...]],
                              exclude: Optional[Tuple[str, ...]],
                              include_sig: bool=False) -> List[str]:
    toks: List[str] = []
    for smali in smali_root.rglob("*.smali"):
        try:
            with open(smali, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = INVOKE_RE.match(line)
                    if not m: 
                        continue
                    clazz, method, args, ret = m.groups()
                    if include and not any(clazz.startswith(p) for p in include):
                        continue
                    if exclude and any(clazz.startswith(p) for p in exclude):
                        continue
                    if include_sig:
                        toks.append(f"{clazz}->{method}({args}){ret}")
                    else:
                        toks.append(f"{clazz}->{method}")
        except Exception:
            continue
    return toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apk-dir", required=True, help="APK が入ったディレクトリ")
    ap.add_argument("--out-dir", required=True, help="出力ディレクトリ（*.txt）")
    ap.add_argument("--baksmali-jar", default="baksmali.jar", help="baksmali.jar のパス")
    ap.add_argument("--include", nargs="*", default=["Landroid/","Ljava/","Lkotlin/"],
                    help="含めるクラス接頭辞（空にしたい場合は --include-none）")
    ap.add_argument("--include-none", action="store_true", help="include フィルタなし")
    ap.add_argument("--exclude", nargs="*", default=["Landroidx/test/"], help="除外する接頭辞")
    ap.add_argument("--include-sig", action="store_true", help="(args)return を含む完全シグネチャで出力")
    ap.add_argument("--workers", type=int, default=0, help="将来用（今は単純直列）。0=自動")
    args = ap.parse_args()

    apk_dir    = Path(args.apk_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    baksmali   = Path(args.baksmali_jar)
    include    = None if args.include_none else tuple(args.include) if args.include else None
    exclude    = tuple(args.exclude) if args.exclude else None

    failures = []
    apks = sorted(apk_dir.glob("*.apk"))
    for i, apk in enumerate(apks, 1):
        out_txt = out_dir / f"{apk.stem}.txt"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                smali_root = Path(tmp) / "smali"
                disassemble_all_dex(apk, baksmali, smali_root)
                toks = extract_tokens_from_smali(smali_root, include, exclude, include_sig=args.include_sig)
            out_txt.write_text(" ".join(toks) + "\n", encoding="utf-8")
            print(f"[{i}/{len(apks)}] {apk.name}: {len(toks)} tokens")
        except Exception as e:
            failures.append((apk.name, repr(e)))
            print(f"[{i}/{len(apks)}] {apk.name}: FAILED ({e})")

    if failures:
        fail_csv = out_dir / "failures.csv"
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["apk","error"])
            w.writerows(failures)
        print(f"Completed with {len(failures)} failures (see {fail_csv}).")
    else:
        print("Completed with 0 failures.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APK群から baksmali で smali を起こし、invoke行から API トークンを抽出して *.txt 出力。
- トークン形式（既定）: Lpkg/Class;->method
- オプション: --include-sig を付けると (args)return を含む完全シグネチャに
- 失敗しても処理継続、失敗リストは failures.csv に保存
"""

from __future__ import annotations
import argparse, csv, re, subprocess, tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from zipfile import ZipFile

INVOKE_RE = re.compile(
    r"^\s*invoke-(?:virtual|static|direct|interface|super)(?:/range)?\s+.*?,\s*(L[^;]+;)->([^\(]+)\((.*?)\)(\S)")

def disassemble_all_dex(apk_path: Path, baksmali_jar: Path, smali_out: Path) -> None:
    smali_out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dexdir = Path(tmp) / "dex"; dexdir.mkdir()
        with ZipFile(apk_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".dex"):
                    zf.extract(name, dexdir)
        for dex in sorted(dexdir.rglob("*.dex")):
            cmd = ["java","-jar",str(baksmali_jar),"disassemble","--ignore-errors","-o",str(smali_out),str(dex)]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_tokens_from_smali(smali_root: Path,
                              include: Optional[Tuple[str, ...]],
                              exclude: Optional[Tuple[str, ...]],
                              include_sig: bool=False) -> List[str]:
    toks: List[str] = []
    for smali in smali_root.rglob("*.smali"):
        try:
            with open(smali, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = INVOKE_RE.match(line)
                    if not m: 
                        continue
                    clazz, method, args, ret = m.groups()
                    if include and not any(clazz.startswith(p) for p in include):
                        continue
                    if exclude and any(clazz.startswith(p) for p in exclude):
                        continue
                    if include_sig:
                        toks.append(f"{clazz}->{method}({args}){ret}")
                    else:
                        toks.append(f"{clazz}->{method}")
        except Exception:
            continue
    return toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apk-dir", required=True, help="APK が入ったディレクトリ")
    ap.add_argument("--out-dir", required=True, help="出力ディレクトリ（*.txt）")
    ap.add_argument("--baksmali-jar", default="baksmali.jar", help="baksmali.jar のパス")
    ap.add_argument("--include", nargs="*", default=["Landroid/","Ljava/","Lkotlin/"],
                    help="含めるクラス接頭辞（空にしたい場合は --include-none）")
    ap.add_argument("--include-none", action="store_true", help="include フィルタなし")
    ap.add_argument("--exclude", nargs="*", default=["Landroidx/test/"], help="除外する接頭辞")
    ap.add_argument("--include-sig", action="store_true", help="(args)return を含む完全シグネチャで出力")
    ap.add_argument("--workers", type=int, default=0, help="将来用（今は単純直列）。0=自動")
    args = ap.parse_args()

    apk_dir    = Path(args.apk_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    baksmali   = Path(args.baksmali_jar)
    include    = None if args.include_none else tuple(args.include) if args.include else None
    exclude    = tuple(args.exclude) if args.exclude else None

    failures = []
    apks = sorted(apk_dir.glob("*.apk"))
    for i, apk in enumerate(apks, 1):
        out_txt = out_dir / f"{apk.stem}.txt"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                smali_root = Path(tmp) / "smali"
                disassemble_all_dex(apk, baksmali, smali_root)
                toks = extract_tokens_from_smali(smali_root, include, exclude, include_sig=args.include_sig)
            out_txt.write_text(" ".join(toks) + "\n", encoding="utf-8")
            print(f"[{i}/{len(apks)}] {apk.name}: {len(toks)} tokens")
        except Exception as e:
            failures.append((apk.name, repr(e)))
            print(f"[{i}/{len(apks)}] {apk.name}: FAILED ({e})")

    if failures:
        fail_csv = out_dir / "failures.csv"
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["apk","error"])
            w.writerows(failures)
        print(f"Completed with {len(failures)} failures (see {fail_csv}).")
    else:
        print("Completed with 0 failures.")

if __name__ == "__main__":
    main()
