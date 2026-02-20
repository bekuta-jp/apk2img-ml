#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APK から API コール列（トークン列）を抽出して 1行テキストに出力します。
- import 用: extract_api_sequence(apk_path, filters=...) を利用
- CLI 用  : python apk2apis.py --apk path/app.apk --out out_dir
"""
import argparse, os, re, sys
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Tuple

# Androguard
from androguard.misc import AnalyzeAPK

# --------------- 追加: tqdm（進捗表示用） ---------------
# CHANGED: 進捗表示のためにtqdmを使用可能にした（理由：大量メソッド処理の可視化）
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # tqdm未インストールでも動くようにフォールバック

# --------------- 追加: import 利用時のログ構成ヘルパ ---------------
# CHANGED: import 実行時にも「ERRORのみstderr、INFO/DEBUGはログファイルへ」を適用できるように
#          呼び出し側（Jupyter等）で一度呼ぶだけの設定関数を追加（理由：main() だけではCLI時しか反映されない）
import logging

def setup_logging_for_import(
    log_file: str = "apk2apis.log",
    file_level: int = logging.INFO,
    stderr_level: int = logging.ERROR,
    silence_androguard: bool = False,
):
    """
    import時のログ動作を構成：
      - stderr には `stderr_level` 以上のみ（通常は ERROR）
      - `log_file` へは `file_level` 以上（通常は INFO/DEBUG）
      - Androguard配下のすべてのロガーをファイルのみに流し、stderrを汚さない
    """
    # root: 既存ハンドラを外し、stderr用のみ再構成（Jupyterの既定Handler対策）
    root = logging.getLogger()
    # CHANGED: 既存ハンドラを必ず全削除（理由：Jupyterが付けたStreamHandlerがINFOを出すことがある）
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(min(file_level, stderr_level))
    err_h = logging.StreamHandler(sys.stderr)
    err_h.setLevel(stderr_level)
    err_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(err_h)

    # ファイルハンドラ（単一インスタンスを共有）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # 自モジュール（apk2apis）側: ファイルにのみ流す
    my = logging.getLogger("apk2apis")
    my.setLevel(file_level)
    # CHANGED: 重複付与防止（理由：Jupyterの再実行で多重出力を避ける）
    my.handlers.clear()
    my.addHandler(fh)
    # CHANGED: 親(root)へ伝播させない（理由：stderr 汚染防止）
    my.propagate = False

    # Androguard 側: ルート "androguard" だけでなく配下すべても強制的に付け替え
    # CHANGED: loggerDict を総なめして "androguard" で始まるロガーを一括設定（理由：サブロガーが個別ハンドラ/レベルを持つ場合がある）
    target_min_level = logging.ERROR if silence_androguard else logging.DEBUG
    logger_dict = logging.root.manager.loggerDict
    for name, logger in list(logger_dict.items()):
        if not isinstance(logger, logging.Logger):
            continue
        if name == "androguard" or name.startswith("androguard."):
            logger.setLevel(target_min_level)
            logger.handlers.clear()
            logger.addHandler(fh)
            logger.propagate = False  # CHANGED: 親へ伝播させない（理由：stderr 汚染防止）

    # 念のため親 "androguard" も明示的に設定
    ag = logging.getLogger("androguard")
    ag.setLevel(target_min_level)
    ag.handlers.clear()
    ag.addHandler(fh)
    ag.propagate = False

# --------------- core (import 用) ---------------
INVOKE_PREFIXES = ("invoke-virtual", "invoke-static", "invoke-direct",
                   "invoke-interface", "invoke-super")

# "Landroid/telephony/SmsManager;->sendTextMessage(Ljava/lang/String;...)V"
METHOD_SIG_RE: Pattern[str] = re.compile(r"(L[^;]+;)->([^\(]+)\(")

def extract_api_sequence(
    apk_path: str,
    include_pkg_prefixes: Optional[Tuple[str, ...]] = ("Landroid/", "Ljava/", "Lkotlin/"),
    exclude_pkg_prefixes: Optional[Tuple[str, ...]] = ("Landroidx/test/",),
    # CHANGED: 進捗表示を関数レベルでも切り替えられるようにした（理由：import利用でも使えるように）
    show_progress: bool = False,
) -> List[str]:
    """
    APK から API コール（呼び出し先の完全修飾メソッド）トークン列を抽出。
    例トークン: "Landroid/telephony/SmsManager;->sendTextMessage"
    - include_pkg_prefixes: この接頭辞のどれかで始まる呼び先のみ採用（None で無制限）
    - exclude_pkg_prefixes: この接頭辞は除外
    """
    a, d, dx = AnalyzeAPK(apk_path)  # a: APK, d: DalvikVMFormat or list, dx: Analysis
    tokens: List[str] = []

    # 複数 DEX の場合に対応
    for vm in (d if isinstance(d, list) else [d]):
        # CHANGED: vm.get_methods() だと MethodIdItem（参照テーブル）が混在し get_code() 不可の環境があるため、
        #          クラス -> メソッドで辿り EncodedMethod を確実に取得（理由：get_code() を安全に呼ぶ）。
        classes = list(vm.get_classes())
        it_classes = classes
        # CHANGED: tqdmが利用可能かつ要求された場合のみ進捗表示（理由：長時間処理の可視化）
        if show_progress and tqdm is not None:
            it_classes = tqdm(classes, desc="classes", unit="cls", leave=False)
        for cls in it_classes:  # ← ここを追加（理由は上記）
            methods = list(cls.get_methods())
            it_methods = methods
            if show_progress and tqdm is not None:
                it_methods = tqdm(methods, desc="methods", unit="mth", leave=False)
            for m in it_methods:  # ← EncodedMethod を得る
                # CHANGED: 一部環境で型差異があるため保険として hasattr ガードを追加（理由：堅牢性向上）
                if not hasattr(m, "get_code"):
                    continue
                code = m.get_code()
                if code is None:
                    continue
                try:
                    for ins in code.get_bc().get_instructions():
                        op = ins.get_name()
                        if not op.startswith(INVOKE_PREFIXES):
                            continue
                        out = ins.get_output()  # 文字列中に呼び先メソッド記号が入る
                        # 例: v3, Landroid/telephony/SmsManager;->sendTextMessage(...); ...
                        mt = METHOD_SIG_RE.search(out)
                        if not mt:
                            continue
                        clazz, method = mt.group(1), mt.group(2)  # "Lxxx/xxx;","methodName"
                        if include_pkg_prefixes is not None:
                            if not any(clazz.startswith(pref) for pref in include_pkg_prefixes):
                                continue
                        if exclude_pkg_prefixes is not None:
                            if any(clazz.startswith(pref) for pref in exclude_pkg_prefixes):
                                continue
                        tokens.append(f"{clazz}->{method}")
                except Exception:
                    # 壊れたメソッド等はスキップ（堅牢性優先）
                    continue
    return tokens

# --------------- CLI ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apk", required=True, help="入力 APK")
    ap.add_argument("--out", required=True, help="出力ディレクトリ（*.txt、1 APK = 1 行）")
    ap.add_argument("--stem", help="出力ファイル名の stem（未指定時はAPK名）")
    ap.add_argument("--include", nargs="*", default=["Landroid/", "Ljava/", "Lkotlin/"],
                    help="含めるパッケージ接頭辞（空にしたい場合は --include を省略し --include-none を使う）")
    ap.add_argument("--include-none", action="store_true",
                    help="include フィルタなし（全呼び先を対象）")
    ap.add_argument("--exclude", nargs="*", default=["Landroidx/test/"],
                    help="除外するパッケージ接頭辞")
    # --------------- 追加: ログと進捗のオプション ---------------
    # CHANGED: エラー以外のログをファイルに分離保存するための指定（理由：標準出力/標準エラーを汚さない）
    ap.add_argument("--log-file", default="apk2apis.log",
                    help="INFO/DEBUG を保存するログファイル（ERRORはstderrへ）")
    # CHANGED: tqdmで進捗表示を有効化（理由：長時間処理の体感改善）
    ap.add_argument("--tqdm", action="store_true",
                    help="進捗バーを表示（tqdmインストール時のみ）")

    args = ap.parse_args()

    # --------------- 追加: ログ分離設定 ---------------
    # CHANGED: エラーはstderrに、INFO/DEBUGはファイルに出すためのハンドラ構成（理由：ログ分離）
    import logging
    # root: ERROR以上のみstderr
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.ERROR)
    err_h = logging.StreamHandler(sys.stderr)
    err_h.setLevel(logging.ERROR)
    err_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(err_h)
    # file: INFO以上を保存
    file_h = logging.FileHandler(args.log_file, encoding="utf-8")
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # 自モジュール用ロガー
    logger = logging.getLogger("apk2apis")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_h)
    logger.propagate = False  # CHANGED: rootに伝播させず、stderrへ出さない（理由：分離）

    # Androguardの詳細ログもファイル側にのみ流す
    ag = logging.getLogger("androguard")
    ag.setLevel(logging.DEBUG)         # DEBUGも拾う
    ag.addHandler(file_h)
    ag.propagate = False               # CHANGED: rootへ伝播させない（理由：stderr汚染防止）

    os.makedirs(args.out, exist_ok=True)
    include = None if args.include_none else tuple(args.include) if args.include else None
    exclude = tuple(args.exclude) if args.exclude else None

    logger.info(f"start apk2apis apk={args.apk} out={args.out} stem={args.stem or Path(args.apk).stem} include={include} exclude={exclude}")

    # CHANGED: 関数にも show_progress を渡す（理由：CLI指定 --tqdm を反映）
    toks = extract_api_sequence(args.apk, include_pkg_prefixes=include, exclude_pkg_prefixes=exclude,
                                show_progress=args.tqdm)
    stem = args.stem or Path(args.apk).stem
    out_path = Path(args.out) / f"{stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(" ".join(toks) + "\n")

    logger.info(f"wrote: {out_path} tokens={len(toks)}")
    print(f"[apk2apis] wrote: {out_path} ({len(toks)} tokens)")

if __name__ == "__main__":
    main()
