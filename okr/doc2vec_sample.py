#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set

from androguard.misc import AnalyzeAPK  # Androguard

# ★ADDED: 並列実行・進捗表示・ログ出力用
from multiprocessing import Pool, cpu_count
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

# ★ADDED: androguard が内部で使っている loguru ロガー
from loguru import logger  # androguard.misc 等のログを捕まえる


def find_apk_files(input_dir: Path) -> List[Path]:
    """input_dir 以下の全ての .apk を再帰的に列挙する"""
    apk_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".apk"):
                apk_files.append(Path(root) / f)
    return apk_files


def extract_api_tokens_from_apk(apk_path: Path) -> List[str]:
    """
    1 APK から API 呼び出しトークンを抽出する。

    方針:
      - AnalyzeAPK で解析 (a, d, dx)
      - dx.get_methods() でアプリ内メソッドを走査
      - 各メソッドの get_xref_to() で呼び出し先メソッドを取得
      - 呼び出し先が "外部メソッド" (is_external() == True) のものだけを API とみなす
      - "Landroid/telephony/SmsManager;->sendTextMessage(Ljava/lang/String;...)V" のような
        文字列をトークンとして返す

    ※ Androguard のバージョンによって MethodAnalysis の API が少し違う場合があります。
       もし AttributeError が出たら print(dir(obj)) などで微調整してください。
    """
    try:
        a, d, dx = AnalyzeAPK(str(apk_path))
    except Exception as e:
        print(f"[ERROR] AnalyzeAPK failed for {apk_path}: {e}", file=sys.stderr)
        return []

    api_tokens: Set[str] = set()

    # dx.get_methods() は MethodAnalysis の一覧
    for m in dx.get_methods():
        # アプリ自身のメソッドだけを見る（外部メソッド本体は除外）
        if m.is_external():
            continue

        # m.get_xref_to() は「このメソッドから呼び出しているメソッド」の一覧
        # 要素は (ClassAnalysis, MethodAnalysis, offset) のタプル
        try:
            for _, called_m, _ in m.get_xref_to():
                # 呼び出し先が外部メソッドなら API とみなす
                if not called_m.is_external():
                    continue

                # called_m.method は androguard.core.bytecodes.dvm.EncodedMethod
                method_obj = called_m.method
                class_name = method_obj.get_class_name()    # 例: Landroid/telephony/SmsManager;
                name = method_obj.get_name()               # 例: sendTextMessage
                desc = method_obj.get_descriptor()         # 例: (Ljava/lang/String;...)V

                full_sig = f"{class_name}->{name}{desc}"
                api_tokens.add(full_sig)
        except Exception as e:
            # APK 内の一部メソッド解析に失敗しても、とりあえず続行
            print(f"[WARN] xref parse failed in {apk_path}: {e}", file=sys.stderr)
            continue

    # Doc2Vec は Bag-of-Words なので順序はあまり重要でないが、
    # 安定のためソートしておく
    return sorted(api_tokens)


# ★ADDED: loguru (androguard) の出力先をログファイルに切り替える
def setup_androguard_logger(log_file: str) -> None:
    """
    loguru のデフォルト sink (stderr) を削除して、log_file へ出力するように設定。
    各プロセス内で呼び出しても問題ない（プロセスごとに logger は独立）。
    """
    try:
        logger.remove()
    except Exception:
        # 既にハンドラが無い場合など
        pass

    # enqueue=False でシンプルにファイルへ書き出し
    logger.add(
        log_file,
        level="DEBUG",       # サンプルと同じく DEBUG まで全部出す
        encoding="utf-8",
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )


# ★ADDED: 1 APK を処理するワーカー関数（並列実行用）
def _process_single_apk(args):
    """
    並列ワーカー用のラッパ関数。
    androguard の詳細ログなど「標準出力に出るもの」を log_file に送る。
    メインプロセス側の tqdm 以外はコンソールに出さない。
    """
    apk_path, input_dir, output_dir, ext, log_file = args

    # ★ADDED: このプロセス内の androguard/loguru ログをファイルに送る
    if log_file is not None:
        setup_androguard_logger(log_file)

    rel = apk_path.relative_to(input_dir)          # 例: benign/app1.apk
    out_dir_for_apk = output_dir / rel.parent      # 例: output_dir/benign
    out_dir_for_apk.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_for_apk / (apk_path.stem + ext)

    # ログファイルに stdout / stderr をリダイレクト
    # （print した内容や、loguru 以外の標準出力もこちらに流れる）
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as lf, \
             redirect_stdout(lf), redirect_stderr(lf):
            print(f"[INFO] Processing: {apk_path} -> {out_path}", flush=True)
            tokens = extract_api_tokens_from_apk(apk_path)
    else:
        # log_file 指定が無ければ従来通り標準出力へ
        print(f"[INFO] Processing: {apk_path} -> {out_path}", file=sys.stderr)
        tokens = extract_api_tokens_from_apk(apk_path)

    # 抽出結果を書き出し
    with out_path.open("w", encoding="utf-8") as f:
        if tokens:
            f.write(" ".join(tokens) + "\n")
        else:
            # API が 1 つも取れなかった場合は空行だけ残す
            f.write("\n")

    # tqdm 用に戻り値は特に使わないが、完了を示すため True を返す
    return True


def main():
    parser = argparse.ArgumentParser(
        description="APK から API 呼び出しトークンを抽出して Doc2Vec 用コーパスを作成するスクリプト"
    )
    parser.add_argument("input_dir", type=str,
                        help="解析対象 APK が格納されたルートディレクトリ")
    parser.add_argument("output_dir", type=str,
                        help="API コーパスを書き出すルートディレクトリ")
    parser.add_argument("--ext", type=str, default=".txt",
                        help="出力コーパスファイルの拡張子 (デフォルト: .txt)")

    # ★ADDED: 並列実行数指定オプション
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="並列実行するプロセス数 (デフォルト: 1; 0 以下で CPU コア数を自動設定)")

    # ★ADDED: ログファイル指定オプション
    parser.add_argument("--log-file", type=str, default=None,
                        help="androguard などの標準出力を保存するログファイルパス "
                             "(デフォルト: output_dir/extract_api_doc2vec.log)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] input_dir が存在しません: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ★ADDED: ログファイルのデフォルトパスを決定
    log_file = Path(args.log_file) if args.log_file else (output_dir / "extract_api_doc2vec.log")
    log_file = str(log_file)  # ワーカーには str で渡す

    apk_files = find_apk_files(input_dir)
    total = len(apk_files)
    if total == 0:
        print("[INFO] APK files not found.", file=sys.stderr)
        return

    # ★CHANGED: 並列実行用のタスクリストを作成
    tasks = [
        (apk_path, input_dir, output_dir, args.ext, log_file)
        for apk_path in apk_files
    ]

    # ★CHANGED: jobs の解釈（0 以下なら CPU コア数にする）
    jobs = args.jobs
    if jobs is None or jobs <= 0:
        jobs = cpu_count()

    # ★CHANGED: tqdm で進捗バーを出しつつ、並列 or 逐次で実行
    if jobs == 1:
        # 逐次実行（進捗表示は tqdm、詳細ログはログファイルへ）
        for t in tqdm(tasks, total=total, desc="Processing APKs", file=sys.stdout):
            _process_single_apk(t)
    else:
        # 並列実行
        with Pool(processes=jobs) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_single_apk, tasks),
                total=total,
                desc=f"Processing APKs (jobs={jobs})",
                file=sys.stdout,
            ):
                pass

    print(f"[INFO] Finished. Log: {log_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
