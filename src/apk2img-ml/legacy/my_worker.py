# my_worker.py
# 並列実行用ワーカー関数をモジュール化（Jupyterのspawn問題を回避）
from __future__ import annotations
from pathlib import Path
from typing import Tuple

# 親・子プロセスどちらでも余計なログが出ないようにする初期化
def init_child_quiet() -> None:
    # Androguardは環境にloguruがあるとloguruで出力するため抑止
    try:
        from loguru import logger as _loguru
        _loguru.remove()               # 既定のコンソールシンク削除
        _loguru.disable("androguard")  # 名前空間ごと無効化
    except Exception:
        pass

def extract_and_write(
    apk_path_str: str,
    out_dir_str: str,
    include_prefixes: tuple[str, ...] = ("Landroid/", "Ljava/", "Lkotlin/"),
    exclude_prefixes: tuple[str, ...] = ("Landroidx/test/",),
) -> Tuple[str, bool, str]:
    """
    1 APK を解析して API トークン列を保存する。
    戻り値: (apk_path_str, 成功/失敗, 失敗時エラーメッセージ)
    """
    init_child_quiet()
    try:
        from apk2apis import extract_api_sequence  # 子プロセスでimport
        toks = extract_api_sequence(
            apk_path_str,
            include_pkg_prefixes=include_prefixes,
            exclude_pkg_prefixes=exclude_prefixes,
            show_progress=False,  # 並列時は内部tqdmはOFF
        )
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(apk_path_str).stem + ".txt")
        out_path.write_text(" ".join(toks) + "\n", encoding="utf-8")
        return (apk_path_str, True, "")
    except Exception as e:
        # 失敗しても全体は継続できるようエラー文字列を返す
        return (apk_path_str, False, repr(e))

__all__ = ["extract_and_write", "init_child_quiet"]
