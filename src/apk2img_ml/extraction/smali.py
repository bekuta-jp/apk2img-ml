from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from .config import TokenFilter, format_token

INVOKE_RE = re.compile(
    r"^\\s*invoke-(?:virtual|static|direct|interface|super)(?:/range)?\\s+.*?,\\s*"
    r"(L[^;]+;)->([^\\(]+)\\((.*?)\\)(\\S)"
)


def _iter_smali_files(root: Path) -> Iterable[Path]:
    smali_dirs = [d for d in root.glob("smali*") if d.is_dir()]
    if smali_dirs:
        for smali_dir in smali_dirs:
            yield from smali_dir.rglob("*.smali")
        return
    if root.is_dir():
        yield from root.rglob("*.smali")


def extract_tokens_from_smali_tree(root: Path, token_filter: TokenFilter) -> List[str]:
    tokens: List[str] = []
    for smali_path in _iter_smali_files(root):
        try:
            with smali_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    match = INVOKE_RE.match(line)
                    if not match:
                        continue
                    class_name, method_name, args_sig, ret_sig = match.groups()
                    if not token_filter.allows_class(class_name):
                        continue
                    tokens.append(
                        format_token(
                            class_name,
                            method_name,
                            args_sig,
                            ret_sig,
                            include_signature=token_filter.include_signature,
                        )
                    )
        except OSError:
            continue
    return tokens
