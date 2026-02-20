from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TokenFilter:
    """Filter settings for extracted API tokens."""

    include_prefixes: Optional[Tuple[str, ...]] = ("Landroid/", "Ljava/", "Lkotlin/")
    exclude_prefixes: Optional[Tuple[str, ...]] = ("Landroidx/test/",)
    include_signature: bool = False
    fallback_on_empty: bool = True

    def allows_class(self, class_name: str) -> bool:
        if self.include_prefixes:
            if not any(class_name.startswith(prefix) for prefix in self.include_prefixes):
                return False
        if self.exclude_prefixes:
            if any(class_name.startswith(prefix) for prefix in self.exclude_prefixes):
                return False
        return True


def format_token(
    class_name: str,
    method_name: str,
    args_sig: str,
    ret_sig: str,
    *,
    include_signature: bool,
) -> str:
    if include_signature:
        return f"{class_name}->{method_name}({args_sig}){ret_sig}"
    return f"{class_name}->{method_name}"
