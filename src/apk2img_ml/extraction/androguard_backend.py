from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .config import TokenFilter, format_token

INVOKE_PREFIXES = (
    "invoke-virtual",
    "invoke-static",
    "invoke-direct",
    "invoke-interface",
    "invoke-super",
)
METHOD_SIG_RE = re.compile(r"(L[^;]+;)->([^\(]+)\((.*?)\)(\S)")


def extract_with_androguard(apk_path: Path, token_filter: TokenFilter) -> List[str]:
    """Extract API call tokens from one APK with Androguard."""

    try:
        from androguard.misc import AnalyzeAPK
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("androguard is not available") from exc

    try:
        _apk, dex_units, _analysis = AnalyzeAPK(str(apk_path))
    except Exception as exc:
        raise RuntimeError(f"AnalyzeAPK failed: {exc}") from exc

    tokens: List[str] = []
    dex_list = dex_units if isinstance(dex_units, list) else [dex_units]

    for vm in dex_list:
        for class_def in vm.get_classes():
            for method in class_def.get_methods():
                if not hasattr(method, "get_code"):
                    continue
                code = method.get_code()
                if code is None:
                    continue
                try:
                    for instruction in code.get_bc().get_instructions():
                        op_name = instruction.get_name()
                        if not op_name.startswith(INVOKE_PREFIXES):
                            continue
                        out_text = instruction.get_output()
                        match = METHOD_SIG_RE.search(out_text)
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
                except Exception:
                    continue

    return tokens
