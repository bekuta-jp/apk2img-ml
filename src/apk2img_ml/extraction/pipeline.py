from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .androguard_backend import extract_with_androguard
from .apktool_backend import extract_with_apktool
from .baksmali_backend import extract_with_baksmali
from .config import TokenFilter


class ExtractionBackend(str, Enum):
    HYBRID = "hybrid"
    ANDROGUARD = "androguard"
    BAKSMALI = "baksmali"
    APKTOOL = "apktool"


@dataclass
class ExtractionResult:
    apk_path: Path
    output_path: Path
    backend: str
    token_count: int
    ok: bool
    skipped: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class BackendOptions:
    baksmali_jar: Path = Path("baksmali.jar")
    java_cmd: str = "java"
    include_assets_dex: bool = False
    apktool_cmd: str = "apktool"
    no_res: bool = True
    retry_strip_assets: bool = True


def _collect_apks(input_dir: Path, recursive: bool) -> list[Path]:
    pattern_iter: Iterable[Path]
    if recursive:
        pattern_iter = input_dir.rglob("*.apk")
    else:
        pattern_iter = input_dir.glob("*.apk")
    return sorted(path for path in pattern_iter if path.is_file())


def _extract_with_backend(
    apk_path: Path,
    backend: ExtractionBackend,
    token_filter: TokenFilter,
    options: BackendOptions,
) -> list[str]:
    if backend == ExtractionBackend.ANDROGUARD:
        return extract_with_androguard(apk_path, token_filter)
    if backend == ExtractionBackend.BAKSMALI:
        return extract_with_baksmali(
            apk_path,
            token_filter,
            baksmali_jar=options.baksmali_jar,
            java_cmd=options.java_cmd,
            include_assets_dex=options.include_assets_dex,
        )
    if backend == ExtractionBackend.APKTOOL:
        return extract_with_apktool(
            apk_path,
            token_filter,
            apktool_cmd=options.apktool_cmd,
            no_res=options.no_res,
            retry_strip_assets=options.retry_strip_assets,
        )
    raise ValueError(f"unsupported backend: {backend}")


def extract_tokens(
    apk_path: Path,
    *,
    backend: ExtractionBackend = ExtractionBackend.HYBRID,
    token_filter: Optional[TokenFilter] = None,
    options: Optional[BackendOptions] = None,
) -> tuple[list[str], str]:
    """Extract API tokens from one APK and return (tokens, backend_used)."""

    token_filter = token_filter or TokenFilter()
    options = options or BackendOptions()

    if backend != ExtractionBackend.HYBRID:
        tokens = _extract_with_backend(apk_path, backend, token_filter, options)
        return tokens, backend.value

    errors: list[str] = []
    chain = [ExtractionBackend.ANDROGUARD, ExtractionBackend.BAKSMALI, ExtractionBackend.APKTOOL]
    for candidate in chain:
        try:
            tokens = _extract_with_backend(apk_path, candidate, token_filter, options)
        except Exception as exc:
            errors.append(f"{candidate.value}: {exc}")
            continue

        if tokens:
            return tokens, candidate.value
        if not token_filter.fallback_on_empty:
            return tokens, candidate.value

        errors.append(f"{candidate.value}: empty tokens")

    raise RuntimeError("all extraction backends failed: " + " | ".join(errors))


def _write_tokens(out_path: Path, tokens: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")


def _target_path(apk_path: Path, input_dir: Path, output_dir: Path) -> Path:
    try:
        rel = apk_path.relative_to(input_dir)
    except ValueError:
        rel = apk_path.name
    rel_path = Path(rel)
    return output_dir / rel_path.with_suffix(".txt")


def _extract_and_write(
    apk_path: Path,
    input_dir: Path,
    output_dir: Path,
    backend: ExtractionBackend,
    token_filter: TokenFilter,
    options: BackendOptions,
    overwrite: bool,
) -> ExtractionResult:
    out_path = _target_path(apk_path, input_dir, output_dir)

    if out_path.exists() and not overwrite:
        return ExtractionResult(
            apk_path=apk_path,
            output_path=out_path,
            backend="skipped",
            token_count=0,
            ok=True,
            skipped=True,
        )

    try:
        tokens, used_backend = extract_tokens(
            apk_path,
            backend=backend,
            token_filter=token_filter,
            options=options,
        )
        _write_tokens(out_path, tokens)
        return ExtractionResult(
            apk_path=apk_path,
            output_path=out_path,
            backend=used_backend,
            token_count=len(tokens),
            ok=True,
        )
    except Exception as exc:
        return ExtractionResult(
            apk_path=apk_path,
            output_path=out_path,
            backend=backend.value,
            token_count=0,
            ok=False,
            error=str(exc),
        )


def batch_extract(
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = True,
    backend: ExtractionBackend = ExtractionBackend.HYBRID,
    token_filter: Optional[TokenFilter] = None,
    options: Optional[BackendOptions] = None,
    workers: int = 1,
    overwrite: bool = False,
    failures_csv: Optional[Path] = None,
) -> list[ExtractionResult]:
    """Batch extract API tokens from APK files under input_dir."""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    token_filter = token_filter or TokenFilter()
    options = options or BackendOptions()

    apks = _collect_apks(input_dir, recursive=recursive)
    if not apks:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExtractionResult] = []

    if workers <= 1:
        iterator = apks
        if tqdm is not None:
            iterator = tqdm(apks, desc="extract", unit="apk")
        for apk_path in iterator:
            result = _extract_and_write(
                apk_path,
                input_dir,
                output_dir,
                backend,
                token_filter,
                options,
                overwrite,
            )
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _extract_and_write,
                    apk_path,
                    input_dir,
                    output_dir,
                    backend,
                    token_filter,
                    options,
                    overwrite,
                )
                for apk_path in apks
            ]

            progress = tqdm(total=len(futures), desc="extract", unit="apk") if tqdm is not None else None
            for future in as_completed(futures):
                results.append(future.result())
                if progress is not None:
                    progress.update(1)
            if progress is not None:
                progress.close()

    results.sort(key=lambda item: str(item.apk_path))

    if failures_csv is not None:
        failed = [item for item in results if not item.ok]
        if failed:
            failures_csv.parent.mkdir(parents=True, exist_ok=True)
            with failures_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["apk_path", "output_path", "backend", "error"])
                for item in failed:
                    writer.writerow([item.apk_path, item.output_path, item.backend, item.error or ""])

    return results
