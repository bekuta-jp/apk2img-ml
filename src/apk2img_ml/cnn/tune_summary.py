from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TABLE_COLUMNS = (
    "dataset",
    "pretrained",
    "model",
    "tune_ts",
    "best_eval_ts",
    "val_acc",
    "test_acc",
    "macro_f1",
)
SUMMARY_COLUMNS = DEFAULT_TABLE_COLUMNS + (
    "micro_f1",
    "weighted_f1",
    "benign_f1",
    "malware_f1",
)
SUMMARY_SORT_COLUMNS = SUMMARY_COLUMNS + (
    "tune_timestamp",
    "best_eval_timestamp",
    "binary_benign_f1",
    "binary_malware_f1",
)


@dataclass(frozen=True)
class OptunaResultRecord:
    dataset: str
    pretrained: bool | None
    pretrained_label: str
    model: str
    tune_timestamp: str
    best_eval_timestamp: str | None
    tuned_parameters: dict[str, str]
    best_params: dict[str, Any]
    objective_name: str
    best_objective_value: float | None
    test_acc: float | None
    macro_f1: float | None
    micro_f1: float | None
    weighted_f1: float | None
    binary_benign_f1: float | None
    binary_malware_f1: float | None
    tune_log_path: Path
    best_eval_log_path: Path | None


@dataclass(frozen=True)
class OptunaSummaryReport:
    results_root: Path
    records: tuple[OptunaResultRecord, ...]
    warnings: tuple[str, ...]


def summarize_optuna_results(
    results_root: Path,
    *,
    latest_only: bool = False,
) -> OptunaSummaryReport:
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"results root does not exist: {root}")

    records: list[OptunaResultRecord] = []
    warnings: list[str] = []

    for tune_log_path in sorted(root.rglob("optuna_tuning_log.json")):
        try:
            records.append(_build_record(tune_log_path))
        except Exception as exc:
            warnings.append(f"Skipped {tune_log_path}: {exc}")

    records = _sort_records(records, sort_by="dataset")
    if latest_only:
        records = _latest_records(records)

    return OptunaSummaryReport(
        results_root=root,
        records=tuple(records),
        warnings=tuple(warnings),
    )


def render_optuna_summary_report(
    report: OptunaSummaryReport,
    *,
    sort_by: str | tuple[str, ...] = "dataset",
    table_only: bool = False,
    ascending: bool = False,
) -> str:
    records = _sort_records(list(report.records), sort_by=sort_by, ascending=ascending)
    lines: list[str] = []
    if not table_only:
        lines.append(f"Found {len(records)} Optuna runs under: {report.results_root}")

    if not records:
        if table_only:
            lines.append(f"No Optuna runs found under: {report.results_root}")
        if report.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in report.warnings)
        return "\n".join(lines)

    overview_rows = [
        [_column_display_value(record, column) for column in DEFAULT_TABLE_COLUMNS]
        for record in records
    ]
    if not table_only:
        lines.append("")
    lines.extend(
        _render_table(
            headers=list(DEFAULT_TABLE_COLUMNS),
            rows=overview_rows,
        )
    )

    if not table_only:
        for index, record in enumerate(records, start=1):
            lines.append("")
            lines.append(
                f"[{index}] dataset={record.dataset} pretrained={record.pretrained_label} model={record.model}"
            )
            lines.append(f"  tune_timestamp : {record.tune_timestamp}")
            lines.append(f"  best_eval_ts   : {record.best_eval_timestamp or '-'}")
            lines.append(
                f"  changed        : {_fmt_mapping(record.tuned_parameters) or 'none detected'}"
            )
            lines.append(f"  best_params    : {_fmt_mapping(record.best_params) or '-'}")
            lines.append(
                "  best_metrics   : "
                f"{record.objective_name}={_fmt_float(record.best_objective_value)}, "
                f"test_acc={_fmt_float(record.test_acc)}, "
                f"macro_f1={_fmt_float(record.macro_f1)}, "
                f"micro_f1={_fmt_float(record.micro_f1)}, "
                f"weighted_f1={_fmt_float(record.weighted_f1)}, "
                f"benign_f1={_fmt_float(record.binary_benign_f1)}, "
                f"malware_f1={_fmt_float(record.binary_malware_f1)}"
            )
            lines.append(f"  tune_log       : {record.tune_log_path}")
            if record.best_eval_log_path is not None:
                lines.append(f"  best_eval_log  : {record.best_eval_log_path}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report.warnings)

    return "\n".join(lines)


def _build_record(tune_log_path: Path) -> OptunaResultRecord:
    tune_log = _load_json(tune_log_path)
    best_trial = tune_log.get("best_trial")
    if not isinstance(best_trial, dict) or not best_trial:
        raise ValueError("best_trial is missing")

    args = tune_log.get("args")
    if not isinstance(args, dict):
        args = {}

    best_params = best_trial.get("params")
    if not isinstance(best_params, dict):
        best_params = {}

    best_eval_block = tune_log.get("best_eval")
    if not isinstance(best_eval_block, dict):
        best_eval_block = {}

    best_eval_log_path = _resolve_best_eval_log_path(tune_log_path, best_eval_block)
    best_eval_log = _load_json(best_eval_log_path) if best_eval_log_path is not None else None
    best_eval_summary = _best_eval_summary(best_eval_log, best_eval_block)

    dataset = _infer_dataset_name(tune_log_path, args)
    pretrained = _infer_pretrained_flag(tune_log_path, args)
    pretrained_label = _format_pretrained_label(pretrained)

    model = str(
        best_params.get("model")
        or _get_nested(best_eval_log, "args", "model")
        or _first_or_none(args.get("model_candidates"))
        or "unknown"
    )

    return OptunaResultRecord(
        dataset=dataset,
        pretrained=pretrained,
        pretrained_label=pretrained_label,
        model=model,
        tune_timestamp=str(tune_log.get("timestamp") or tune_log_path.parent.name),
        best_eval_timestamp=_best_eval_timestamp(best_eval_log, best_eval_block, best_eval_log_path),
        tuned_parameters=_extract_tuned_parameters(args),
        best_params=best_params,
        objective_name=str(tune_log.get("objective") or "best_val_acc"),
        best_objective_value=_as_float(best_trial.get("value")),
        test_acc=_metric_from_summary(best_eval_summary, "test_acc_mean", "test_acc"),
        macro_f1=_metric_from_summary(best_eval_summary, "macro_f1_mean", "macro"),
        micro_f1=_metric_from_summary(best_eval_summary, "micro_f1_mean", "micro"),
        weighted_f1=_metric_from_summary(best_eval_summary, "weighted_f1_mean", "weighted"),
        binary_benign_f1=_metric_from_summary(
            best_eval_summary, "binary_benign_f1_mean", "binary_benign"
        ),
        binary_malware_f1=_metric_from_summary(
            best_eval_summary, "binary_malware_f1_mean", "binary_malware"
        ),
        tune_log_path=tune_log_path,
        best_eval_log_path=best_eval_log_path,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _resolve_best_eval_log_path(
    tune_log_path: Path,
    best_eval_block: dict[str, Any],
) -> Path | None:
    local_best_eval_dir = tune_log_path.parent / "best_eval"
    candidates: list[Path] = []

    stored_log_path = best_eval_block.get("log_json_path")
    if isinstance(stored_log_path, str) and stored_log_path:
        remote_path = Path(stored_log_path)
        candidates.append(remote_path)
        if remote_path.parent.name:
            candidates.append(local_best_eval_dir / remote_path.parent.name / remote_path.name)

    stored_run_dir = best_eval_block.get("run_dir")
    if isinstance(stored_run_dir, str) and stored_run_dir:
        run_name = Path(stored_run_dir).name
        if run_name:
            candidates.append(local_best_eval_dir / run_name / "train_log.json")

    if local_best_eval_dir.exists():
        local_logs = sorted(local_best_eval_dir.glob("*/train_log.json"), reverse=True)
        candidates.extend(local_logs)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _best_eval_summary(
    best_eval_log: dict[str, Any] | None,
    best_eval_block: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(best_eval_log, dict):
        summary = best_eval_log.get("summary")
        if isinstance(summary, dict):
            return summary

    summary = best_eval_block.get("summary")
    if isinstance(summary, dict):
        return summary

    return {}


def _best_eval_timestamp(
    best_eval_log: dict[str, Any] | None,
    best_eval_block: dict[str, Any],
    best_eval_log_path: Path | None,
) -> str | None:
    timestamp = None
    if isinstance(best_eval_log, dict):
        raw_timestamp = best_eval_log.get("timestamp")
        if raw_timestamp is not None:
            timestamp = str(raw_timestamp)

    if timestamp is None:
        run_dir = best_eval_block.get("run_dir")
        if isinstance(run_dir, str) and run_dir:
            timestamp = Path(run_dir).name

    if timestamp is None and best_eval_log_path is not None:
        timestamp = best_eval_log_path.parent.name

    return timestamp


def _infer_dataset_name(tune_log_path: Path, args: dict[str, Any]) -> str:
    condition_dir = _find_condition_dir(tune_log_path)
    if condition_dir is not None and condition_dir.parent.name:
        return condition_dir.parent.name

    data_root = args.get("data_root")
    if isinstance(data_root, str) and data_root:
        return Path(data_root).name

    return tune_log_path.parent.name


def _infer_pretrained_flag(tune_log_path: Path, args: dict[str, Any]) -> bool | None:
    pretrained = args.get("pretrained")
    if isinstance(pretrained, bool):
        return pretrained

    condition_dir = _find_condition_dir(tune_log_path)
    if condition_dir is None:
        return None

    suffix = condition_dir.name.removeprefix("cnn_optuna_").lower()
    if suffix in {"pre", "pretrained", "true", "yes", "on"}:
        return True
    if suffix in {"no", "false", "scratch", "nopre", "off"}:
        return False
    return None


def _find_condition_dir(tune_log_path: Path) -> Path | None:
    for parent in tune_log_path.parents:
        if parent.name.startswith("cnn_optuna_") or parent.name.startswith("optuna_cnn_"):
            return parent
    return None


def _extract_tuned_parameters(args: dict[str, Any]) -> dict[str, str]:
    tuned: dict[str, str] = {}

    model_candidates = _normalize_sequence(args.get("model_candidates"))
    if len(model_candidates) > 1:
        tuned["model"] = _fmt_sequence(model_candidates)

    batch_candidates = _normalize_sequence(args.get("batch_candidates"))
    if len(batch_candidates) > 1:
        tuned["batch"] = _fmt_sequence(batch_candidates)

    optimizer_candidates = _normalize_sequence(args.get("optimizer_candidates"))
    if len(optimizer_candidates) > 1:
        tuned["optimizer"] = _fmt_sequence(optimizer_candidates)

    weight_decay_candidates = _normalize_sequence(args.get("weight_decay_candidates"))
    if len(weight_decay_candidates) > 1:
        tuned["weight_decay"] = _fmt_sequence(weight_decay_candidates)

    lr_low = args.get("lr_low")
    lr_high = args.get("lr_high")
    if lr_low is not None and lr_high is not None and lr_low != lr_high:
        tuned["lr"] = f"[{_fmt_scalar(lr_low)}, {_fmt_scalar(lr_high)}] (log)"

    return tuned


def _metric_from_summary(summary: dict[str, Any], primary_key: str, legacy_key: str) -> float | None:
    primary = summary.get(primary_key)
    if primary is not None:
        return _as_float(primary)

    runs = summary.get("runs")
    if isinstance(runs, list) and runs:
        first_run = runs[0]
        if isinstance(first_run, dict):
            if legacy_key == "test_acc":
                return _as_float(first_run.get("test_acc"))
            test_f1 = first_run.get("test_f1")
            if isinstance(test_f1, dict):
                return _as_float(test_f1.get(legacy_key))
    return None


def _get_nested(data: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_or_none(value: Any) -> Any:
    items = _normalize_sequence(value)
    return items[0] if items else None


def _normalize_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _latest_records(records: list[OptunaResultRecord]) -> list[OptunaResultRecord]:
    latest: dict[tuple[str, str, str], OptunaResultRecord] = {}
    for record in records:
        key = (record.dataset, record.pretrained_label, record.model)
        current = latest.get(key)
        if current is None or record.tune_timestamp > current.tune_timestamp:
            latest[key] = record
    return _sort_records(list(latest.values()), sort_by="dataset")


def available_summary_sort_columns() -> tuple[str, ...]:
    return SUMMARY_SORT_COLUMNS


def parse_sort_columns(values: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if values is None:
        return ("dataset",)

    raw_values: list[str]
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = [str(value) for value in values]

    columns: list[str] = []
    valid_columns = set(SUMMARY_SORT_COLUMNS)
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            column = piece.strip()
            if not column:
                continue
            normalized = _normalize_sort_column(column)
            if normalized not in valid_columns:
                supported = ", ".join(SUMMARY_SORT_COLUMNS)
                raise ValueError(f"unsupported sort column: {column}. supported: {supported}")
            columns.append(normalized)

    if not columns:
        return ("dataset",)
    return tuple(columns)


def _sort_records(
    records: list[OptunaResultRecord],
    *,
    sort_by: str | tuple[str, ...],
    ascending: bool = False,
) -> list[OptunaResultRecord]:
    normalized_columns = parse_sort_columns(sort_by)
    sorted_records = list(records)
    for normalized in reversed(normalized_columns):
        sorted_records = _sort_records_one_key(
            sorted_records,
            sort_by=normalized,
            ascending=ascending,
        )
    return sorted_records


def _sort_records_one_key(
    records: list[OptunaResultRecord],
    *,
    sort_by: str,
    ascending: bool = False,
) -> list[OptunaResultRecord]:
    normalized = _normalize_sort_column(sort_by)

    if normalized == "val_acc":
        return sorted(
            records,
            key=lambda record: (_sort_float(record.best_objective_value), record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "test_acc":
        return sorted(
            records,
            key=lambda record: (_sort_float(record.test_acc), record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "macro_f1":
        return sorted(
            records,
            key=lambda record: (_sort_float(record.macro_f1), record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "micro_f1":
        return sorted(
            records,
            key=lambda record: (_sort_float(record.micro_f1), record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "weighted_f1":
        return sorted(
            records,
            key=lambda record: (_sort_float(record.weighted_f1), record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "benign_f1":
        return sorted(
            records,
            key=lambda record: (_sort_float(record.binary_benign_f1), record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "malware_f1":
        return sorted(
            records,
            key=lambda record: (_sort_float(record.binary_malware_f1), record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "tune_ts":
        return sorted(records, key=lambda record: record.tune_timestamp, reverse=not ascending)
    if normalized == "best_eval_ts":
        return sorted(
            records,
            key=lambda record: (record.best_eval_timestamp or "", record.tune_timestamp),
            reverse=not ascending,
        )
    if normalized == "pretrained":
        return sorted(
            records,
            key=lambda record: (
                record.pretrained_label,
                record.dataset,
                record.model,
                record.tune_timestamp,
            ),
        )
    if normalized == "model":
        return sorted(
            records,
            key=lambda record: (
                record.model,
                record.dataset,
                record.pretrained_label,
                record.tune_timestamp,
            ),
        )

    return sorted(
        records,
        key=lambda record: (
            record.dataset,
            record.pretrained_label,
            record.model,
            record.tune_timestamp,
        ),
    )


def _normalize_sort_column(value: str) -> str:
    aliases = {
        "tune_timestamp": "tune_ts",
        "best_eval_timestamp": "best_eval_ts",
        "binary_benign_f1": "benign_f1",
        "binary_malware_f1": "malware_f1",
    }
    return aliases.get(value, value)


def _column_display_value(record: OptunaResultRecord, column: str) -> str:
    normalized = _normalize_sort_column(column)
    if normalized == "dataset":
        return record.dataset
    if normalized == "pretrained":
        return record.pretrained_label
    if normalized == "model":
        return record.model
    if normalized == "tune_ts":
        return record.tune_timestamp
    if normalized == "best_eval_ts":
        return record.best_eval_timestamp or "-"
    if normalized == "val_acc":
        return _fmt_float(record.best_objective_value)
    if normalized == "test_acc":
        return _fmt_float(record.test_acc)
    if normalized == "macro_f1":
        return _fmt_float(record.macro_f1)
    if normalized == "micro_f1":
        return _fmt_float(record.micro_f1)
    if normalized == "weighted_f1":
        return _fmt_float(record.weighted_f1)
    if normalized == "benign_f1":
        return _fmt_float(record.binary_benign_f1)
    if normalized == "malware_f1":
        return _fmt_float(record.binary_malware_f1)
    raise ValueError(f"unsupported summary column: {column}")


def _sort_float(value: float | None) -> float:
    return float("-inf") if value is None else value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_pretrained_label(pretrained: bool | None) -> str:
    if pretrained is True:
        return "true"
    if pretrained is False:
        return "false"
    return "unknown"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_sequence(values: list[Any]) -> str:
    return "[" + ", ".join(_fmt_scalar(value) for value in values) + "]"


def _fmt_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return ""
    return ", ".join(f"{key}={_fmt_value(value)}" for key, value in mapping.items())


def _fmt_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + _fmt_mapping(value) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_value(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(_fmt_value(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    rendered = [render_row(headers), separator]
    rendered.extend(render_row(row) for row in rows)
    return rendered
