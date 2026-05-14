"""CNN utilities with lazy imports for optional training dependencies."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SUPPORTED_MODEL_NAMES",
    "Tiny3Conv",
    "get_model",
    "TrainEvalConfig",
    "TuneConfig",
    "run_train_eval_mrun",
    "run_tune_cnn",
    "run_tune_cnn_by_model",
]


def __getattr__(name: str) -> Any:
    if name in {"SUPPORTED_MODEL_NAMES", "Tiny3Conv", "get_model"}:
        from .models import SUPPORTED_MODEL_NAMES, Tiny3Conv, get_model

        exports = {
            "SUPPORTED_MODEL_NAMES": SUPPORTED_MODEL_NAMES,
            "Tiny3Conv": Tiny3Conv,
            "get_model": get_model,
        }
        return exports[name]

    if name in {
        "TrainEvalConfig",
        "TuneConfig",
        "run_train_eval_mrun",
        "run_tune_cnn",
        "run_tune_cnn_by_model",
    }:
        from .train_eval_mrun import (
            TrainEvalConfig,
            TuneConfig,
            run_train_eval_mrun,
            run_tune_cnn,
            run_tune_cnn_by_model,
        )

        exports = {
            "TrainEvalConfig": TrainEvalConfig,
            "TuneConfig": TuneConfig,
            "run_train_eval_mrun": run_train_eval_mrun,
            "run_tune_cnn": run_tune_cnn,
            "run_tune_cnn_by_model": run_tune_cnn_by_model,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
