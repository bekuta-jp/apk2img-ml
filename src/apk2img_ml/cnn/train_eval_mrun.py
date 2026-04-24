from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from apk2img_ml.cnn.models import get_model
else:
    from .models import get_model

try:  # pragma: no cover
    import japanize_matplotlib  # noqa: F401
except Exception:  # pragma: no cover
    japanize_matplotlib = None


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

VERBOSE_EPOCH = False
MODEL_HELP = (
    "tiny|alexnet|vgg16|resnet18|resnet34|resnet50|resnet101|resnet152|"
    "densenet|densenet121|mobilenet|mobilenet_v2|efficientnet_b0-b7|"
    "efficientnet_v2_s|efficientnet_v2_m|efficientnet_v2_l"
)
SUPPORTED_OPTIMIZERS = ("adam", "adamw", "sgd")
SUPPORTED_LR_SCHEDULERS = (
    "none",
    "step",
    "multistep",
    "exponential",
    "cosine",
    "plateau",
    "cosine_warm_restarts",
    "onecycle",
)
DEFAULT_TUNE_MODELS = ("resnet18", "resnet50", "mobilenet_v2")
DEFAULT_TUNE_BATCHES = (16, 32, 64)
DEFAULT_TUNE_OPTIMIZERS = ("adam", "adamw")
DEFAULT_TUNE_WEIGHT_DECAYS = (0.0, 1e-6, 1e-5, 1e-4)
DEFAULT_SCHEDULER_MILESTONES = (10, 20)


@dataclass(frozen=True)
class TrainEvalConfig:
    data_root: Path
    model: str = "resnet50"
    pretrained: bool = True
    epochs: int = 15
    batch: int = 32
    lr: float = 1e-4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    lr_scheduler: str = "none"
    scheduler_step_size: int = 5
    scheduler_milestones: tuple[int, ...] = DEFAULT_SCHEDULER_MILESTONES
    scheduler_gamma: float = 0.1
    scheduler_exp_gamma: float = 0.95
    scheduler_patience: int = 2
    scheduler_t_max: int | None = None
    scheduler_eta_min: float = 0.0
    scheduler_t_0: int | None = None
    scheduler_t_mult: int = 1
    scheduler_pct_start: float = 0.3
    scheduler_div_factor: float = 25.0
    scheduler_final_div_factor: float = 1e4
    workers: int = 4
    in_ch: int = 1
    resize: str = "256,256"
    seed: int = 3407
    runs: int = 1
    log_dir: Path = Path("logs")
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    restore_best: bool = True


@dataclass(frozen=True)
class TuneConfig:
    data_root: Path
    trials: int = 20
    epochs: int = 15
    pretrained: bool = True
    workers: int = 4
    in_ch: int = 1
    resize: str = "256,256"
    lr_scheduler: str = "none"
    scheduler_step_size: int = 5
    scheduler_milestones: tuple[int, ...] = DEFAULT_SCHEDULER_MILESTONES
    scheduler_gamma: float = 0.1
    scheduler_exp_gamma: float = 0.95
    scheduler_patience: int = 2
    scheduler_t_max: int | None = None
    scheduler_eta_min: float = 0.0
    scheduler_t_0: int | None = None
    scheduler_t_mult: int = 1
    scheduler_pct_start: float = 0.3
    scheduler_div_factor: float = 25.0
    scheduler_final_div_factor: float = 1e4
    seed: int = 3407
    runs: int = 1
    log_dir: Path = Path("logs/optuna_cnn")
    study_name: str | None = None
    storage: str | None = None
    timeout: int | None = None
    n_jobs: int = 1
    model_candidates: tuple[str, ...] = DEFAULT_TUNE_MODELS
    batch_candidates: tuple[int, ...] = DEFAULT_TUNE_BATCHES
    optimizer_candidates: tuple[str, ...] = DEFAULT_TUNE_OPTIMIZERS
    lr_low: float = 1e-5
    lr_high: float = 1e-3
    weight_decay_candidates: tuple[float, ...] = DEFAULT_TUNE_WEIGHT_DECAYS
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    restore_best: bool = True
    pruner_startup_trials: int = 5
    pruner_warmup_epochs: int = 2
    evaluate_best: bool = True


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got: {value}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got: {value}")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative float, got: {value}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive float, got: {value}")
    return parsed


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError(f"expected positive integers, got: {value}")
    return items


def _parse_resize_arg(value: str) -> tuple[int, int] | None:
    text = str(value).strip().lower()
    if text in ("none", "no", "false", "0", ""):
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise ValueError(f'--resize must be "256", "256,256", or "none": got={value}')


def _base_pil_mode_from_in_ch(in_ch: int) -> str:
    if in_ch == 1:
        return "L"
    if in_ch in (2, 3):
        return "RGB"
    return "RGBA"


def _match_channel_count(tensor: torch.Tensor, target_channels: int) -> torch.Tensor:
    current_channels = int(tensor.shape[0])
    if current_channels == target_channels:
        return tensor
    if current_channels > target_channels:
        return tensor[:target_channels, :, :]

    repeat_count = int(math.ceil(target_channels / current_channels))
    expanded = tensor.repeat(repeat_count, 1, 1)
    return expanded[:target_channels, :, :]


def _normalize_mean_std(in_ch: int) -> tuple[list[float], list[float]]:
    return [0.5] * in_ch, [0.5] * in_ch


def build_transform(in_ch: int, resize: tuple[int, int] | None) -> T.Compose:
    pil_mode = _base_pil_mode_from_in_ch(in_ch)
    mean, std = _normalize_mean_std(in_ch)

    transform_steps: list[Any] = [
        T.Lambda(lambda image: image.convert(pil_mode)),
    ]
    if resize is not None:
        transform_steps.append(T.Resize(resize))
    transform_steps.extend(
        [
            T.ToTensor(),
            T.Lambda(lambda tensor: _match_channel_count(tensor, in_ch)),
            T.Normalize(mean, std),
        ]
    )
    return T.Compose(transform_steps)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(base_seed: int):
    def init_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return init_worker


def _loader_kwargs(config: TrainEvalConfig, device: str, run_seed: int) -> dict[str, Any]:
    persistent_workers = config.workers > 0
    return {
        "num_workers": config.workers,
        "pin_memory": device == "cuda",
        "worker_init_fn": _worker_init_fn(run_seed) if config.workers > 0 else None,
        "persistent_workers": persistent_workers,
    }


def _build_train_val_dataloaders(
    config: TrainEvalConfig,
    device: str,
    run_seed: int,
    transform: T.Compose,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(run_seed)

    dev_dataset = ImageFolder(str(config.data_root / "dev"), transform=transform)

    train_len = int(len(dev_dataset) * 0.8)
    val_len = len(dev_dataset) - train_len
    train_dataset, val_dataset = random_split(dev_dataset, [train_len, val_len], generator=generator)

    loader_kwargs = _loader_kwargs(config, device, run_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def _build_test_dataloader(
    config: TrainEvalConfig,
    device: str,
    run_seed: int,
    transform: T.Compose,
) -> tuple[DataLoader, ImageFolder]:
    test_dataset = ImageFolder(str(config.data_root / "test"), transform=transform)
    loader_kwargs = _loader_kwargs(config, device, run_seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch,
        shuffle=False,
        **loader_kwargs,
    )

    return test_loader, test_dataset


def _build_dataloaders(
    config: TrainEvalConfig,
    device: str,
    run_seed: int,
    transform: T.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader, ImageFolder]:
    train_loader, val_loader = _build_train_val_dataloaders(config, device, run_seed, transform)
    test_loader, test_dataset = _build_test_dataloader(config, device, run_seed, transform)
    return train_loader, val_loader, test_loader, test_dataset


def _plot_histories(histories: list[list[float]], ylabel: str, title: str, out_path: Path) -> None:
    plt.figure()
    for idx, history in enumerate(histories, start=1):
        plt.plot(range(1, len(history) + 1), history, label=f"Run {idx}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _validate_lr_scheduler_config(config: Any) -> None:
    scheduler_name = config.lr_scheduler.lower()
    if scheduler_name not in SUPPORTED_LR_SCHEDULERS:
        raise ValueError(
            f"lr_scheduler must be one of {', '.join(SUPPORTED_LR_SCHEDULERS)}: "
            f"{config.lr_scheduler}"
        )
    if config.scheduler_step_size <= 0:
        raise ValueError(f"scheduler_step_size must be positive: {config.scheduler_step_size}")
    if not config.scheduler_milestones:
        raise ValueError("scheduler_milestones must not be empty")
    if any(milestone <= 0 for milestone in config.scheduler_milestones):
        raise ValueError(f"scheduler_milestones must be positive: {config.scheduler_milestones}")
    if config.scheduler_gamma <= 0:
        raise ValueError(f"scheduler_gamma must be positive: {config.scheduler_gamma}")
    if scheduler_name == "plateau" and config.scheduler_gamma >= 1.0:
        raise ValueError("scheduler_gamma must be < 1.0 when lr_scheduler is plateau")
    if config.scheduler_exp_gamma <= 0:
        raise ValueError(f"scheduler_exp_gamma must be positive: {config.scheduler_exp_gamma}")
    if config.scheduler_patience < 0:
        raise ValueError(f"scheduler_patience must be non-negative: {config.scheduler_patience}")
    if config.scheduler_t_max is not None and config.scheduler_t_max <= 0:
        raise ValueError(f"scheduler_t_max must be positive when set: {config.scheduler_t_max}")
    if config.scheduler_eta_min < 0:
        raise ValueError(f"scheduler_eta_min must be non-negative: {config.scheduler_eta_min}")
    if config.scheduler_t_0 is not None and config.scheduler_t_0 <= 0:
        raise ValueError(f"scheduler_t_0 must be positive when set: {config.scheduler_t_0}")
    if config.scheduler_t_mult <= 0:
        raise ValueError(f"scheduler_t_mult must be positive: {config.scheduler_t_mult}")
    if not 0.0 < config.scheduler_pct_start < 1.0:
        raise ValueError(f"scheduler_pct_start must be between 0 and 1: {config.scheduler_pct_start}")
    if config.scheduler_div_factor <= 0:
        raise ValueError(f"scheduler_div_factor must be positive: {config.scheduler_div_factor}")
    if config.scheduler_final_div_factor <= 0:
        raise ValueError(
            f"scheduler_final_div_factor must be positive: {config.scheduler_final_div_factor}"
        )


def _validate_train_eval_config(config: TrainEvalConfig) -> None:
    if config.workers < 0:
        raise ValueError(f"workers must be non-negative: {config.workers}")
    if config.in_ch <= 0:
        raise ValueError(f"in_ch must be positive: {config.in_ch}")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive: {config.epochs}")
    if config.batch <= 0:
        raise ValueError(f"batch must be positive: {config.batch}")
    if config.lr <= 0:
        raise ValueError(f"lr must be positive: {config.lr}")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative: {config.weight_decay}")
    if config.early_stopping_patience < 0:
        raise ValueError(
            f"early_stopping_patience must be non-negative: {config.early_stopping_patience}"
        )
    if config.early_stopping_min_delta < 0:
        raise ValueError(
            f"early_stopping_min_delta must be non-negative: {config.early_stopping_min_delta}"
        )
    if config.runs <= 0:
        raise ValueError(f"runs must be positive: {config.runs}")
    if config.optimizer.lower() not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {', '.join(SUPPORTED_OPTIMIZERS)}: {config.optimizer}"
        )
    _validate_lr_scheduler_config(config)


def _build_optimizer(
    name: str,
    parameters: Any,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_name = name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def _build_lr_scheduler(
    config: TrainEvalConfig,
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
) -> Any | None:
    scheduler_name = config.lr_scheduler.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if scheduler_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.scheduler_milestones),
            gamma=config.scheduler_gamma,
        )
    if scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler_exp_gamma,
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_t_max or config.epochs,
            eta_min=config.scheduler_eta_min,
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
        )
    if scheduler_name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler_t_0 or config.scheduler_step_size,
            T_mult=config.scheduler_t_mult,
            eta_min=config.scheduler_eta_min,
        )
    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.scheduler_pct_start,
            div_factor=config.scheduler_div_factor,
            final_div_factor=config.scheduler_final_div_factor,
        )
    raise ValueError(f"unsupported lr_scheduler: {config.lr_scheduler}")


def _is_batch_lr_scheduler(scheduler: Any | None) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)


def _step_lr_scheduler(scheduler: Any | None, val_acc: float) -> None:
    if scheduler is None:
        return
    if _is_batch_lr_scheduler(scheduler):
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_acc)
        return
    scheduler.step()


def _step_batch_lr_scheduler(scheduler: Any | None) -> None:
    if _is_batch_lr_scheduler(scheduler):
        scheduler.step()


def _report_trial(trial: Any, value: float, step: int) -> None:
    trial.report(value, step=step)
    if trial.should_prune():
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Optuna is required for trial pruning.") from exc
        raise optuna.TrialPruned(f"pruned at step {step} with val_acc={value:.4f}")


def _evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device, non_blocking=True)).argmax(1).cpu()
            correct += (predictions == targets).sum().item()
    return correct / len(loader.dataset)


def _snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_model(
    config: TrainEvalConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    *,
    run_idx: int,
    trial: Any | None = None,
    trial_step_offset: int = 0,
) -> dict[str, Any]:
    optimizer = _build_optimizer(
        config.optimizer,
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_lr_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    epoch_valacc: list[float] = []
    epoch_lrs: list[float] = []
    best_val_acc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    early_stopped = False
    stopped_epoch: int | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(
            train_loader,
            desc=f"Run {run_idx + 1}/{config.runs} Epoch {epoch:02d} [train]",
            leave=False,
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            _step_batch_lr_scheduler(scheduler)
            running_loss += loss.item() * targets.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if VERBOSE_EPOCH:
            print(f"  train loss = {epoch_loss:.4f}")

        val_acc = _evaluate_accuracy(model, val_loader, device)
        if VERBOSE_EPOCH:
            print(f"  val acc   = {val_acc:.3f}")

        epoch_lrs.append(float(optimizer.param_groups[0]["lr"]))
        epoch_losses.append(float(epoch_loss))
        epoch_valacc.append(float(val_acc))

        if trial is not None:
            _report_trial(trial, float(val_acc), trial_step_offset + epoch)

        improved = float(val_acc) > best_val_acc + config.early_stopping_min_delta
        if improved:
            best_val_acc = float(val_acc)
            best_epoch = epoch
            epochs_without_improvement = 0
            if config.restore_best:
                best_state = _snapshot_state_dict(model)
        else:
            epochs_without_improvement += 1

        _step_lr_scheduler(scheduler, float(val_acc))

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            early_stopped = True
            stopped_epoch = epoch
            break

    if config.restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_loss_per_epoch": epoch_losses,
        "val_acc_per_epoch": epoch_valacc,
        "lr_per_epoch": epoch_lrs,
        "best_val_acc": best_val_acc if best_val_acc != float("-inf") else 0.0,
        "best_epoch": best_epoch,
        "epochs_completed": len(epoch_losses),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
    }


def _predict_labels(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device, non_blocking=True)).argmax(1).cpu()
            y_true.extend(targets.tolist())
            y_pred.extend(predictions.tolist())
    return y_true, y_pred


def run_train_eval_mrun(config: TrainEvalConfig) -> dict[str, Any]:
    _validate_train_eval_config(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize_hw = _parse_resize_arg(config.resize)

    if resize_hw is None and config.batch > 1:
        print(
            "[WARN] --resize none and --batch>1 can fail when image sizes differ. "
            "Use --batch 1 or normalize image sizes ahead of time."
        )

    transform = build_transform(config.in_ch, resize_hw)

    log_root = config.log_dir
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = log_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_json_path = run_dir / "train_log.json"
    lr_png_path = run_dir / "lr_curves.png"
    loss_png_path = run_dir / "loss_curves.png"
    valacc_png_path = run_dir / "val_acc_curves.png"

    all_runs: list[dict[str, Any]] = []
    all_test_acc: list[float] = []
    all_test_f1m: list[float] = []
    all_test_f1_micro: list[float] = []
    all_test_f1_weighted: list[float] = []
    all_test_f1_bin_mal: list[float] = []
    all_test_f1_bin_ben: list[float] = []
    all_test_f1_per_class: list[list[float]] = []
    all_lr_histories: list[list[float]] = []
    all_loss_histories: list[list[float]] = []
    all_valacc_histories: list[list[float]] = []

    for run_idx in range(config.runs):
        run_seed = config.seed + run_idx
        set_seed(run_seed)

        train_loader, val_loader, test_loader, test_dataset = _build_dataloaders(
            config, device, run_seed, transform
        )

        model = get_model(
            config.model,
            num_classes=2,
            pretrained=config.pretrained,
            in_ch=config.in_ch,
        ).to(device)
        train_result = _train_model(
            config,
            model,
            train_loader,
            val_loader,
            device,
            run_idx=run_idx,
        )

        epoch_losses = train_result["train_loss_per_epoch"]
        epoch_valacc = train_result["val_acc_per_epoch"]
        epoch_lrs = train_result["lr_per_epoch"]
        best_val_acc = float(train_result["best_val_acc"])

        y_true, y_pred = _predict_labels(model, test_loader, device)

        test_acc = float(accuracy_score(y_true, y_pred))
        class_to_idx = getattr(test_dataset, "class_to_idx", {})
        benign_idx = class_to_idx.get("benign", 0)
        malware_idx = class_to_idx.get("malware", 1)
        labels_order = [benign_idx, malware_idx]

        test_f1_macro = float(f1_score(y_true, y_pred, average="macro"))
        test_f1_micro = float(f1_score(y_true, y_pred, average="micro"))
        test_f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
        test_f1_bin_mal = float(
            f1_score(y_true, y_pred, average="binary", pos_label=malware_idx)
        )
        test_f1_bin_ben = float(
            f1_score(y_true, y_pred, average="binary", pos_label=benign_idx)
        )
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels_order)
        test_f1_per_class = [float(per_class_f1[0]), float(per_class_f1[1])]

        classification = classification_report(
            y_true,
            y_pred,
            labels=labels_order,
            target_names=["benign", "malware"],
            digits=4,
        )
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels_order).tolist()

        print(
            f"[Run {run_idx + 1}/{config.runs}] seed={run_seed}  best_val_acc={best_val_acc:.3f}  "
            f"best_epoch={train_result['best_epoch']}  "
            f"test_acc={test_acc:.3f}  "
            f"F1(macro)={test_f1_macro:.3f}  F1(micro)={test_f1_micro:.3f}  "
            f"F1(weighted)={test_f1_weighted:.3f}  "
            f"F1(bin:mal)={test_f1_bin_mal:.3f}  F1(bin:ben)={test_f1_bin_ben:.3f}  "
            f"F1(per-class:ben,mal)={[round(x, 3) for x in test_f1_per_class]}"
        )

        run_log = {
            "run": run_idx + 1,
            "seed": run_seed,
            "epochs": config.epochs,
            "train_loss_per_epoch": epoch_losses,
            "val_acc_per_epoch": epoch_valacc,
            "lr_per_epoch": epoch_lrs,
            "best_val_acc": best_val_acc,
            "best_epoch": train_result["best_epoch"],
            "epochs_completed": train_result["epochs_completed"],
            "early_stopped": train_result["early_stopped"],
            "stopped_epoch": train_result["stopped_epoch"],
            "test_acc": test_acc,
            "test_f1": {
                "macro": test_f1_macro,
                "micro": test_f1_micro,
                "weighted": test_f1_weighted,
                "binary_malware": test_f1_bin_mal,
                "binary_benign": test_f1_bin_ben,
                "per_class": {
                    "benign": test_f1_per_class[0],
                    "malware": test_f1_per_class[1],
                },
            },
            "test_macro_f1": test_f1_macro,
            "classification_report": classification,
            "confusion_matrix": conf_mat,
            "class_to_idx": class_to_idx,
            "label_order": {"benign": benign_idx, "malware": malware_idx},
        }

        all_runs.append(run_log)
        all_test_acc.append(test_acc)
        all_test_f1m.append(test_f1_macro)
        all_test_f1_micro.append(test_f1_micro)
        all_test_f1_weighted.append(test_f1_weighted)
        all_test_f1_bin_mal.append(test_f1_bin_mal)
        all_test_f1_bin_ben.append(test_f1_bin_ben)
        all_test_f1_per_class.append(test_f1_per_class)
        all_lr_histories.append(epoch_lrs)
        all_loss_histories.append(epoch_losses)
        all_valacc_histories.append(epoch_valacc)

    summary = {
        "timestamp": timestamp,
        "args": {
            **asdict(config),
            "data_root": str(config.data_root),
            "log_dir": str(config.log_dir),
        },
        "device": device,
        "runs": all_runs,
        "summary": {
            "test_acc_mean": float(np.mean(all_test_acc)) if all_test_acc else None,
            "test_acc_std": float(np.std(all_test_acc, ddof=0)) if all_test_acc else None,
            "macro_f1_mean": float(np.mean(all_test_f1m)) if all_test_f1m else None,
            "macro_f1_std": float(np.std(all_test_f1m, ddof=0)) if all_test_f1m else None,
            "micro_f1_mean": float(np.mean(all_test_f1_micro)) if all_test_f1_micro else None,
            "micro_f1_std": float(np.std(all_test_f1_micro, ddof=0)) if all_test_f1_micro else None,
            "weighted_f1_mean": (
                float(np.mean(all_test_f1_weighted)) if all_test_f1_weighted else None
            ),
            "weighted_f1_std": (
                float(np.std(all_test_f1_weighted, ddof=0)) if all_test_f1_weighted else None
            ),
            "binary_malware_f1_mean": (
                float(np.mean(all_test_f1_bin_mal)) if all_test_f1_bin_mal else None
            ),
            "binary_malware_f1_std": (
                float(np.std(all_test_f1_bin_mal, ddof=0)) if all_test_f1_bin_mal else None
            ),
            "binary_benign_f1_mean": (
                float(np.mean(all_test_f1_bin_ben)) if all_test_f1_bin_ben else None
            ),
            "binary_benign_f1_std": (
                float(np.std(all_test_f1_bin_ben, ddof=0)) if all_test_f1_bin_ben else None
            ),
            "per_class_f1_mean": (
                np.array(all_test_f1_per_class, dtype=float).mean(axis=0).tolist()
                if all_test_f1_per_class
                else None
            ),
            "per_class_f1_std": (
                np.array(all_test_f1_per_class, dtype=float).std(axis=0, ddof=0).tolist()
                if all_test_f1_per_class
                else None
            ),
            "per_class_f1_order": ["benign", "malware"],
        },
    }

    if all_runs:
        mean_acc = float(np.mean(all_test_acc))
        std_acc = float(np.std(all_test_acc, ddof=0))
        mean_f1m = float(np.mean(all_test_f1m))
        std_f1m = float(np.std(all_test_f1m, ddof=0))
        mean_f1_micro = float(np.mean(all_test_f1_micro))
        std_f1_micro = float(np.std(all_test_f1_micro, ddof=0))
        mean_f1_weighted = float(np.mean(all_test_f1_weighted))
        std_f1_weighted = float(np.std(all_test_f1_weighted, ddof=0))
        mean_f1_bin_mal = float(np.mean(all_test_f1_bin_mal))
        std_f1_bin_mal = float(np.std(all_test_f1_bin_mal, ddof=0))
        mean_f1_bin_ben = float(np.mean(all_test_f1_bin_ben))
        std_f1_bin_ben = float(np.std(all_test_f1_bin_ben, ddof=0))
        per_class_arr = np.array(all_test_f1_per_class, dtype=float)
        mean_f1_pc = per_class_arr.mean(axis=0).tolist()
        std_f1_pc = per_class_arr.std(axis=0, ddof=0).tolist()

        print(f"\n=== SUMMARY over {config.runs} runs ===")
        print(f"Test Accuracy: mean={mean_acc:.4f}  std={std_acc:.4f}")
        print(f"F1(macro)   : mean={mean_f1m:.4f}  std={std_f1m:.4f}")
        print(f"F1(micro)   : mean={mean_f1_micro:.4f}  std={std_f1_micro:.4f}")
        print(f"F1(weighted): mean={mean_f1_weighted:.4f}  std={std_f1_weighted:.4f}")
        print(f"F1(binary mal): mean={mean_f1_bin_mal:.4f}  std={std_f1_bin_mal:.4f}")
        print(f"F1(binary ben): mean={mean_f1_bin_ben:.4f}  std={std_f1_bin_ben:.4f}")
        print(
            "F1(per-class ben,mal): "
            f"mean={[round(x, 4) for x in mean_f1_pc]}  std={[round(x, 4) for x in std_f1_pc]}"
        )

    with log_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    _plot_histories(all_lr_histories, "Learning Rate", "LR per Epoch (all runs)", lr_png_path)
    _plot_histories(all_loss_histories, "Train Loss", "Train Loss per Epoch (all runs)", loss_png_path)
    _plot_histories(
        all_valacc_histories,
        "Val Accuracy",
        "Val Accuracy per Epoch (all runs)",
        valacc_png_path,
    )

    return {
        "summary": summary,
        "run_dir": run_dir,
        "log_json_path": log_json_path,
        "lr_png_path": lr_png_path,
        "loss_png_path": loss_png_path,
        "valacc_png_path": valacc_png_path,
    }


def _serializable_dataclass(config: TrainEvalConfig | TuneConfig) -> dict[str, Any]:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def _validate_tune_config(config: TuneConfig) -> None:
    if config.trials <= 0:
        raise ValueError(f"trials must be positive: {config.trials}")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive: {config.epochs}")
    if config.workers < 0:
        raise ValueError(f"workers must be non-negative: {config.workers}")
    if config.in_ch <= 0:
        raise ValueError(f"in_ch must be positive: {config.in_ch}")
    if config.runs <= 0:
        raise ValueError(f"runs must be positive: {config.runs}")
    _validate_lr_scheduler_config(config)
    if not config.model_candidates:
        raise ValueError("model_candidates must not be empty")
    if not config.batch_candidates:
        raise ValueError("batch_candidates must not be empty")
    if not config.optimizer_candidates:
        raise ValueError("optimizer_candidates must not be empty")
    if not config.weight_decay_candidates:
        raise ValueError("weight_decay_candidates must not be empty")
    if config.lr_low <= 0 or config.lr_high <= 0:
        raise ValueError("lr bounds must be positive")
    if config.lr_low > config.lr_high:
        raise ValueError(f"lr_low must be <= lr_high: {config.lr_low} > {config.lr_high}")
    if any(batch <= 0 for batch in config.batch_candidates):
        raise ValueError(f"batch candidates must be positive: {config.batch_candidates}")
    if any(weight_decay < 0 for weight_decay in config.weight_decay_candidates):
        raise ValueError(
            f"weight_decay candidates must be non-negative: {config.weight_decay_candidates}"
        )
    unknown_optimizers = [
        optimizer for optimizer in config.optimizer_candidates if optimizer.lower() not in SUPPORTED_OPTIMIZERS
    ]
    if unknown_optimizers:
        raise ValueError(
            f"optimizer candidates must be in {', '.join(SUPPORTED_OPTIMIZERS)}: "
            f"{unknown_optimizers}"
        )
    if config.early_stopping_patience < 0:
        raise ValueError(
            f"early_stopping_patience must be non-negative: {config.early_stopping_patience}"
        )
    if config.early_stopping_min_delta < 0:
        raise ValueError(
            f"early_stopping_min_delta must be non-negative: {config.early_stopping_min_delta}"
        )
    if config.pruner_startup_trials < 0:
        raise ValueError(f"pruner_startup_trials must be non-negative: {config.pruner_startup_trials}")
    if config.pruner_warmup_epochs < 0:
        raise ValueError(f"pruner_warmup_epochs must be non-negative: {config.pruner_warmup_epochs}")


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Optuna is required for tune-cnn. Install the CNN extras, e.g. pip install -e ".[cnn]".'
        ) from exc
    return optuna


def _run_validation_mrun(config: TrainEvalConfig, *, trial: Any | None = None) -> dict[str, Any]:
    _validate_train_eval_config(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize_hw = _parse_resize_arg(config.resize)

    if resize_hw is None and config.batch > 1:
        print(
            "[WARN] --resize none and --batch>1 can fail when image sizes differ. "
            "Use --batch 1 or normalize image sizes ahead of time."
        )

    transform = build_transform(config.in_ch, resize_hw)
    all_runs: list[dict[str, Any]] = []
    best_val_scores: list[float] = []

    for run_idx in range(config.runs):
        run_seed = config.seed + run_idx
        set_seed(run_seed)

        train_loader, val_loader = _build_train_val_dataloaders(config, device, run_seed, transform)

        model = get_model(
            config.model,
            num_classes=2,
            pretrained=config.pretrained,
            in_ch=config.in_ch,
        ).to(device)
        train_result = _train_model(
            config,
            model,
            train_loader,
            val_loader,
            device,
            run_idx=run_idx,
            trial=trial,
            trial_step_offset=run_idx * config.epochs,
        )

        best_val_acc = float(train_result["best_val_acc"])
        best_val_scores.append(best_val_acc)
        all_runs.append(
            {
                "run": run_idx + 1,
                "seed": run_seed,
                **train_result,
            }
        )

    return {
        "config": _serializable_dataclass(config),
        "device": device,
        "runs": all_runs,
        "score": float(np.mean(best_val_scores)) if best_val_scores else 0.0,
        "score_std": float(np.std(best_val_scores, ddof=0)) if best_val_scores else 0.0,
    }


def _suggest_train_eval_config(config: TuneConfig, trial: Any) -> TrainEvalConfig:
    return TrainEvalConfig(
        data_root=config.data_root,
        model=trial.suggest_categorical("model", list(config.model_candidates)),
        pretrained=config.pretrained,
        epochs=config.epochs,
        batch=trial.suggest_categorical("batch", list(config.batch_candidates)),
        lr=trial.suggest_float("lr", config.lr_low, config.lr_high, log=True),
        optimizer=trial.suggest_categorical("optimizer", list(config.optimizer_candidates)),
        weight_decay=trial.suggest_categorical(
            "weight_decay",
            list(config.weight_decay_candidates),
        ),
        workers=config.workers,
        in_ch=config.in_ch,
        resize=config.resize,
        lr_scheduler=config.lr_scheduler,
        scheduler_step_size=config.scheduler_step_size,
        scheduler_milestones=config.scheduler_milestones,
        scheduler_gamma=config.scheduler_gamma,
        scheduler_exp_gamma=config.scheduler_exp_gamma,
        scheduler_patience=config.scheduler_patience,
        scheduler_t_max=config.scheduler_t_max,
        scheduler_eta_min=config.scheduler_eta_min,
        scheduler_t_0=config.scheduler_t_0,
        scheduler_t_mult=config.scheduler_t_mult,
        scheduler_pct_start=config.scheduler_pct_start,
        scheduler_div_factor=config.scheduler_div_factor,
        scheduler_final_div_factor=config.scheduler_final_div_factor,
        seed=config.seed,
        runs=config.runs,
        log_dir=config.log_dir,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        restore_best=config.restore_best,
    )


def _train_eval_config_from_params(
    config: TuneConfig,
    params: dict[str, Any],
    *,
    log_dir: Path,
) -> TrainEvalConfig:
    return TrainEvalConfig(
        data_root=config.data_root,
        model=str(params["model"]),
        pretrained=config.pretrained,
        epochs=config.epochs,
        batch=int(params["batch"]),
        lr=float(params["lr"]),
        optimizer=str(params["optimizer"]),
        weight_decay=float(params["weight_decay"]),
        workers=config.workers,
        in_ch=config.in_ch,
        resize=config.resize,
        lr_scheduler=config.lr_scheduler,
        scheduler_step_size=config.scheduler_step_size,
        scheduler_milestones=config.scheduler_milestones,
        scheduler_gamma=config.scheduler_gamma,
        scheduler_exp_gamma=config.scheduler_exp_gamma,
        scheduler_patience=config.scheduler_patience,
        scheduler_t_max=config.scheduler_t_max,
        scheduler_eta_min=config.scheduler_eta_min,
        scheduler_t_0=config.scheduler_t_0,
        scheduler_t_mult=config.scheduler_t_mult,
        scheduler_pct_start=config.scheduler_pct_start,
        scheduler_div_factor=config.scheduler_div_factor,
        scheduler_final_div_factor=config.scheduler_final_div_factor,
        seed=config.seed,
        runs=config.runs,
        log_dir=log_dir,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        restore_best=config.restore_best,
    )


def _serialize_optuna_trial(trial: Any) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
    }


def run_tune_cnn(config: TuneConfig) -> dict[str, Any]:
    _validate_tune_config(config)
    optuna = _import_optuna()

    log_root = config.log_dir
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tune_dir = log_root / timestamp
    tune_dir.mkdir(parents=True, exist_ok=True)
    log_json_path = tune_dir / "optuna_tuning_log.json"

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.pruner_startup_trials,
        n_warmup_steps=config.pruner_warmup_epochs,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=config.storage is not None,
    )

    def objective(trial: Any) -> float:
        train_config = _suggest_train_eval_config(config, trial)
        trial.set_user_attr("train_config", _serializable_dataclass(train_config))
        validation_result = _run_validation_mrun(train_config, trial=trial)
        trial.set_user_attr("validation", validation_result)
        return float(validation_result["score"])

    study.optimize(
        objective,
        n_trials=config.trials,
        timeout=config.timeout,
        n_jobs=config.n_jobs,
        show_progress_bar=True,
    )

    best_trial = None
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "args": _serializable_dataclass(config),
        "direction": "maximize",
        "objective": "mean best validation accuracy across runs",
        "study_name": study.study_name,
        "best_trial": _serialize_optuna_trial(best_trial) if best_trial is not None else None,
        "trials": [_serialize_optuna_trial(trial) for trial in study.trials],
    }
    with log_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    best_eval_result: dict[str, Any] | None = None
    if best_trial is not None and config.evaluate_best:
        best_config = _train_eval_config_from_params(
            config,
            best_trial.params,
            log_dir=tune_dir / "best_eval",
        )
        best_eval_result = run_train_eval_mrun(best_config)

    if best_eval_result is not None:
        summary["best_eval"] = {
            "run_dir": str(best_eval_result["run_dir"]),
            "log_json_path": str(best_eval_result["log_json_path"]),
            "summary": best_eval_result["summary"]["summary"],
        }
        with log_json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    if best_trial is not None:
        print(
            f"Best trial #{best_trial.number}: val_acc={best_trial.value:.4f}  "
            f"params={best_trial.params}"
        )

    return {
        "summary": summary,
        "study": study,
        "tune_dir": tune_dir,
        "log_json_path": log_json_path,
        "best_eval_result": best_eval_result,
    }


def _safe_study_suffix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def run_tune_cnn_by_model(config: TuneConfig) -> dict[str, Any]:
    """Run an independent Optuna study for each requested model candidate."""
    _validate_tune_config(config)

    log_root = config.log_dir
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tune_dir = log_root / f"{timestamp}_by_model"
    tune_dir.mkdir(parents=True, exist_ok=True)
    log_json_path = tune_dir / "per_model_tuning_log.json"

    model_results: list[dict[str, Any]] = []
    best_model: dict[str, Any] | None = None

    for index, model_name in enumerate(config.model_candidates, start=1):
        safe_model = _safe_study_suffix(model_name)
        print(f"\n=== Tuning model {index}/{len(config.model_candidates)}: {model_name} ===")

        study_name = None
        if config.study_name is not None:
            study_name = f"{config.study_name}_{safe_model}"

        model_config = replace(
            config,
            model_candidates=(model_name,),
            log_dir=tune_dir / safe_model,
            study_name=study_name,
        )
        result = run_tune_cnn(model_config)
        summary = result["summary"]
        best_trial = summary.get("best_trial")
        best_value = None if best_trial is None else best_trial.get("value")

        model_summary = {
            "model": model_name,
            "tune_dir": str(result["tune_dir"]),
            "log_json_path": str(result["log_json_path"]),
            "best_value": best_value,
            "best_trial": best_trial,
            "best_eval": summary.get("best_eval"),
        }
        model_results.append(model_summary)

        if best_value is not None and (
            best_model is None or float(best_value) > float(best_model["best_value"])
        ):
            best_model = model_summary

    aggregate_summary = {
        "timestamp": timestamp,
        "args": _serializable_dataclass(config),
        "mode": "per_model",
        "objective": "best validation accuracy per independent model study",
        "models": model_results,
        "best_model": best_model,
    }
    with log_json_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate_summary, handle, ensure_ascii=False, indent=2)

    if best_model is not None:
        print(
            f"\nBest model: {best_model['model']}  "
            f"val_acc={float(best_model['best_value']):.4f}"
        )
    print(f"Saved per-model Optuna summary: {log_json_path}")

    return {
        "summary": aggregate_summary,
        "tune_dir": tune_dir,
        "log_json_path": log_json_path,
        "model_results": model_results,
        "best_model": best_model,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path, help="parent directory containing dev/ and test/")
    parser.add_argument(
        "--model",
        default="resnet50",
        help=MODEL_HELP,
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use torchvision ImageNet pretrained weights when available",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=_positive_float, default=1e-4)
    parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, default="adam")
    parser.add_argument("--weight-decay", type=_non_negative_float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=SUPPORTED_LR_SCHEDULERS, default="none")
    parser.add_argument("--scheduler-step-size", type=_positive_int, default=5)
    parser.add_argument(
        "--scheduler-milestones",
        type=_csv_positive_ints,
        default=DEFAULT_SCHEDULER_MILESTONES,
    )
    parser.add_argument("--scheduler-gamma", type=_positive_float, default=0.1)
    parser.add_argument("--scheduler-exp-gamma", type=_positive_float, default=0.95)
    parser.add_argument("--scheduler-patience", type=_non_negative_int, default=2)
    parser.add_argument("--scheduler-t-max", type=_positive_int)
    parser.add_argument("--scheduler-eta-min", type=_non_negative_float, default=0.0)
    parser.add_argument("--scheduler-t-0", type=_positive_int)
    parser.add_argument("--scheduler-t-mult", type=_positive_int, default=1)
    parser.add_argument("--scheduler-pct-start", type=_positive_float, default=0.3)
    parser.add_argument("--scheduler-div-factor", type=_positive_float, default=25.0)
    parser.add_argument("--scheduler-final-div-factor", type=_positive_float, default=1e4)
    parser.add_argument("--workers", type=_non_negative_int, default=4)
    parser.add_argument("--in-ch", type=_positive_int, default=1, help="input channel count")
    parser.add_argument(
        "--resize",
        type=str,
        default="256,256",
        help='resize spec: "256", "256,256", or "none"',
    )
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--runs", type=int, default=1, help="number of repeated runs")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="directory for logs and plots")
    parser.add_argument("--early-stopping-patience", type=_non_negative_int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=_non_negative_float, default=0.0)
    parser.add_argument("--restore-best", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainEvalConfig(
        data_root=args.data_root,
        model=args.model,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_milestones=args.scheduler_milestones,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_exp_gamma=args.scheduler_exp_gamma,
        scheduler_patience=args.scheduler_patience,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_eta_min=args.scheduler_eta_min,
        scheduler_t_0=args.scheduler_t_0,
        scheduler_t_mult=args.scheduler_t_mult,
        scheduler_pct_start=args.scheduler_pct_start,
        scheduler_div_factor=args.scheduler_div_factor,
        scheduler_final_div_factor=args.scheduler_final_div_factor,
        workers=args.workers,
        in_ch=args.in_ch,
        resize=args.resize,
        seed=args.seed,
        runs=args.runs,
        log_dir=args.log_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best=args.restore_best,
    )
    result = run_train_eval_mrun(config)
    print(f"Saved logs under: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
