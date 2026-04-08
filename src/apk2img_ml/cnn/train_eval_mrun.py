from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class TrainEvalConfig:
    data_root: Path
    model: str = "resnet50"
    epochs: int = 15
    batch: int = 32
    workers: int = 4
    in_ch: int = 1
    resize: str = "256,256"
    seed: int = 3407
    runs: int = 1
    log_dir: Path = Path("logs")


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


def _build_dataloaders(
    config: TrainEvalConfig,
    device: str,
    run_seed: int,
    transform: T.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader, ImageFolder]:
    generator = torch.Generator()
    generator.manual_seed(run_seed)

    dev_dataset = ImageFolder(str(config.data_root / "dev"), transform=transform)
    test_dataset = ImageFolder(str(config.data_root / "test"), transform=transform)

    train_len = int(len(dev_dataset) * 0.8)
    val_len = len(dev_dataset) - train_len
    train_dataset, val_dataset = random_split(dev_dataset, [train_len, val_len], generator=generator)

    persistent_workers = config.workers > 0
    loader_kwargs = {
        "num_workers": config.workers,
        "pin_memory": device == "cuda",
        "worker_init_fn": _worker_init_fn(run_seed) if config.workers > 0 else None,
        "persistent_workers": persistent_workers,
    }

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch,
        shuffle=False,
        **loader_kwargs,
    )

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


def run_train_eval_mrun(config: TrainEvalConfig) -> dict[str, Any]:
    if config.workers < 0:
        raise ValueError(f"workers must be non-negative: {config.workers}")
    if config.in_ch <= 0:
        raise ValueError(f"in_ch must be positive: {config.in_ch}")

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
            pretrained=True,
            in_ch=config.in_ch,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        epoch_losses: list[float] = []
        epoch_valacc: list[float] = []
        epoch_lrs: list[float] = []
        best_val_acc = 0.0

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
                running_loss += loss.item() * targets.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            if VERBOSE_EPOCH:
                print(f"  train loss = {epoch_loss:.4f}")

            model.eval()
            correct = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    predictions = model(inputs.to(device, non_blocking=True)).argmax(1).cpu()
                    correct += (predictions == targets).sum().item()
            val_acc = correct / len(val_loader.dataset)
            if VERBOSE_EPOCH:
                print(f"  val acc   = {val_acc:.3f}")

            epoch_lrs.append(float(optimizer.param_groups[0]["lr"]))
            epoch_losses.append(float(epoch_loss))
            epoch_valacc.append(float(val_acc))
            best_val_acc = max(best_val_acc, float(val_acc))

        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                predictions = model(inputs.to(device, non_blocking=True)).argmax(1).cpu()
                y_true.extend(targets.tolist())
                y_pred.extend(predictions.tolist())

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path, help="parent directory containing dev/ and test/")
    parser.add_argument(
        "--model",
        default="resnet50",
        help=MODEL_HELP,
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainEvalConfig(
        data_root=args.data_root,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        in_ch=args.in_ch,
        resize=args.resize,
        seed=args.seed,
        runs=args.runs,
        log_dir=args.log_dir,
    )
    result = run_train_eval_mrun(config)
    print(f"Saved logs under: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
