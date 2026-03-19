#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval.py
-------------
  * images256/dev  → train:val=8:2
  * images256/test → 最終評価
  * モデル選択: tiny | alexnet | vgg16 | resnet50 | densenet | mobilenet
  * 任意チャンネル(1/3/4/…)入力対応
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse, torch, torchvision.transforms as T
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from models_zoo import get_model  # 同ディレクトリ

# 追記: 再現性/多回実行/ログ/描画
import random, json
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import japanize_matplotlib

# ──────────── 1. 引数 ────────────
pa = argparse.ArgumentParser()
pa.add_argument('--data-root', required=True, help='images256 親ディレクトリ')
pa.add_argument('--model', default='resnet50', help='tiny|alexnet|vgg16|resnet50|densenet|mobilenet')
pa.add_argument('--epochs', type=int, default=15)
pa.add_argument('--batch',  type=int, default=32)
pa.add_argument('--workers',type=int, default=4)
pa.add_argument('--in-ch',  type=int, default=1, help='入力チャンネル数（1,3,4など）')

# 追記: リサイズ指定（互換維持: デフォルトは 256×256）
pa.add_argument(
    '--resize', type=str, default='256,256',
    help='Resize指定。例: "256" or "256,256" or "320,256" / 無効化は "none"'
)

# 追記: シード/複数実行/ログ
pa.add_argument('--seed',   type=int, default=3407, help='乱数シード')
pa.add_argument('--runs',   type=int, default=1,    help='同条件での反復実行回数')
pa.add_argument('--log-dir', type=str, default='logs', help='ログ・図の保存先ディレクトリ')

args = pa.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 追記: シード固定ユーティリティ
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 決定論優先（非決定的Opで例外が出る場合あり）
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ──────────── 2. DataLoader ────────────
def _mode_from_in_ch(in_ch:int) -> str:
    if in_ch == 1:  return 'L'     # 1ch (grayscale)
    if in_ch == 3:  return 'RGB'   # 3ch
    if in_ch == 4:  return 'RGBA'  # 4ch (RGB + alpha)
    # それ以上はPILでの直接変換がないので、RGB(A)を基準に後段で拡張するのが安全
    # ここでは簡便にRGBAまでをサポート
    raise ValueError(f'in-ch={in_ch} は未対応。1/3/4を指定してください。')

def _normalize_mean_std(in_ch:int):
    # すべて0.5/0.5に統一（必要ならデータセットに合わせて変更）
    return [0.5]*in_ch, [0.5]*in_ch

# 追記: --resize のパース（変更点）
def _parse_resize_arg(s: str):
    s = str(s).strip().lower()
    if s in ('none', 'no', 'false', '0', ''):
        return None
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if len(parts) == 1:
        v = int(parts[0])
        return (v, v)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise ValueError(f'--resize は "256" / "256,256" / "none" の形式で指定してください: got={s}')

pil_mode = _mode_from_in_ch(args.in_ch)
mean, std = _normalize_mean_std(args.in_ch)

resize_hw = _parse_resize_arg(args.resize)  # 追記: (H,W) or None（変更点）

# 追記: Resizeなし & batch>1 は DataLoader で形状不一致になりやすいので警告（変更点）
if resize_hw is None and args.batch > 1:
    print("[WARN] --resize none かつ --batch>1 です。画像サイズが混在するとバッチ化でエラーになります。"
          "（対策: --batch 1 / 事前にサイズ統一 / crop/pad を追加）")

# 追記: Resizeを条件付きで入れる（変更点）
tfm_list = [
    T.Lambda(lambda im: im.convert(pil_mode)),  # 任意チャンネル化（1/3/4）
]
if resize_hw is not None:
    tfm_list.append(T.Resize(resize_hw))  # 追記: 任意サイズ
tfm_list += [
    T.ToTensor(),
    T.Normalize(mean, std),
]
tfm = T.Compose(tfm_list)

# 追記: ログ準備（時刻付きディレクトリ／ファイル名）
log_root = Path(args.log_dir)
log_root.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = log_root / ts                  # ← 実行日時ディレクトリ
run_dir.mkdir(parents=True, exist_ok=True)
log_json_path   = run_dir / "train_log.json"
lr_png_path     = run_dir / "lr_curves.png"
loss_png_path   = run_dir / "loss_curves.png"       # 追記
valacc_png_path = run_dir / "val_acc_curves.png"    # 追記

# 追記: 既存のエポック出力を抑制するフラグ（標準出力を簡潔に）
VERBOSE_EPOCH = False  # 既存printをこのフラグで抑制

# 追記: ループ横断の記録
all_runs = []
all_test_acc = []
all_test_f1m = []  # macroF1 互換のため残す（変更しない）
# 追記: 追加のF1記録（変更点）
all_test_f1_micro = []       # 追記
all_test_f1_weighted = []    # 追記
all_test_f1_bin_mal = []     # 追記
all_test_f1_bin_ben = []     # 追記
all_test_f1_per_class = []   # 追記: [f1_benign, f1_malware]

all_lr_histories = []
all_loss_histories = []     # 追記
all_valacc_histories = []   # 追記

# 追記: 複数回実行ループ
base_seed = args.seed
for run_idx in range(args.runs):
    run_seed = base_seed + run_idx
    set_seed(run_seed)
    g = torch.Generator()
    g.manual_seed(run_seed)

    dev_ds  = ImageFolder(f"{args.data_root}/dev",  transform=tfm)
    test_ds = ImageFolder(f"{args.data_root}/test", transform=tfm)

    train_len = int(len(dev_ds)*0.8)
    val_len   = len(dev_ds)-train_len
    # 最小置換: generatorを付与（分割順固定）
    train_ds, val_ds = random_split(dev_ds, [train_len,val_len], generator=g)

    pin = (device == 'cuda')

    # 各ワーカーの乱数初期化
    def _worker_init_fn(worker_id: int):
        worker_seed = run_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_ld = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=pin,
        generator=g, worker_init_fn=_worker_init_fn, persistent_workers=True
    )
    val_ld   = DataLoader(
        val_ds,   batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        worker_init_fn=_worker_init_fn, persistent_workers=True
    )
    test_ld  = DataLoader(
        test_ds,  batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        worker_init_fn=_worker_init_fn, persistent_workers=True
    )

    # ──────────── 3. モデル・Optimizer ────────────
    model = get_model(args.model, num_classes=2, pretrained=True, in_ch=args.in_ch).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit  = torch.nn.CrossEntropyLoss()

    # 追記: 記録用
    epoch_losses = []
    epoch_valacc = []
    epoch_lrs    = []

    # ──────────── 4. 学習ループ ────────────
    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        # --- train ---
        model.train(); running=0.0
        for x,y in tqdm(train_ld, desc=f"Run {run_idx+1}/{args.runs} Epoch {epoch:02d} [train]", leave=False):
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            running += loss.item()*y.size(0)
        epoch_loss = running/len(train_ld.dataset)
        if VERBOSE_EPOCH:
            print(f"  train loss = {epoch_loss:.4f}")

        # --- val ---
        model.eval(); correct=0
        with torch.no_grad():
            for x,y in val_ld:
                pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()
                correct += (pred==y).sum().item()
        val_acc = correct/len(val_ld.dataset)
        if VERBOSE_EPOCH:
            print(f"  val acc   = {val_acc:.3f}")

        # 学習率の記録（スケジューラ未使用でもフラットでプロット）
        epoch_lrs.append(opt.param_groups[0]['lr'])
        epoch_losses.append(epoch_loss)
        epoch_valacc.append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

    # ──────────── 5. テスト評価 ────────────
    model.eval(); y_true, y_pred = [],[]
    with torch.no_grad():
        for x,y in test_ld:
            p = model(x.to(device, non_blocking=True)).argmax(1).cpu()
            y_true.extend(y.tolist()); y_pred.extend(p.tolist())

    # 要約値（標準出力用）
    from sklearn.metrics import accuracy_score, f1_score
    test_acc  = float(accuracy_score(y_true, y_pred))

    # 追記: F1を全種類算出（変更点）
    #  - ImageFolderのラベルIDを確実に取得（benign/malware の pos_label を安定化）
    class_to_idx = getattr(test_ds, "class_to_idx", {})
    benign_idx  = class_to_idx.get("benign", 0)   # 追記
    malware_idx = class_to_idx.get("malware", 1)  # 追記
    labels_order = [benign_idx, malware_idx]      # 追記: per-class/レポートの順序固定

    test_f1_macro    = float(f1_score(y_true, y_pred, average='macro'))      # 追記: 既存macro（名称を明確化）
    test_f1_micro    = float(f1_score(y_true, y_pred, average='micro'))      # 追記
    test_f1_weighted = float(f1_score(y_true, y_pred, average='weighted'))   # 追記
    # 追記: binary F1（陽性クラス指定）
    test_f1_bin_mal  = float(f1_score(y_true, y_pred, average='binary', pos_label=malware_idx))
    test_f1_bin_ben  = float(f1_score(y_true, y_pred, average='binary', pos_label=benign_idx))
    # 追記: per-class F1（average=None）
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels_order)
    test_f1_per_class = [float(per_class_f1[0]), float(per_class_f1[1])]  # [benign, malware]

    # 詳細（ログ用）
    # 追記: labels を明示して target_names と対応づけ（変更点）
    cls_report = classification_report(
        y_true, y_pred,
        labels=labels_order,                    # 追記
        target_names=['benign','malware'],
        digits=4
    )
    conf_mat   = confusion_matrix(y_true, y_pred, labels=labels_order).tolist()  # 追記: 順序固定（変更点）

    # 標準出力（各Run要約のみ）
    # 追記: 出力項目を増やす（変更点）
    print(
        f"[Run {run_idx+1}/{args.runs}] seed={run_seed}  best_val_acc={best_val_acc:.3f}  "
        f"test_acc={test_acc:.3f}  "
        f"F1(macro)={test_f1_macro:.3f}  F1(micro)={test_f1_micro:.3f}  F1(weighted)={test_f1_weighted:.3f}  "
        f"F1(bin:mal)={test_f1_bin_mal:.3f}  F1(bin:ben)={test_f1_bin_ben:.3f}  "
        f"F1(per-class:ben,mal)={[round(x,3) for x in test_f1_per_class]}"
    )

    # 走行ログ集約
    run_log = {
        "run": run_idx+1,
        "seed": run_seed,
        "epochs": args.epochs,
        "train_loss_per_epoch": epoch_losses,
        "val_acc_per_epoch": epoch_valacc,
        "lr_per_epoch": epoch_lrs,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        # 追記: F1を全種類保存（変更点）
        "test_f1": {
            "macro": test_f1_macro,
            "micro": test_f1_micro,
            "weighted": test_f1_weighted,
            "binary_malware": test_f1_bin_mal,
            "binary_benign": test_f1_bin_ben,
            "per_class": {
                "benign": test_f1_per_class[0],
                "malware": test_f1_per_class[1],
            }
        },
        # 互換: 既存キーも残す（変更点: 追記）
        "test_macro_f1": test_f1_macro,
        "classification_report": cls_report,
        "confusion_matrix": conf_mat,
        # 追記: ラベル対応をログに残す（変更点）
        "class_to_idx": class_to_idx,
        "label_order": {"benign": benign_idx, "malware": malware_idx},
    }
    all_runs.append(run_log)
    all_test_acc.append(test_acc)

    # 互換: 既存macro配列は維持（変更点: 追記）
    all_test_f1m.append(test_f1_macro)

    # 追記: 新しいF1配列
    all_test_f1_micro.append(test_f1_micro)
    all_test_f1_weighted.append(test_f1_weighted)
    all_test_f1_bin_mal.append(test_f1_bin_mal)
    all_test_f1_bin_ben.append(test_f1_bin_ben)
    all_test_f1_per_class.append(test_f1_per_class)

    all_lr_histories.append(epoch_lrs)
    all_loss_histories.append(epoch_losses)       # 追記
    all_valacc_histories.append(epoch_valacc)    # 追記

# 追記: 全Runの要約（平均）
if len(all_runs) > 0:
    mean_acc = float(np.mean(all_test_acc))
    std_acc  = float(np.std(all_test_acc, ddof=0))

    # 互換: 既存macroF1の要約は維持（変更点: 追記）
    mean_f1m  = float(np.mean(all_test_f1m))
    std_f1m   = float(np.std(all_test_f1m, ddof=0))

    # 追記: 追加F1の要約（変更点）
    mean_f1_micro    = float(np.mean(all_test_f1_micro))
    std_f1_micro     = float(np.std(all_test_f1_micro, ddof=0))
    mean_f1_weighted = float(np.mean(all_test_f1_weighted))
    std_f1_weighted  = float(np.std(all_test_f1_weighted, ddof=0))
    mean_f1_bin_mal  = float(np.mean(all_test_f1_bin_mal))
    std_f1_bin_mal   = float(np.std(all_test_f1_bin_mal, ddof=0))
    mean_f1_bin_ben  = float(np.mean(all_test_f1_bin_ben))
    std_f1_bin_ben   = float(np.std(all_test_f1_bin_ben, ddof=0))

    # per-class は [benign, malware] の2要素を列ごとに平均（追記）
    per_class_arr = np.array(all_test_f1_per_class, dtype=float)  # shape: (runs, 2)
    mean_f1_pc = per_class_arr.mean(axis=0).tolist()
    std_f1_pc  = per_class_arr.std(axis=0, ddof=0).tolist()

    print(f"\n=== SUMMARY over {args.runs} runs ===")
    print(f"Test Accuracy: mean={mean_acc:.4f}  std={std_acc:.4f}")
    print(f"F1(macro)   : mean={mean_f1m:.4f}  std={std_f1m:.4f}")
    print(f"F1(micro)   : mean={mean_f1_micro:.4f}  std={std_f1_micro:.4f}")
    print(f"F1(weighted): mean={mean_f1_weighted:.4f}  std={std_f1_weighted:.4f}")
    print(f"F1(binary mal): mean={mean_f1_bin_mal:.4f}  std={std_f1_bin_mal:.4f}")
    print(f"F1(binary ben): mean={mean_f1_bin_ben:.4f}  std={std_f1_bin_ben:.4f}")
    print(f"F1(per-class ben,mal): mean={[round(x,4) for x in mean_f1_pc]}  std={[round(x,4) for x in std_f1_pc]}")

# 追記: ログ保存（JSON）
summary = {
    "timestamp": ts,
    "args": vars(args),
    "device": device,
    "runs": all_runs,
    "summary": {
        "test_acc_mean": float(np.mean(all_test_acc)) if all_test_acc else None,
        "test_acc_std":  float(np.std(all_test_acc, ddof=0)) if all_test_acc else None,

        # 互換: 既存macroのサマリは維持（変更点: 追記）
        "macro_f1_mean": float(np.mean(all_test_f1m)) if all_test_f1m else None,
        "macro_f1_std":  float(np.std(all_test_f1m, ddof=0)) if all_test_f1m else None,

        # 追記: 追加F1サマリ（変更点）
        "micro_f1_mean": float(np.mean(all_test_f1_micro)) if all_test_f1_micro else None,
        "micro_f1_std":  float(np.std(all_test_f1_micro, ddof=0)) if all_test_f1_micro else None,
        "weighted_f1_mean": float(np.mean(all_test_f1_weighted)) if all_test_f1_weighted else None,
        "weighted_f1_std":  float(np.std(all_test_f1_weighted, ddof=0)) if all_test_f1_weighted else None,
        "binary_malware_f1_mean": float(np.mean(all_test_f1_bin_mal)) if all_test_f1_bin_mal else None,
        "binary_malware_f1_std":  float(np.std(all_test_f1_bin_mal, ddof=0)) if all_test_f1_bin_mal else None,
        "binary_benign_f1_mean": float(np.mean(all_test_f1_bin_ben)) if all_test_f1_bin_ben else None,
        "binary_benign_f1_std":  float(np.std(all_test_f1_bin_ben, ddof=0)) if all_test_f1_bin_ben else None,
        "per_class_f1_mean": (np.array(all_test_f1_per_class, dtype=float).mean(axis=0).tolist()
                              if all_test_f1_per_class else None),
        "per_class_f1_std":  (np.array(all_test_f1_per_class, dtype=float).std(axis=0, ddof=0).tolist()
                              if all_test_f1_per_class else None),
        "per_class_f1_order": ["benign", "malware"],  # 追記: mean/stdの順序
    }
}
with open(log_json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 追記: 学習率カーブ（Runごとに系列を重ねる）
plt.figure()
for idx, lr_hist in enumerate(all_lr_histories, start=1):
    plt.plot(range(1, len(lr_hist)+1), lr_hist, label=f"Run {idx}")
plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.title("LR per Epoch (all runs)")
plt.legend(); plt.tight_layout(); plt.savefig(lr_png_path, dpi=150); plt.close()

# 追記: 損失カーブ（Runごと）
plt.figure()
for idx, loss_hist in enumerate(all_loss_histories, start=1):
    plt.plot(range(1, len(loss_hist)+1), loss_hist, label=f"Run {idx}")
plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Train Loss per Epoch (all runs)")
plt.legend(); plt.tight_layout(); plt.savefig(loss_png_path, dpi=150); plt.close()

# 追記: 検証精度カーブ（Runごと）
plt.figure()
for idx, acc_hist in enumerate(all_valacc_histories, start=1):
    plt.plot(range(1, len(acc_hist)+1), acc_hist, label=f"Run {idx}")
plt.xlabel("Epoch"); plt.ylabel("Val Accuracy"); plt.title("Val Accuracy per Epoch (all runs)")
plt.legend(); plt.tight_layout(); plt.savefig(valacc_png_path, dpi=150); plt.close()

# 既存の詳細出力（エポックごと）はVERBOSE_EPOCH=Falseにより標準出力しない
# （詳細は上記のJSONログに保存）
