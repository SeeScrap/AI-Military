"""
============================================================
Tank Detection AI — Resumable Training Script
RTX 3050 Ti Laptop (4GB VRAM) Optimized

Usage:
  # Train classifier (stage 2)
  python train.py --mode classifier
  python train.py --mode classifier --resume          # auto-find latest checkpoint
  python train.py --mode classifier --resume --checkpoint models/checkpoints/epoch_5.pth

  # Train detector (stage 1 - requires labeled YOLO dataset)
  python train.py --mode detector
  python train.py --mode detector --resume            # resume YOLO training

  # Train both
  python train.py --mode all
============================================================
"""

import os
import sys
import time
import glob
import signal
import shutil
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load config ──────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() and CFG["device"] == "cuda" else "cpu")
print(f"[INFO] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Graceful stop flag ────────────────────────────────────────
# Set this to True from outside (web UI thread) to stop training after current epoch
_stop_requested = False

def _signal_handler(sig, frame):
    global _stop_requested
    print("\n[WARN] Stop signal received — will save checkpoint after this epoch...")
    _stop_requested = True


# ════════════════════════════════════════════════════════════
#  CLASSIFIER TRAINING  (Stage 2 — ResNet50)
# ════════════════════════════════════════════════════════════

def get_transforms(is_train: bool, imgsz: int):
    """Data augmentation transforms."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((imgsz + 32, imgsz + 32)),
            transforms.RandomCrop(imgsz),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


def build_classifier(num_classes: int, backbone: str = "resnet50") -> nn.Module:
    """Build ResNet50 classifier with custom head."""
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif backbone == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def save_checkpoint(state: dict, path: str):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[CKPT] Saved checkpoint -> {path}")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Auto-find the latest checkpoint file."""
    pattern = os.path.join(checkpoint_dir, "classifier_epoch_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by epoch number
    files.sort(key=lambda x: int(x.split("epoch_")[-1].replace(".pth", "")))
    return files[-1]


def plot_training_history(history: dict, save_path: str):
    """Plot and save loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0a0e12")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111820")
        ax.tick_params(colors="#aabbcc")
        ax.spines[:].set_color("#334455")

    ax1.plot(history["train_loss"], color="#00ff88", label="Train Loss")
    ax1.plot(history["val_loss"],   color="#ffb300", label="Val Loss")
    ax1.set_title("Loss", color="#e0e8f0")
    ax1.legend()
    ax1.set_xlabel("Epoch", color="#aabbcc")

    ax2.plot(history["train_acc"], color="#00ff88", label="Train Acc")
    ax2.plot(history["val_acc"],   color="#ffb300", label="Val Acc")
    ax2.set_title("Accuracy", color="#e0e8f0")
    ax2.legend()
    ax2.set_xlabel("Epoch", color="#aabbcc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Training history saved -> {save_path}")


def train_classifier(resume: bool = False, checkpoint_path: str | None = None,
                     progress_callback=None):
    """
    Train ResNet50 tank classifier with full checkpoint support.

    Args:
        resume: If True, load from checkpoint before training.
        checkpoint_path: Explicit checkpoint path. If None and resume=True,
                         auto-finds the latest checkpoint.
        progress_callback: Optional callable(epoch, total, train_loss, val_loss,
                           train_acc, val_acc) for live reporting.
    """
    global _stop_requested
    _stop_requested = False

    c = CFG["classifier"]
    data_dir     = c["data_dir"]
    ckpt_dir     = c["checkpoint_dir"]
    weights_dir  = c["weights_dir"]
    imgsz        = c["imgsz"]
    batch        = c["batch"]
    epochs       = c["epochs"]
    lr           = c["lr"]
    wd           = c["weight_decay"]
    save_every   = c["save_every"]
    backbone     = c["backbone"]
    val_split    = c["val_split"]
    # Windows: num_workers > 0 ไม่ทำงานใน thread — ใช้ 0 เสมอบน Windows
    import platform
    num_workers = 0 if platform.system() == "Windows" else (c["num_workers"] if DEVICE.type == "cuda" else 0)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Dataset ──────────────────────────────────────────────
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        print(f"[ERROR] No training data found in '{data_dir}'")
        print("        Upload images via the web UI first.")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(True, imgsz))
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")

    n_val   = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=get_transforms(False, imgsz))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────
    model = build_classifier(num_classes, backbone).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler(enabled=(DEVICE.type == "cuda" and CFG["fp16"]))

    start_epoch = 0
    best_acc    = 0.0
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Resume from checkpoint ────────────────────────────────
    if resume:
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(ckpt_dir)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[RESUME] Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
            ckpt_num_classes = ckpt.get("num_classes", None)

            if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
                # ── Class count changed: partial load (transfer learning) ──
                print(f"[RESUME] ⚠ Class count changed: {ckpt_num_classes} → {num_classes}")
                print(f"[RESUME]   Loading backbone weights, reinitializing classifier head...")

                # Filter out mismatched keys (the fc/classifier layers)
                saved_state = ckpt["model_state"]
                model_state = model.state_dict()
                compatible = {}
                skipped = []
                for k, v in saved_state.items():
                    if k in model_state and v.shape == model_state[k].shape:
                        compatible[k] = v
                    else:
                        skipped.append(k)
                model_state.update(compatible)
                model.load_state_dict(model_state)

                loaded_pct = len(compatible) / len(saved_state) * 100
                print(f"[RESUME]   Loaded {len(compatible)}/{len(saved_state)} layers ({loaded_pct:.0f}%)")
                print(f"[RESUME]   Skipped (reinit): {skipped}")
                print(f"[RESUME]   Training restarts from epoch 0 with new head.")
                # Don't restore optimizer/scheduler/epoch — it's a new head
                best_acc = 0.0

                # Delete old incompatible checkpoints
                old_ckpts = glob.glob(os.path.join(ckpt_dir, "classifier_epoch_*.pth"))
                for f in old_ckpts:
                    try:
                        os.remove(f)
                        print(f"[CLEAN] Removed incompatible checkpoint: {os.path.basename(f)}")
                    except OSError:
                        pass  # skip locked files
            else:
                # ── Same class count: full resume ──
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch = ckpt["epoch"] + 1
                best_acc    = ckpt.get("best_acc", 0.0)
                history     = ckpt.get("history", history)
                print(f"[RESUME] Resuming from epoch {start_epoch} | Best acc: {best_acc:.2%}")
        else:
            print("[RESUME] No checkpoint found — starting fresh.")

    # ── Save class names ──────────────────────────────────────
    class_names_path = os.path.join(weights_dir, "class_names.txt")
    with open(class_names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    # ── Training Loop ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training: {backbone} | {num_classes} classes | {epochs} epochs")
    print(f"  Device: {DEVICE} | FP16: {CFG['fp16']}")
    print(f"  Train: {n_train} | Val: {n_val} | Batch: {batch}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        if _stop_requested:
            print("[STOP] Training stopped by user.")
            break

        epoch_start = time.time()

        # ── Train phase ───────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{epochs} [Train]",
                    leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda" and CFG["fp16"])):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validation phase ──────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"Epoch {epoch+1}/{epochs} [Val]",
                                     leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda" and CFG["fp16"])):
                    outputs = model(imgs)
                    loss    = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {epoch+1:03d}/{epochs}] "
              f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.2%} | "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2%} | "
              f"Time={elapsed:.1f}s")

        # ── Progress callback (for Web UI) ────────────────────
        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, val_loss,
                              train_acc, val_acc)

        # ── Save best weights ─────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(weights_dir, "classifier_best.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "best_acc":    best_acc,
                "class_names": class_names,
                "backbone":    backbone,
                "num_classes": num_classes,
            }, best_path)
            print(f"[BEST] New best model saved! ValAcc={val_acc:.2%}")

        # ── Save periodic checkpoint ──────────────────────────
        if (epoch + 1) % save_every == 0 or _stop_requested:
            ckpt_path = os.path.join(ckpt_dir, f"classifier_epoch_{epoch+1}.pth")
            save_checkpoint({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_acc":        best_acc,
                "history":         history,
                "class_names":     class_names,
                "backbone":        backbone,
                "num_classes":     num_classes,
            }, ckpt_path)

            # ── Cleanup: keep only the latest 3 checkpoints ──
            max_keep = 3
            all_ckpts = sorted(
                glob.glob(os.path.join(ckpt_dir, "classifier_epoch_*.pth")),
                key=os.path.getmtime
            )
            if len(all_ckpts) > max_keep:
                for old in all_ckpts[:-max_keep]:
                    try:
                        os.remove(old)
                        print(f"[CLEAN] Removed old checkpoint: {os.path.basename(old)}")
                    except OSError:
                        pass  # skip if file is locked

    # ── Final plots ───────────────────────────────────────────
    plot_training_history(history,
                          os.path.join(weights_dir, "training_history.png"))
    print(f"\n[DONE] Training complete! Best ValAcc: {best_acc:.2%}")
    print(f"       Best weights: models/weights/classifier_best.pth")


# ════════════════════════════════════════════════════════════
#  DETECTOR TRAINING  (Stage 1 — YOLOv8)
# ════════════════════════════════════════════════════════════

def train_detector(resume: bool = False):
    """Train YOLOv8 detector. Resume supported via ultralytics built-in."""
    from ultralytics import YOLO

    d = CFG["detector"]
    data_yaml  = d["data_yaml"]
    model_name = d["model"]
    epochs     = d["epochs"]
    imgsz      = d["imgsz"]
    batch      = d["batch"]

    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        print("        Place YOLO-format images+labels in data/datasets/")
        return

    if not os.path.isdir("data/datasets/images/train") or \
       not os.listdir("data/datasets/images/train"):
        print("[ERROR] No training images found in data/datasets/images/train/")
        return

    if resume:
        # Find last YOLO run checkpoint
        last_ckpt = "models/detector/weights/last.pt"
        if os.path.exists(last_ckpt):
            print(f"[RESUME] Resuming detector from: {last_ckpt}")
            model = YOLO(last_ckpt)
        else:
            print("[RESUME] No YOLO checkpoint found — starting fresh.")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    print(f"\n{'='*60}")
    print(f"  YOLOv8 Detector Training")
    print(f"  Model: {model_name} | Epochs: {epochs} | Batch: {batch}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    model.train(
        data        = data_yaml,
        epochs      = epochs,
        imgsz       = imgsz,
        batch       = batch,
        device      = 0 if DEVICE.type == "cuda" else "cpu",
        half        = CFG["fp16"],
        project     = d["project"],
        name        = d["name"],
        save_period = d["save_period"],
        patience    = d["patience"],
        lr0         = d["lr0"],
        resume      = resume and os.path.exists("models/detector/weights/last.pt"),
        verbose     = True,
    )
    print("[DONE] Detector training complete!")
    print("       Best weights: models/detector/weights/best.pt")


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tank Detection AI — Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --mode classifier
  python train.py --mode classifier --resume
  python train.py --mode classifier --resume --checkpoint models/checkpoints/epoch_5.pth
  python train.py --mode detector
  python train.py --mode detector --resume
  python train.py --mode all
        """
    )
    parser.add_argument("--mode", choices=["classifier", "detector", "all"],
                        default="classifier",
                        help="What to train (default: classifier)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path for classifier resume")
    return parser.parse_args()


if __name__ == "__main__":
    # Register Ctrl+C handler only when running as CLI (main thread)
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = parse_args()

    if args.mode in ("classifier", "all"):
        train_classifier(resume=args.resume, checkpoint_path=args.checkpoint)

    if args.mode in ("detector", "all"):
        train_detector(resume=args.resume)

    # Final VRAM cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

