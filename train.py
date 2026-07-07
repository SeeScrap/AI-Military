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
from pathlib import Path
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

def get_transforms(is_train: bool, imgsz: int, config: dict | None = None):
    """Data augmentation transforms dynamically built from config."""
    if is_train:
        aug_cfg = config.get("augmentations", {}) if config else {}
        use_flip = aug_cfg.get("flip", True)
        use_rotate = aug_cfg.get("rotate", True)
        use_color = aug_cfg.get("color", False)

        transform_list = [
            transforms.Resize((imgsz + 32, imgsz + 32)),
            transforms.RandomCrop(imgsz)
        ]
        if use_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if use_rotate:
            transform_list.append(transforms.RandomRotation(15))
        if use_color:
            transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


def build_classifier(num_classes: int, backbone: str = "resnet50",
                     dropout: float = 0.3) -> nn.Module:
    """Build classifier with custom head. Supports resnet* and efficientnet_b*."""

    if backbone.startswith("resnet"):
        # ── ResNet family (resnet18, resnet34, resnet50, resnet101, resnet152)
        model_fn = getattr(models, backbone, None)
        weights_enum = getattr(models, f"{backbone.replace('resnet', 'ResNet')}_Weights", None)
        if model_fn is None:
            raise ValueError(f"Unknown backbone: {backbone}")
        model = model_fn(weights=weights_enum.DEFAULT if weights_enum else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout + 0.1),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    elif backbone.startswith("efficientnet_b"):
        # ── EfficientNet family (b0–b7)
        model_fn = getattr(models, backbone, None)
        if model_fn is None:
            raise ValueError(f"Unknown backbone: {backbone}")
        # Build weights enum name: efficientnet_b3 → EfficientNet_B3_Weights
        variant = backbone.replace("efficientnet_", "").upper()  # "B3"
        weights_name = f"EfficientNet_{variant}_Weights"
        weights_enum = getattr(models, weights_name, None)
        model = model_fn(weights=weights_enum.DEFAULT if weights_enum else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout + 0.1),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    print(f"[INFO] Built {backbone} classifier (dropout={dropout})")
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
                     config: dict | None = None, progress_callback=None):
    """
    Train ResNet50 tank classifier with full checkpoint support.

    Args:
        resume: If True, load from checkpoint before training.
        checkpoint_path: Explicit checkpoint path. If None and resume=True,
                         auto-finds the latest checkpoint.
        config: Optional dict of hyperparameter overrides from the web UI.
                Keys: backbone, epochs, batch, lr, weight_decay, imgsz,
                      val_split, dropout, augmentations, optimizer, scheduler
        progress_callback: Optional callable(epoch, total, train_loss, val_loss,
                           train_acc, val_acc) for live reporting.
    """
    global _stop_requested
    _stop_requested = False

    if config is None:
        config = {}

    c = CFG["classifier"]
    data_dir     = c["data_dir"]
    ckpt_dir     = c["checkpoint_dir"]
    weights_dir  = c["weights_dir"]
    imgsz        = int(config.get("imgsz",        config.get("img_size",   c["imgsz"])))
    batch        = int(config.get("batch",        config.get("batch_size", c["batch"])))
    epochs       = int(config.get("epochs",       c["epochs"]))
    lr           = float(config.get("lr",           c["lr"]))
    wd           = float(config.get("weight_decay", c["weight_decay"]))
    save_every   = c["save_every"]
    backbone     = config.get("backbone",     c["backbone"])
    val_split    = float(config.get("val_split",    c["val_split"]))
    dropout      = float(config.get("dropout",      0.3))
    patience     = int(config.get("patience",       10))
    monitor      = config.get("monitor",            "val_acc")
    use_mixup    = config.get("augmentations", {}).get("mixup", False)

    # ── Sanitize values ───────────────────────────────────────
    # UI may send val_split as percentage (e.g. 20) instead of fraction (0.2)
    if val_split > 1.0:
        val_split = val_split / 100.0
    val_split = max(0.05, min(0.5, val_split))  # clamp to 5%-50%
    batch    = max(1, batch)
    epochs   = max(1, epochs)
    lr       = max(1e-6, lr)
    dropout  = max(0.0, min(0.9, dropout))

    # Windows: num_workers > 0 ไม่ทำงานใน thread — ใช้ 0 เสมอบน Windows
    import platform
    num_workers = 0 if platform.system() == "Windows" else (c["num_workers"] if DEVICE.type == "cuda" else 0)

    # Log UI config overrides
    if config:
        print(f"[INFO] UI config overrides: {config}")
        print(f"[INFO] Resolved: backbone={backbone}, epochs={epochs}, batch={batch}, "
              f"lr={lr}, wd={wd}, imgsz={imgsz}, val_split={val_split}, dropout={dropout}, "
              f"patience={patience}, monitor={monitor}, mixup={use_mixup}")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Dataset ──────────────────────────────────────────────
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        print(f"[ERROR] No training data found in '{data_dir}'")
        print("        Upload images via the web UI first.")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(True, imgsz, config))
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Total images: {len(full_dataset)}")

    n_val   = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    # Safety: ensure both splits have at least 1 sample
    if n_train < 1:
        n_train = max(1, len(full_dataset) - 1)
        n_val   = len(full_dataset) - n_train
    if n_val < 1:
        n_val   = 1
        n_train = len(full_dataset) - 1
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=get_transforms(False, imgsz, config))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────
    model = build_classifier(num_classes, backbone, dropout=dropout).to(DEVICE)
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

    # ── Early Stopping State ──────────────────────────────────
    patience_counter = 0
    if monitor == "val_acc":
        best_monitored_val = best_acc
    else:
        best_monitored_val = min(history["val_loss"]) if history["val_loss"] else float("inf")

    # ── Training Loop ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training: {backbone} | {num_classes} classes | {epochs} epochs")
    print(f"  Device: {DEVICE} | FP16: {CFG['fp16']}")
    print(f"  Train: {n_train} | Val: {n_val} | Batch: {batch}")
    print(f"  LR: {lr} | WD: {wd} | ImgSz: {imgsz} | Dropout: {dropout}")
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
            
            # Mixup implementation
            if use_mixup and torch.rand(1).item() < 0.5:
                beta_dist = torch.distributions.Beta(0.2, 0.2)
                lam = beta_dist.sample().item()
                rand_index = torch.randperm(imgs.size(0)).to(DEVICE)
                target_a = labels
                target_b = labels[rand_index]
                imgs = lam * imgs + (1.0 - lam) * imgs[rand_index]
                
                with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda" and CFG["fp16"])):
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, target_a) + (1.0 - lam) * criterion(outputs, target_b)
            else:
                with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda" and CFG["fp16"])):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    
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

        # ── Early Stopping Check ──────────────────────────────
        improved = False
        if monitor == "val_acc":
            if val_acc > best_monitored_val:
                best_monitored_val = val_acc
                improved = True
        else: # val_loss
            if val_loss < best_monitored_val:
                best_monitored_val = val_loss
                improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[EARLY STOP] No improvement in {monitor} for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"[EARLY STOP] Patience reached. Stopping training early at epoch {epoch+1}.")
                break

    # ── Final plots ───────────────────────────────────────────
    plot_training_history(history,
                          os.path.join(weights_dir, "training_history.png"))
    print(f"\n[DONE] Training complete! Best ValAcc: {best_acc:.2%}")
    print(f"       Best weights: models/weights/classifier_best.pth")


# ════════════════════════════════════════════════════════════
#  AUTO YOLO DATASET GENERATOR
# ════════════════════════════════════════════════════════════

YOLO_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def import_label_studio_dataset(zip_path: str, label_studio_dir: str = "data/label_studio",
                               classifier_dir: str = "data/classifier", log_fn=None) -> tuple[int, int, list[str]]:
    """
    Extract Label Studio YOLO ZIP dataset, copy to label_studio_dir,
    and crop bounding boxes to save into classifier_dir.
    """
    import zipfile
    import uuid
    from PIL import Image

    if log_fn is None:
        log_fn = print

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Temporary directory to extract zip contents safely
    temp_extract_dir = os.path.join("data", "temp_ls_extract")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)

    try:
        log_fn("กำลังแตกไฟล์ ZIP ของ Label Studio...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        # Locate folders in temp_extract_dir
        classes_txt_path = None
        images_dir_path = None
        labels_dir_path = None

        for root, dirs, files in os.walk(temp_extract_dir):
            if "classes.txt" in files:
                classes_txt_path = os.path.join(root, "classes.txt")
            if "images" in dirs:
                images_dir_path = os.path.join(root, "images")
            if "labels" in dirs:
                labels_dir_path = os.path.join(root, "labels")

        if not classes_txt_path or not images_dir_path or not labels_dir_path:
            raise ValueError(
                "โครงสร้างไฟล์ ZIP ไม่ถูกต้อง (หา classes.txt, images/ หรือ labels/ ไม่พบ) "
                "กรุณาตรวจสอบว่าได้ส่งออกจาก Label Studio ในรูปแบบ YOLO (.zip)"
            )

        # Read class names
        with open(classes_txt_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

        if not classes:
            raise ValueError("ไม่พบรายชื่อคลาสในไฟล์ classes.txt")

        log_fn(f"พบรุ่นรถถัง/ยานเกราะ {len(classes)} คลาส: {classes}")

        # Clean and prepare target directories
        ls_images_dir = os.path.join(label_studio_dir, "images")
        ls_labels_dir = os.path.join(label_studio_dir, "labels")
        ls_classes_txt = os.path.join(label_studio_dir, "classes.txt")

        if os.path.exists(label_studio_dir):
            shutil.rmtree(label_studio_dir)
        os.makedirs(ls_images_dir, exist_ok=True)
        os.makedirs(ls_labels_dir, exist_ok=True)

        # Copy classes.txt
        shutil.copy2(classes_txt_path, ls_classes_txt)

        # Copy images and labels
        imported_images = 0
        image_files = [f for f in os.listdir(images_dir_path) if Path(f).suffix.lower() in YOLO_ALLOWED_EXT]

        for fname in image_files:
            src_img = os.path.join(images_dir_path, fname)
            dst_img = os.path.join(ls_images_dir, fname)
            shutil.copy2(src_img, dst_img)
            
            # Corresponding label
            base_name = Path(fname).stem
            lbl_name = f"{base_name}.txt"
            src_lbl = os.path.join(labels_dir_path, lbl_name)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(ls_labels_dir, lbl_name)
                shutil.copy2(src_lbl, dst_lbl)
            imported_images += 1

        log_fn(f"คัดลอกไฟล์ภาพที่นำเข้าทั้งหมด {imported_images} รูป")

        # Crop bounding boxes for Classifier dataset
        cropped_count = 0
        os.makedirs(classifier_dir, exist_ok=True)

        for fname in image_files:
            base_name = Path(fname).stem
            lbl_name = f"{base_name}.txt"
            lbl_path = os.path.join(ls_labels_dir, lbl_name)
            img_path = os.path.join(ls_images_dir, fname)

            if not os.path.exists(lbl_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
            except Exception as e:
                log_fn(f"[WARN] ไม่สามารถเปิดไฟล์รูปภาพได้: {fname} | {e}")
                continue

            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = lf.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    class_idx = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                except ValueError:
                    continue

                # Fallback class name
                if class_idx < 0 or class_idx >= len(classes):
                    class_name = f"class_{class_idx}"
                else:
                    class_name = classes[class_idx]

                # Sanitize class folder name
                safe_class = "".join(c for c in class_name if c.isalnum() or c in " -_/.")
                safe_class = safe_class.strip().replace(" ", "_").replace("/", "_")
                if not safe_class:
                    safe_class = f"class_{class_idx}"

                cls_subfolder = os.path.join(classifier_dir, safe_class)
                os.makedirs(cls_subfolder, exist_ok=True)

                # Convert normalized coordinates (YOLO format) to pixels
                x1 = (x_center - box_w / 2.0) * w
                y1 = (y_center - box_h / 2.0) * h
                x2 = (x_center + box_w / 2.0) * w
                y2 = (y_center + box_h / 2.0) * h

                # Clamp coords
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))

                # Save cropped image if box size is valid
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    try:
                        crop_img = img.crop((x1, y1, x2, y2))
                        crop_name = f"crop_ls_{uuid.uuid4().hex[:12]}.jpg"
                        crop_img.save(os.path.join(cls_subfolder, crop_name), quality=92)
                        cropped_count += 1
                    except Exception as e:
                        log_fn(f"[WARN] ไม่สามารถเซฟภาพที่ Crop ได้สำหรับ {fname} | {e}")

        log_fn(f"ทำการ Crop รูปตาม Bounding Box และบันทึกเข้า Classifier ทั้งหมด {cropped_count} รูป")
        return imported_images, cropped_count, classes

    finally:
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)


def prepare_yolo_dataset(classifier_dir: str, val_split: float = 0.2,
                         log_fn=None) -> tuple[int, int]:
    """
    Prepare YOLO dataset.
    If a Label Studio dataset exists in data/label_studio, it is split and copied.
    Otherwise, we auto-generate dummy full-image bounding boxes from classifier images.
    """
    from pathlib import Path
    import random

    if log_fn is None:
        log_fn = print

    yolo_base  = "data/datasets"
    img_train  = os.path.join(yolo_base, "images", "train")
    img_val    = os.path.join(yolo_base, "images", "val")
    lbl_train  = os.path.join(yolo_base, "labels", "train")
    lbl_val    = os.path.join(yolo_base, "labels", "val")

    # Clean old datasets
    for d in [img_train, img_val, lbl_train, lbl_val]:
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Check if Label Studio data exists
    ls_dir = CFG.get("detector", {}).get("label_studio_dir", "data/label_studio")
    ls_images_dir = os.path.join(ls_dir, "images")
    ls_labels_dir = os.path.join(ls_dir, "labels")
    ls_classes_txt = os.path.join(ls_dir, "classes.txt")

    if os.path.exists(ls_images_dir) and os.path.exists(ls_classes_txt) and os.listdir(ls_images_dir):
        # --- CASE A: Use Label Studio Dataset ---
        log_fn(f"[YOLO-PREP] พบข้อมูลจาก Label Studio ในโฟลเดอร์ '{ls_dir}'")
        
        # Read classes
        with open(ls_classes_txt, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        if not classes:
            classes = ["vehicle"]
            
        # Collect all image files
        all_images = []
        for fname in os.listdir(ls_images_dir):
            if Path(fname).suffix.lower() in YOLO_ALLOWED_EXT:
                all_images.append(fname)
                
        if not all_images:
            log_fn("[YOLO-PREP] ⚠ โฟลเดอร์รูปภาพของ Label Studio ว่างเปล่า! กำลังเปลี่ยนไปดึงข้อมูลจากรูปภาพ Classifier...")
            return _generate_dummy_yolo_dataset(classifier_dir, img_train, img_val, lbl_train, lbl_val, yolo_base, log_fn)
            
        # Shuffle and split
        random.seed(42)
        random.shuffle(all_images)
        n_val   = max(1, int(len(all_images) * val_split))
        n_train = len(all_images) - n_val
        
        train_imgs = all_images[:n_train]
        val_imgs   = all_images[n_train:]
        
        def copy_ls_subset(img_list, dst_img_dir, dst_lbl_dir):
            for fname in img_list:
                # Copy image
                src_img_path = os.path.join(ls_images_dir, fname)
                dst_img_path = os.path.join(dst_img_dir, fname)
                shutil.copy2(src_img_path, dst_img_path)
                
                # Copy label if exists
                lbl_name = f"{Path(fname).stem}.txt"
                src_lbl_path = os.path.join(ls_labels_dir, lbl_name)
                if os.path.exists(src_lbl_path):
                    dst_lbl_path = os.path.join(dst_lbl_dir, lbl_name)
                    shutil.copy2(src_lbl_path, dst_lbl_path)
                    
        copy_ls_subset(train_imgs, img_train, lbl_train)
        copy_ls_subset(val_imgs,   img_val,   lbl_val)
        
        # Update data.yaml
        data_yaml_path = "data.yaml"
        data_yaml_content = {
            "path": yolo_base,
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": {i: name for i, name in enumerate(classes)},
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            f.write("# ============================================================\n")
            f.write("# YOLO Dataset — Imported from Label Studio\n")
            f.write("# ============================================================\n\n")
            yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)
            
        log_fn(f"[YOLO-PREP] แบ่งชุดข้อมูลสำเร็จ: Train {n_train} รูป + Val {n_val} รูป")
        log_fn(f"[YOLO-PREP] อัปเดต data.yaml เรียบร้อย มี {len(classes)} คลาส: {classes}")
        return n_train, n_val
        
    else:
        # --- CASE B: Fall back to Auto-generating from Classifier ---
        return _generate_dummy_yolo_dataset(classifier_dir, img_train, img_val, lbl_train, lbl_val, yolo_base, log_fn)


def _generate_dummy_yolo_dataset(classifier_dir, img_train, img_val, lbl_train, lbl_val, yolo_base, log_fn):
    from pathlib import Path
    import random
    
    log_fn("[YOLO-PREP] ไม่พบข้อมูล Label Studio. กำลังเปิดโหมดจำลองกรอบ (Auto-generate dummy bounding boxes)...")
    
    # Collect all images from classifier folders
    all_images = []
    if os.path.isdir(classifier_dir):
        for cls_name in sorted(os.listdir(classifier_dir)):
            cls_path = os.path.join(classifier_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if Path(fname).suffix.lower() in YOLO_ALLOWED_EXT:
                    all_images.append(os.path.join(cls_path, fname))

    if not all_images:
        log_fn("[YOLO-PREP] ไม่พบรูปภาพใดๆ ในโฟลเดอร์ classifier!")
        return 0, 0

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)
    # Default split of 0.2
    val_split = 0.2
    n_val   = max(1, int(len(all_images) * val_split))
    n_train = len(all_images) - n_val

    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:]

    def copy_and_label(img_list, img_dir, lbl_dir):
        for i, src in enumerate(img_list):
            ext = Path(src).suffix.lower()
            dst_name = f"vehicle_{i:05d}{ext}"
            dst_img  = os.path.join(img_dir, dst_name)
            dst_lbl  = os.path.join(lbl_dir, f"vehicle_{i:05d}.txt")

            # Copy image
            shutil.copy2(src, dst_img)

            # Write YOLO label: class=0, full-image bbox (cx=0.5, cy=0.5, w=1.0, h=1.0)
            with open(dst_lbl, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

    copy_and_label(train_imgs, img_train, lbl_train)
    copy_and_label(val_imgs,   img_val,   lbl_val)

    # Update data.yaml
    data_yaml_path = "data.yaml"
    data_yaml_content = {
        "path": yolo_base,
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "vehicle"},
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write("# ============================================================\n")
        f.write("# YOLO Dataset — Auto-generated from classifier images\n")
        f.write("# Single class: vehicle (tank/IFV/APC/SPG)\n")
        f.write("# ============================================================\n\n")
        yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)

    log_fn(f"[YOLO-PREP] สร้าง Dataset จำลองเสร็จสิ้น: Train {n_train} รูป + Val {n_val} รูป")
    log_fn(f"[YOLO-PREP] อัปเดต data.yaml → 1 คลาส (vehicle)")
    return n_train, n_val


# ════════════════════════════════════════════════════════════
#  DETECTOR TRAINING  (Stage 1 — YOLOv8)
# ════════════════════════════════════════════════════════════

def train_detector(resume: bool = False, config: dict | None = None,
                   log_fn=None):
    """Train YOLOv8 detector. Resume supported via ultralytics built-in."""
    global _stop_requested
    from ultralytics import YOLO

    if config is None:
        config = {}
    if log_fn is None:
        log_fn = print

    d = CFG["detector"]
    data_yaml  = d["data_yaml"]
    model_name = config.get("yolo_model",  d["model"])
    epochs     = int(config.get("yolo_epochs", d["epochs"]))
    imgsz      = int(config.get("yolo_imgsz",  d["imgsz"]))
    batch      = int(config.get("yolo_batch",  d["batch"]))
    patience   = int(config.get("yolo_patience", d["patience"]))

    if not os.path.exists(data_yaml):
        log_fn(f"[ERROR] data.yaml not found: {data_yaml}")
        return

    if not os.path.isdir("data/datasets/images/train") or \
       not os.listdir("data/datasets/images/train"):
        log_fn("[ERROR] No training images found in data/datasets/images/train/")
        return

    if _stop_requested:
        log_fn("[STOP] Training stopped before detector started.")
        return

    if resume:
        last_ckpt = f"runs/detect/{d['project']}/{d['name']}/weights/last.pt"
        if os.path.exists(last_ckpt):
            log_fn(f"[RESUME] Resuming detector from: {last_ckpt}")
            model = YOLO(last_ckpt)
        else:
            log_fn("[RESUME] No YOLO checkpoint found — starting fresh.")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    log_fn(f"[YOLO] Training: {model_name} | {epochs} epochs | batch {batch} | imgsz {imgsz}")

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
        patience    = patience,
        lr0         = d["lr0"],
        resume      = resume and os.path.exists(f"runs/detect/{d['project']}/{d['name']}/weights/last.pt"),
        verbose     = True,
        exist_ok    = True,
    )
    log_fn("[DONE] Detector training complete!")
    log_fn(f"       Best weights: runs/detect/{d['project']}/{d['name']}/weights/best.pt")


# ════════════════════════════════════════════════════════════
#  COMBINED TRAINING (YOLO + Classifier)
# ════════════════════════════════════════════════════════════

def train_both(resume: bool = False, checkpoint_path: str | None = None,
               config: dict | None = None, progress_callback=None,
               log_fn=None):
    """
    Train both YOLO detector and classifier sequentially.

    1. Auto-generate YOLO dataset from classifier images
    2. Train YOLO detector (Stage 1)
    3. Clear VRAM
    4. Train classifier (Stage 2)
    """
    global _stop_requested

    if config is None:
        config = {}
    if log_fn is None:
        log_fn = print

    classifier_dir = CFG["classifier"]["data_dir"]
    val_split = float(config.get("val_split", CFG["classifier"]["val_split"]))
    if val_split > 1.0:
        val_split = val_split / 100.0

    # ── Step 1: Prepare YOLO dataset ──────────────────────
    log_fn("[STAGE 0] Preparing YOLO dataset from classifier images...")
    n_train, n_val = prepare_yolo_dataset(classifier_dir, val_split, log_fn=log_fn)
    if n_train == 0:
        log_fn("[ERROR] No images to create YOLO dataset!")
        return

    # ── Step 2: Train YOLO detector ───────────────────────
    log_fn("[STAGE 1] Starting YOLO Detector training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_detector(resume=resume, config=config, log_fn=log_fn)

    if _stop_requested:
        log_fn("[STOP] Training stopped after detector stage.")
        return

    # ── Step 3: Clear VRAM between stages ─────────────────
    log_fn("[STAGE] Clearing VRAM between stages...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # ── Step 4: Train classifier ──────────────────────────
    log_fn("[STAGE 2] Starting Classifier training...")
    train_classifier(
        resume=resume,
        checkpoint_path=checkpoint_path,
        config=config,
        progress_callback=progress_callback,
    )

    log_fn("[DONE] Both stages complete!")


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

    if args.mode == "all":
        train_both(resume=args.resume, checkpoint_path=args.checkpoint)
    elif args.mode == "classifier":
        train_classifier(resume=args.resume, checkpoint_path=args.checkpoint)
    elif args.mode == "detector":
        # Auto-prepare dataset if needed
        classifier_dir = CFG["classifier"]["data_dir"]
        prepare_yolo_dataset(classifier_dir)
        train_detector(resume=args.resume)

    # Final VRAM cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

