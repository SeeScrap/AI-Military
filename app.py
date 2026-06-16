"""
============================================================
Tank Detection AI — Flask Web Training UI
Opens at: http://localhost:5000

Features:
- Upload images + label with tank model name
- Start / Resume / Stop training
- Live training progress via Server-Sent Events (SSE)
- Checkpoint management
============================================================
"""

import os
import io
import glob
import json
import time
import uuid
import shutil
import threading
import torch
import yaml
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import (Flask, render_template, request, redirect,
                   url_for, jsonify, Response, send_from_directory)
from werkzeug.utils import secure_filename
from PIL import Image

# ── Config ───────────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

app = Flask(__name__)
app.secret_key = CFG["web"]["secret_key"]

UPLOAD_DIR = CFG["web"]["upload_dir"]
CLASSIFIER_DIR = CFG["classifier"]["data_dir"]
CHECKPOINT_DIR = CFG["classifier"]["checkpoint_dir"]
WEIGHTS_DIR = CFG["classifier"]["weights_dir"]
MAX_MB = CFG["web"]["max_upload_mb"]
TANK_CLASSES = CFG["tank_classes"]

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

os.makedirs(UPLOAD_DIR,     exist_ok=True)
os.makedirs(CLASSIFIER_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR,    exist_ok=True)

# ── Global training state ─────────────────────────────────────
training_state = {
    "running":    False,
    "epoch":      0,
    "total":      0,
    "train_loss": [],
    "val_loss":   [],
    "train_acc":  [],
    "val_acc":    [],
    "log":        [],
    "best_acc":   0.0,
    "error":      None,
}
_train_thread: threading.Thread | None = None
_stop_event = threading.Event()
_state_lock = threading.Lock()
_inference_engine = None


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════

def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        _log("VRAM cache cleared.")


def validate_dataset() -> tuple[bool, str | None]:
    """Check if dataset is sufficient for training (min 10 imgs per class)."""
    counts = get_class_counts()
    if not counts:
        return False, "ไม่พบข้อมูลรูปภาพในระบบ กรุณาอัปโหลดก่อน"

    if len(counts) < 2:
        return False, f"ต้องการอย่างน้อย 2 Class เพื่อเทรน (ปัจจุบันมี {len(counts)})"

    min_required = 10
    low_classes = [c for c, n in counts.items() if n < min_required]
    if low_classes:
        return False, f"บางรุ่นมีรูปน้อยกว่า {min_required} รูป: {', '.join(low_classes)}"

    return True, None


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def get_class_counts() -> dict:
    """Count images per class in classifier data dir."""
    counts = {}
    if os.path.isdir(CLASSIFIER_DIR):
        for cls in sorted(os.listdir(CLASSIFIER_DIR)):
            cls_path = os.path.join(CLASSIFIER_DIR, cls)
            if os.path.isdir(cls_path):
                imgs = [f for f in os.listdir(cls_path)
                        if Path(f).suffix.lower() in ALLOWED_EXT]
                counts[cls] = len(imgs)
    return counts


def get_checkpoints() -> list[dict]:
    """List available classifier checkpoints."""
    pattern = os.path.join(CHECKPOINT_DIR, "classifier_epoch_*.pth")
    files = sorted(glob.glob(pattern),
                   key=lambda x: int(x.split("epoch_")[-1].replace(".pth", "")))
    result = []
    for f in files:
        epoch = int(Path(f).stem.split("epoch_")[-1])
        size = os.path.getsize(f) / 1e6
        result.append({"epoch": epoch, "path": f, "size_mb": round(size, 1)})
    return result


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    training_state["log"].append(entry)
    if len(training_state["log"]) > 500:
        training_state["log"] = training_state["log"][-500:]
    print(entry)


def _progress_callback(epoch, total, train_loss, val_loss, train_acc, val_acc):
    training_state["epoch"] = epoch
    training_state["total"] = total
    training_state["train_loss"].append(round(train_loss, 4))
    training_state["val_loss"].append(round(val_loss, 4))
    training_state["train_acc"].append(round(train_acc, 4))
    training_state["val_acc"].append(round(val_acc, 4))
    training_state["best_acc"] = max(training_state["best_acc"], val_acc)
    _log(f"Epoch {epoch}/{total} | TrainLoss={train_loss:.4f} | "
         f"ValAcc={val_acc:.2%}")


def _run_training(resume: bool, checkpoint_path: str | None, config: dict | None = None):
    global training_state
    training_state["running"] = True
    training_state["error"] = None
    training_state["train_loss"] = []
    training_state["val_loss"] = []
    training_state["train_acc"] = []
    training_state["val_acc"] = []
    training_state["log"] = []
    training_state["epoch"] = 0
    training_state["best_acc"] = 0.0  # reset so new run starts clean

    _log("Training started...")
    if resume:
        _log(f"Resuming from checkpoint: {checkpoint_path or 'auto'}")

    try:
        # Patch the stop signal for thread-safe stop
        import train as trainer_module
        trainer_module._stop_requested = False
        _stop_event.clear()

        def patched_callback(ep, tot, tl, vl, ta, va):
            _progress_callback(ep, tot, tl, vl, ta, va)
            if _stop_event.is_set():
                trainer_module._stop_requested = True

        trainer_module.train_classifier(
            resume=resume,
            checkpoint_path=checkpoint_path,
            config=config or {},
            progress_callback=patched_callback,
        )
        _log("Training finished successfully!")
    except Exception as e:
        training_state["error"] = str(e)
        _log(f"ERROR: {e}")
    finally:
        training_state["running"] = False


# ════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    counts = get_class_counts()
    total_imgs = sum(counts.values())
    checkpoints = get_checkpoints()
    best_exists = os.path.exists(os.path.join(
        WEIGHTS_DIR, "classifier_best.pth"))
    return render_template("index.html",
                           counts=counts,
                           total_imgs=total_imgs,
                           checkpoints=checkpoints,
                           best_exists=best_exists,
                           training=training_state,
                           tank_classes=TANK_CLASSES)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        counts = get_class_counts()
        all_classes = sorted(set(list(counts.keys()) + TANK_CLASSES))
        return render_template("upload.html",
                               counts=counts,
                               tank_classes=all_classes)

    # POST — handle file uploads
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    class_name = request.form.get("class_name", "").strip()

    if not class_name:
        return jsonify({"error": "Please specify a tank model name"}), 400

    # Sanitize class name (keep alphanumeric, space, dash, slash, dot)
    safe_class = "".join(c for c in class_name if c.isalnum() or c in " -_/.")
    safe_class = safe_class.strip().replace(" ", "_")

    if not safe_class:
        return jsonify({"error": "Invalid class name"}), 400

    save_dir = os.path.join(CLASSIFIER_DIR, safe_class)
    os.makedirs(save_dir, exist_ok=True)

    saved = []
    errors = []
    for f in files:
        if not f.filename:
            continue
        if not allowed_file(f.filename):
            errors.append(f"{f.filename}: unsupported format")
            continue

        # Check size
        f.seek(0, 2)
        size_mb = f.tell() / 1e6
        f.seek(0)
        if size_mb > MAX_MB:
            errors.append(f"{f.filename}: file too large ({size_mb:.1f} MB)")
            continue

        # Save with unique name to avoid collisions
        ext = Path(secure_filename(f.filename)).suffix.lower()
        name = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(save_dir, name)

        try:
            img = Image.open(f).convert("RGB")
            img.save(path, quality=92)
            saved.append(name)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")

    return jsonify({
        "saved":   len(saved),
        "errors":  errors,
        "class":   safe_class,
        "total":   len(os.listdir(save_dir)),
    })


@app.route("/train/start", methods=["POST"])
def train_start():
    global _train_thread, _inference_engine

    with _state_lock:
        if training_state["running"]:
            return jsonify({"error": "Training already running"}), 400

    # 1. Validate dataset
    ok, err = validate_dataset()
    if not ok:
        return jsonify({"error": err}), 400

    # 2. Clear Inference Engine & VRAM to free memory for trainer
    if _inference_engine is not None:
        _log("Releasing Inference Engine to free VRAM...")
        del _inference_engine
        _inference_engine = None
    clear_vram()

    data = request.get_json(silent=True) or {}
    resume = data.get("resume", False)
    checkpoint_path = data.get("checkpoint", None)
    config = data.get("config", {})

    _train_thread = threading.Thread(
        target=_run_training,
        args=(resume, checkpoint_path, config),
        daemon=True,
    )
    _train_thread.start()
    return jsonify({"status": "started", "resume": resume})


@app.route("/train/stop", methods=["POST"])
def train_stop():
    if not training_state["running"]:
        return jsonify({"error": "No training running"}), 400
    _stop_event.set()
    return jsonify({"status": "stop_requested"})


@app.route("/train/status")
def train_status():
    return jsonify(training_state)


@app.route("/train/stream")
def train_stream():
    """Server-Sent Events for live training progress."""
    def event_generator():
        last_epoch = -1
        try:
            while True:
                ep = training_state["epoch"]
                if ep != last_epoch or not training_state["running"]:
                    data = json.dumps({
                        "epoch":      ep,
                        "total":      training_state["total"],
                        "running":    training_state["running"],
                        "best_acc":   training_state["best_acc"],
                        "train_loss": training_state["train_loss"][-1] if training_state["train_loss"] else None,
                        "val_loss":   training_state["val_loss"][-1] if training_state["val_loss"] else None,
                        "train_acc":  training_state["train_acc"][-1] if training_state["train_acc"] else None,
                        "val_acc":    training_state["val_acc"][-1] if training_state["val_acc"] else None,
                        "log":        training_state["log"][-5:],
                        "error":      training_state["error"],
                    })
                    yield f"data: {data}\n\n"
                    last_epoch = ep
                    if not training_state["running"]:
                        break
                time.sleep(1)
        except GeneratorExit:
            pass  # client disconnected cleanly

    return Response(event_generator(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/checkpoints")
def checkpoints():
    return jsonify(get_checkpoints())


@app.route("/classes")
def classes():
    return jsonify(get_class_counts())


@app.route("/models/weights/training_history.png")
def training_plot():
    path = os.path.join(WEIGHTS_DIR, "training_history.png")
    if os.path.exists(path):
        return send_from_directory(WEIGHTS_DIR, "training_history.png")
    return "", 404


def get_inference_engine(conf_threshold=None):
    global _inference_engine
    if _inference_engine is None:
        from inference import TankInferenceEngine
        _inference_engine = TankInferenceEngine()

    if conf_threshold is not None:
        _inference_engine.conf_thr = conf_threshold

    return _inference_engine


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "GET":
        best_exists = os.path.exists(os.path.join(
            WEIGHTS_DIR, "classifier_best.pth"))
        return render_template("test.html",
                               best_exists=best_exists,
                               tank_classes=TANK_CLASSES,
                               training=training_state)

    # POST - run inference
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        conf_thr = request.form.get("conf_threshold", None)
        if conf_thr is not None:
            conf_thr = float(conf_thr)

        # Read file bytes
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Could not decode image"}), 400

        h, w = img_bgr.shape[:2]

        # Load engine and update conf_threshold
        engine = get_inference_engine(conf_threshold=conf_thr)

        # Run prediction
        results = engine.predict_image(img_bgr)

        # Draw HUD overlay
        annotated_bgr = engine.draw_hud(img_bgr, results)

        # Encode annotated image back to JPEG base64
        _, buffer = cv2.imencode(".jpg", annotated_bgr)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "width": w,
            "height": h,
            "targets": results,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = CFG["web"]["port"]
    print("=" * 60)
    print("  Tank Detection AI — Web Training UI")
    print(f"  Open: http://localhost:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
