"""
============================================================
Tank Detection AI — Inference Engine
Runs YOLO detection + ResNet classifier pipeline

Usage:
  python inference.py --source image.jpg
  python inference.py --source video.mp4
  python inference.py --source 0           # webcam
  python inference.py --source 0 --show    # display live
============================================================
"""

import os
import cv2
import yaml
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


# ── Load config ──────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() and CFG["device"] == "cuda" else "cpu")

# ── Colour palette for HUD ───────────────────────────────────
COLOURS = {
    "tank":  (0, 255, 136),    # green
    "IFV":   (0, 200, 255),    # cyan
    "APC":   (255, 180, 0),    # amber
    "SPG":   (255, 80,  80),   # red
    "default": (180, 180, 180),
}


class TankInferenceEngine:
    """Two-stage inference: YOLO detect → ResNet classify."""

    def __init__(self,
                 detector_weights: str = None,
                 classifier_weights: str = None,
                 conf_threshold: float = None,
                 iou_threshold: float = None):

        cfg_inf = CFG["inference"]
        self.conf_thr      = conf_threshold  or cfg_inf["conf_threshold"]
        self.iou_thr       = iou_threshold   or cfg_inf["iou_threshold"]
        self.min_size      = cfg_inf["min_vehicle_size"]
        det_weights        = detector_weights  or cfg_inf["detector_weights"]
        cls_weights        = classifier_weights or cfg_inf["classifier_weights"]

        # ── Stage 1: YOLO Detector ────────────────────────────
        self.detector = None
        if os.path.exists(det_weights):
            from ultralytics import YOLO
            self.detector = YOLO(det_weights)
            self.detector.to(DEVICE)
            print(f"[INFO] Detector loaded: {det_weights}")
        else:
            print(f"[WARN] Detector weights not found: {det_weights}")
            print("       Detection stage will be skipped (full-image classify mode).")

        # ── Stage 2: ResNet Classifier ────────────────────────
        self.classifier   = None
        self.class_names  = []
        if os.path.exists(cls_weights):
            ckpt = torch.load(cls_weights, map_location=DEVICE)
            self.class_names = ckpt.get("class_names", [])
            backbone         = ckpt.get("backbone", "resnet50")
            num_classes      = ckpt.get("num_classes", len(self.class_names))

            if backbone == "resnet50":
                model = models.resnet50(weights=None)
                model.fc = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(model.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )
            elif backbone.startswith("resnet"):
                model_fn = getattr(models, backbone, None)
                if model_fn is None:
                    raise ValueError(f"Unknown backbone: {backbone}")
                model = model_fn(weights=None)
                model.fc = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(model.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )
            elif backbone.startswith("efficientnet_b"):
                model_fn = getattr(models, backbone, None)
                if model_fn is None:
                    raise ValueError(f"Unknown backbone: {backbone}")
                model = model_fn(weights=None)
                in_f = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(in_f, num_classes),
                )
            else:
                raise ValueError(f"Unknown backbone in checkpoint: {backbone}")

            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.classifier = model.to(DEVICE)
            print(f"[INFO] Classifier loaded: {cls_weights} ({num_classes} classes)")
        else:
            print(f"[WARN] Classifier weights not found: {cls_weights}")
            print("       Train the classifier first via web UI or train.py")

        # ── Image transform for classifier ────────────────────
        self.cls_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    def _classify_crop(self, bgr_crop: np.ndarray) -> tuple[str, float]:
        """Run classifier on a BGR image crop."""
        if self.classifier is None:
            return "Unknown", 0.0
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inp = self.cls_transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.classifier(inp)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        label = self.class_names[idx.item()] if self.class_names else str(idx.item())
        return label, conf.item()

    def predict_image(self, img_bgr: np.ndarray) -> list[dict]:
        """
        Run full pipeline on a BGR image (as from OpenCV).

        Returns list of dicts:
            { x1, y1, x2, y2, vehicle_type, tank_model, conf_det, conf_cls }
        """
        results = []
        h, w = img_bgr.shape[:2]

        if self.detector is not None:
            # ── Stage 1: YOLO detection ───────────────────────
            yolo_res = self.detector.predict(
                img_bgr,
                conf=self.conf_thr,
                iou=self.iou_thr,
                device=DEVICE,
                verbose=False,
            )[0]

            for box in yolo_res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bw, bh = x2 - x1, y2 - y1
                if bw < self.min_size or bh < self.min_size:
                    continue

                conf_det     = float(box.conf[0])
                cls_id       = int(box.cls[0])
                vehicle_type = yolo_res.names.get(cls_id, "vehicle")

                # ── Stage 2: Classify crop ────────────────────
                x1c = max(0, x1 - 10)
                y1c = max(0, y1 - 10)
                x2c = min(w, x2 + 10)
                y2c = min(h, y2 + 10)
                crop = img_bgr[y1c:y2c, x1c:x2c]

                if crop.size > 0:
                    tank_model, conf_cls = self._classify_crop(crop)
                else:
                    tank_model, conf_cls = "Unknown", 0.0

                results.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "vehicle_type": vehicle_type,
                    "tank_model":   tank_model,
                    "conf_det":     conf_det,
                    "conf_cls":     conf_cls,
                })
        else:
            # ── No detector: classify full image ─────────────
            tank_model, conf_cls = self._classify_crop(img_bgr)
            results.append({
                "x1": 0, "y1": 0, "x2": w, "y2": h,
                "vehicle_type": "vehicle",
                "tank_model":   tank_model,
                "conf_det":     1.0,
                "conf_cls":     conf_cls,
            })

        return results

    def draw_hud(self, img_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
        """Draw tactical Gunner HUD overlay on image."""
        overlay = img_bgr.copy()
        h, w = img_bgr.shape[:2]

        for r in results:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            vtype = r["vehicle_type"]
            model = r["tank_model"]
            cd    = r["conf_det"]
            cc    = r["conf_cls"]

            colour = COLOURS.get(vtype, COLOURS["default"])

            # ── Bounding box ──────────────────────────────────
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

            # ── Corner brackets (tactical look) ──────────────
            corner = 18
            thick  = 3
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(overlay, (cx, cy), (cx + dx * corner, cy), colour, thick)
                cv2.line(overlay, (cx, cy), (cx, cy + dy * corner), colour, thick)

            # ── Label panel ───────────────────────────────────
            label_top  = f"{vtype.upper()}"
            label_main = f"{model}"
            label_conf = f"Det:{cd:.0%}  Cls:{cc:.0%}"

            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            pad   = 6

            (tw, th), _ = cv2.getTextSize(label_main, font, scale + 0.1, 2)
            panel_x1 = x1
            panel_y1 = max(0, y1 - th * 3 - pad * 4)
            panel_x2 = x1 + max(tw + pad * 2, 180)
            panel_y2 = y1

            # Semi-transparent panel
            sub = overlay[panel_y1:panel_y2, panel_x1:min(panel_x2, w)]
            if sub.size > 0:
                black = np.zeros_like(sub)
                cv2.addWeighted(sub, 0.35, black, 0.65, 0, sub)
                overlay[panel_y1:panel_y2, panel_x1:min(panel_x2, w)] = sub

            ty = panel_y1 + th + pad
            cv2.putText(overlay, label_top,  (x1 + pad, ty),
                        font, scale - 0.05, (150, 200, 255), 1, cv2.LINE_AA)
            ty += th + pad
            cv2.putText(overlay, label_main, (x1 + pad, ty),
                        font, scale + 0.1, colour, 2, cv2.LINE_AA)
            ty += th + pad
            cv2.putText(overlay, label_conf, (x1 + pad, ty),
                        font, scale - 0.1, (180, 180, 180), 1, cv2.LINE_AA)

        # ── HUD header ────────────────────────────────────────
        cv2.putText(overlay, "GUNNER AI SYSTEM",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 136), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Targets: {len(results)}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return overlay


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Tank AI Inference")
    p.add_argument("--source", default="0",
                   help="Image path / video path / camera index (default: 0)")
    p.add_argument("--output", default=None,
                   help="Save annotated output to this path")
    p.add_argument("--show", action="store_true",
                   help="Show live display window")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou",  type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = TankInferenceEngine(conf_threshold=args.conf, iou_threshold=args.iou)

    # ── Determine source type ─────────────────────────────────
    source = args.source
    is_camera = source.isdigit()
    is_image  = not is_camera and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))

    if is_image:
        # Single image
        img = cv2.imread(source)
        if img is None:
            print(f"[ERROR] Cannot read image: {source}")
            exit(1)
        results = engine.predict_image(img)
        annotated = engine.draw_hud(img, results)
        if args.output:
            cv2.imwrite(args.output, annotated)
            print(f"[OUT] Saved: {args.output}")
        if args.show or not args.output:
            cv2.imshow("Gunner AI", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        # Camera / Video
        cap = cv2.VideoCapture(int(source) if is_camera else source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {source}")
            exit(1)

        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps  = cap.get(cv2.CAP_PROP_FPS) or 30
            fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(args.output, fourcc, fps, (fw, fh))

        print("[INFO] Press 'q' to quit, 's' to screenshot")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results  = engine.predict_image(frame)
            annotated = engine.draw_hud(frame, results)

            if writer:
                writer.write(annotated)
            if args.show or is_camera:
                cv2.imshow("Gunner AI", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    ss_path = f"screenshot_{frame_count:04d}.jpg"
                    cv2.imwrite(ss_path, annotated)
                    print(f"[SHOT] Saved {ss_path}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[DONE] Processed {frame_count} frames")
