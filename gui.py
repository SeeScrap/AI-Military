"""
============================================================
Tank Detection AI — Gunner HUD (OpenCV)
Real-time display from camera or video file

Usage:
  python gui.py                          # default camera (index 0)
  python gui.py --source 0              # webcam
  python gui.py --source 1              # second camera
  python gui.py --source video.mp4      # video file
  python gui.py --source image.jpg      # single image

Controls:
  Q or ESC  — Quit
  S         — Save screenshot
  P         — Pause / Resume
  + / -     — Increase / Decrease confidence threshold
============================================================
"""

import os
import sys
import cv2
import time
import yaml
import argparse
import numpy as np
from datetime import datetime

# ── Config ───────────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


class GunnerHUD:
    """
    Real-time Gunner HUD display using OpenCV.
    Wraps TankInferenceEngine for live detection + classification.
    """

    PALETTE = {
        "tank":    (0, 255, 136),
        "IFV":     (0, 200, 255),
        "APC":     (255, 170, 0),
        "SPG":     (255, 60,  60),
        "vehicle": (180, 180, 180),
    }
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO  = cv2.FONT_HERSHEY_PLAIN

    def __init__(self, source, conf_threshold=None):
        self.source         = source
        self.conf_threshold = conf_threshold or CFG["inference"]["conf_threshold"]
        self.paused         = False
        self.frame_count    = 0
        self.fps_display    = 0.0
        self._fps_timer     = time.time()
        self._fps_frames    = 0

        # Try to load inference engine
        try:
            from inference import TankInferenceEngine
            self.engine = TankInferenceEngine(conf_threshold=self.conf_threshold)
            self.engine_ready = True
        except Exception as e:
            print(f"[WARN] Inference engine not ready: {e}")
            print("       Running in DEMO mode (no AI detection)")
            self.engine       = None
            self.engine_ready = False

    def _update_fps(self):
        self._fps_frames += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps_display = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_timer  = time.time()

    def _draw_crosshair(self, frame: np.ndarray):
        """Draw tactical crosshair in center of frame."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        colour = (0, 220, 100)
        size   = 24
        gap    = 8
        thick  = 1

        # Cross lines with center gap
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), colour, thick)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), colour, thick)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), colour, thick)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), colour, thick)

        # Center dot
        cv2.circle(frame, (cx, cy), 2, colour, -1)

        # Range circles
        for r in (60, 120):
            cv2.circle(frame, (cx, cy), r, (0, 120, 60), 1)

    def _draw_hud_overlay(self, frame: np.ndarray, results: list):
        """Draw full tactical HUD overlay."""
        h, w = frame.shape[:2]

        # ── Semi-transparent top/bottom bars ─────────────────
        for y1, y2 in [(0, 52), (h - 34, h)]:
            bar = frame[y1:y2, :]
            cv2.addWeighted(bar, 0.45, np.zeros_like(bar), 0.55, 0, bar)
            frame[y1:y2, :] = bar

        # ── Crosshair ─────────────────────────────────────────
        self._draw_crosshair(frame)

        # ── Top bar info ──────────────────────────────────────
        now = datetime.now().strftime("%H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")

        cv2.putText(frame, "GUNNER AI SYSTEM", (10, 22),
                    self.FONT, 0.65, (0, 255, 136), 2, cv2.LINE_AA)
        cv2.putText(frame, f"TARGETS: {len(results)}",
                    (10, 42), self.FONT, 0.45, (180, 210, 180), 1, cv2.LINE_AA)

        # FPS
        fps_txt = f"{self.fps_display:.0f} FPS"
        (fw, _), _ = cv2.getTextSize(fps_txt, self.FONT, 0.5, 1)
        cv2.putText(frame, fps_txt, (w - fw - 10, 22),
                    self.FONT, 0.5, (0, 200, 100), 1, cv2.LINE_AA)

        # Clock
        cv2.putText(frame, now, (w - 90, 42),
                    self.FONT, 0.45, (150, 180, 150), 1, cv2.LINE_AA)

        # Conf threshold
        conf_txt = f"CONF: {self.conf_threshold:.0%}"
        cv2.putText(frame, conf_txt, (w - 110, h - 10),
                    self.FONT, 0.4, (100, 150, 120), 1, cv2.LINE_AA)

        # PAUSE indicator
        if self.paused:
            cv2.putText(frame, "[ PAUSED ]",
                        (w // 2 - 60, h // 2 - 60),
                        self.FONT, 0.9, (255, 179, 0), 2, cv2.LINE_AA)

        # Not ready warning
        if not self.engine_ready:
            msg = "AI NOT LOADED — TRAIN MODEL FIRST"
            (mw, _), _ = cv2.getTextSize(msg, self.FONT, 0.55, 1)
            cv2.putText(frame, msg, ((w - mw) // 2, h // 2),
                        self.FONT, 0.55, (255, 80, 80), 1, cv2.LINE_AA)

        # Controls hint (bottom left)
        cv2.putText(frame, "Q:QUIT  S:SHOT  P:PAUSE  +/-:CONF",
                    (10, h - 10), self.FONT, 0.38, (80, 110, 90), 1, cv2.LINE_AA)

        # ── Draw detections ───────────────────────────────────
        for r in results:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            vtype  = r["vehicle_type"]
            model  = r["tank_model"]
            cd     = r["conf_det"]
            cc     = r["conf_cls"]
            colour = self.PALETTE.get(vtype, self.PALETTE["vehicle"])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Corner brackets
            cl = 20
            cv2.line(frame, (x1, y1), (x1 + cl, y1), colour, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + cl), colour, 3)
            cv2.line(frame, (x2, y1), (x2 - cl, y1), colour, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + cl), colour, 3)
            cv2.line(frame, (x1, y2), (x1 + cl, y2), colour, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - cl), colour, 3)
            cv2.line(frame, (x2, y2), (x2 - cl, y2), colour, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - cl), colour, 3)

            # Info panel
            lines = [
                (f"[{vtype.upper()}]", 0.42, (150, 200, 240), 1),
                (model,               0.55, colour,           2),
                (f"D:{cd:.0%} C:{cc:.0%}", 0.38, (160, 160, 160), 1),
            ]
            pad = 5
            line_h = 18
            panel_h = len(lines) * line_h + pad * 2
            py1 = max(0, y1 - panel_h)
            py2 = y1

            # Dark panel
            roi = frame[py1:py2, x1:min(x2, w)]
            if roi.size > 0:
                dark = np.zeros_like(roi)
                cv2.addWeighted(roi, 0.2, dark, 0.8, 0, roi)
                frame[py1:py2, x1:min(x2, w)] = roi

            ty = py1 + pad + line_h - 2
            for text, scale, col, thick in lines:
                cv2.putText(frame, text, (x1 + pad, ty),
                            self.FONT, scale, col, thick, cv2.LINE_AA)
                ty += line_h

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run inference and draw HUD on a frame."""
        if self.engine_ready and not self.paused:
            try:
                results = self.engine.predict_image(frame)
            except Exception:
                results = []
        else:
            results = []

        self._draw_hud_overlay(frame, results)
        self._update_fps()
        return frame

    def run(self):
        """Main loop."""
        src = self.source
        is_image = isinstance(src, str) and src.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))

        if is_image:
            frame = cv2.imread(src)
            if frame is None:
                print(f"[ERROR] Cannot read: {src}")
                return
            out = self._process_frame(frame)
            cv2.imshow("Gunner AI HUD", out)
            print("[INFO] Press any key to close, S to screenshot")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("s"):
                    fn = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(fn, out)
                    print(f"[SHOT] {fn}")
            cv2.destroyAllWindows()
            return

        # Camera / Video
        cap_src = int(src) if (isinstance(src, str) and src.isdigit()) else src
        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {src}")
            return

        # Camera settings
        if isinstance(cap_src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG["camera"]["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["camera"]["height"])
            cap.set(cv2.CAP_PROP_FPS, CFG["camera"]["fps"])

        cv2.namedWindow("Gunner AI HUD", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gunner AI HUD", 1280, 720)

        last_frame = None
        print("[INFO] Controls: Q=Quit  S=Screenshot  P=Pause  +/-=Confidence")

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(cap_src, str):
                        break  # Video ended
                    time.sleep(0.01)
                    continue
                last_frame = frame.copy()
                out = self._process_frame(frame)
                self.frame_count += 1
            else:
                if last_frame is not None:
                    out = last_frame.copy()
                    self._draw_hud_overlay(out, [])
                else:
                    out = np.zeros((720, 1280, 3), dtype=np.uint8)

            cv2.imshow("Gunner AI HUD", out)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):       # Quit
                break
            elif key == ord("s"):            # Screenshot
                fn = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(fn, out)
                print(f"[SHOT] {fn}")
            elif key == ord("p"):            # Pause
                self.paused = not self.paused
                print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")
            elif key == ord("+"):            # Increase confidence
                self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                if self.engine:
                    self.engine.conf_thr = self.conf_threshold
                print(f"[CONF] {self.conf_threshold:.0%}")
            elif key == ord("-"):            # Decrease confidence
                self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                if self.engine:
                    self.engine.conf_thr = self.conf_threshold
                print(f"[CONF] {self.conf_threshold:.0%}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"[DONE] Processed {self.frame_count} frames")


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Gunner AI HUD")
    p.add_argument("--source", default=str(CFG["camera"]["default_index"]),
                   help="Camera index / video path / image path (default: 0)")
    p.add_argument("--conf", type=float, default=None,
                   help="Confidence threshold (default: from config.yaml)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hud  = GunnerHUD(source=args.source, conf_threshold=args.conf)
    hud.run()
