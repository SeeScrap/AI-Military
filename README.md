# Tank Detection AI — Gunner Assist System
## RTX 3050 Ti Laptop Optimized

ระบบ AI ตรวจจับและระบุรุ่นยานเกราะแบบ Real-time สำหรับช่วย Gunner

---

## 📦 ติดตั้ง

### 1. ติดตั้ง PyTorch สำหรับ CUDA (RTX 3050 Ti)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. ติดตั้ง dependencies ที่เหลือ
```bash
pip install -r requirements.txt
```

---

## 🚀 วิธีใช้งาน

### Step 1: เปิด Web UI สำหรับจัดการข้อมูลและ Train

```bash
python app.py
```
แล้วเปิด browser ไปที่ → **http://localhost:5000**

### Step 2: อัปโหลดภาพและ Label

1. ไปที่หน้า **UPLOAD DATA**
2. พิมพ์ชื่อรุ่นรถถัง (เช่น `T-72`, `M1 Abrams`)
3. ลากรูปภาพมาวาง หรือคลิกเลือกไฟล์
4. กด **UPLOAD & LABEL**

> แต่ละรุ่นควรมีอย่างน้อย **20-50 รูป** เพื่อผลลัพธ์ที่ดี

### Step 3: Train Model

```bash
# Train classifier (ผ่าน Web UI กด START TRAINING)
# หรือผ่าน command line:
python train.py --mode classifier

# Resume หลังจากหยุด (Ctrl+C หรือ Stop ผ่าน UI)
python train.py --mode classifier --resume

# Resume จาก checkpoint เฉพาะ
python train.py --mode classifier --resume --checkpoint models/checkpoints/epoch_5.pth
```

### Step 4: รัน Gunner HUD

```bash
# กล้อง webcam
python gui.py

# กล้อง index อื่น
python gui.py --source 1

# ไฟล์วิดีโอ
python gui.py --source video.mp4

# รูปภาพเดี่ยว
python gui.py --source tank.jpg
```

### Inference เฉพาะ (ไม่มี GUI)
```bash
python inference.py --source image.jpg --output result.jpg
python inference.py --source video.mp4 --output annotated.mp4
python inference.py --source 0 --show
```

---

## 🎮 Controls (Gunner HUD)

| Key | Action |
|-----|--------|
| `Q` หรือ `ESC` | ออกจากโปรแกรม |
| `S` | บันทึก screenshot |
| `P` | Pause / Resume |
| `+` | เพิ่ม confidence threshold |
| `-` | ลด confidence threshold |

---

## 📁 โครงสร้างโปรเจกต์

```
AI object/
├── data/
│   ├── classifier/        ← รูปภาพ training แยกตาม class
│   │   ├── T-72/
│   │   ├── M1_Abrams/
│   │   └── ...
│   └── datasets/          ← YOLO dataset (ถ้าต้องการ train detector)
├── models/
│   ├── checkpoints/       ← Checkpoint ทุก epoch
│   └── weights/           ← Best model weights
├── templates/             ← Web UI HTML
├── static/                ← CSS/JS
├── app.py                 ← Web server
├── train.py               ← Training script
├── inference.py           ← Inference engine
├── gui.py                 ← Gunner HUD
├── config.yaml            ← Settings
└── data.yaml              ← YOLO dataset config
```

---

## ⚙️ ปรับแต่ง config.yaml

```yaml
# เพิ่มรุ่นรถถังใหม่
tank_classes:
  - T-72
  - M1 Abrams
  - Leopard 2
  - ชื่อรุ่นใหม่  # เพิ่มได้เลย

# ปรับ batch size ถ้า VRAM เต็ม
classifier:
  batch: 16  # ลดจาก 32 ถ้า out of memory

# ปรับ confidence threshold
inference:
  conf_threshold: 0.45  # 0.0-1.0
```

---

## 🔧 Train YOLO Detector (Stage 1)

สำหรับ Train detector ต้องมี labeled dataset ในรูปแบบ YOLO:
- `data/datasets/images/train/` + `data/datasets/labels/train/`
- Label ด้วย [Roboflow](https://roboflow.com/) หรือ [Label Studio](https://labelstud.io/)

```bash
python train.py --mode detector
python train.py --mode detector --resume
```

---

## ⚠️ หมายเหตุ RTX 3050 Ti (4GB VRAM)

- ใช้ `yolov8s.pt` (ไม่ใช่ `yolov8l` หรือ `yolov8x`)
- batch_size=8 สำหรับ YOLO, batch_size=32 สำหรับ classifier
- เปิด `fp16: true` ใน config.yaml (mixed precision)
- ถ้า CUDA out of memory → ลด batch_size ลงครึ่งหนึ่ง
