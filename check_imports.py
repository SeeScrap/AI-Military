import torch
import torchvision
import ultralytics
import cv2
import flask
import yaml

print("=== Import Check ===")
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:           {vram:.1f} GB")
print(f"OpenCV:         {cv2.__version__}")
print(f"Ultralytics:    {ultralytics.__version__}")
print(f"Flask:          {flask.__version__}")
print("ALL OK!")
