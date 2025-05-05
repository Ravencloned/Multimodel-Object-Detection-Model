import os
import sys
import time
import ctypes
import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import numpy as np
import torchvision.ops as ops
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

# ─── Helpers ──────────────────────────────────────────────────
def get_short_path(long_path):
    buf = ctypes.create_unicode_buffer(512)
    ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, len(buf))
    return buf.value

def pick_image(test_dir):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.update()
    path = filedialog.askopenfilename(
        initialdir=test_dir,
        title="Select one test image",
        filetypes=[("Image Files","*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    if not path:
        print("No image selected. Exiting.")
        sys.exit(0)
    return path

def ensemble(boxes_list, scores_list, labels_list, iou=0.5):
    all_b = np.vstack(boxes_list)
    all_s = np.hstack(scores_list)
    all_l = np.hstack(labels_list)
    final_idxs = []
    for cls in np.unique(all_l):
        idxs = np.where(all_l==cls)[0]
        b = torch.tensor(all_b[idxs], dtype=torch.float32)
        s = torch.tensor(all_s[idxs], dtype=torch.float32)
        keep = ops.nms(b, s, iou).cpu().numpy().tolist()
        final_idxs += [int(idxs[k]) for k in keep]
    return all_b[final_idxs], all_s[final_idxs], all_l[final_idxs]

# ─── Config ──────────────────────────────────────────────────
ROOT            = r"C:/Users/Yash Nautiyal/Desktop/AIML Project"
TEST_DIR        = os.path.join(ROOT, "test", "images")

# YOLOv5
Y5_REPO_LONG    = os.path.join(ROOT, "yolov5")
Y5_REPO         = get_short_path(Y5_REPO_LONG)
Y5_WEIGHTS      = get_short_path(os.path.join(Y5_REPO_LONG, "runs","train","exp","weights","best.pt"))

# YOLOv8
Y8_WEIGHTS      = os.path.join(ROOT, "yolov8","runs","train","exp","weights","best.pt")

# Faster R-CNN
FRCNN_WEIGHTS   = os.path.join(ROOT, "fasterrcnn_res50.pth")
TRAIN_LBLS      = os.path.join(ROOT, "train", "labels")

CONF_THRESH     = 0.25

# ─── 1) PICK IMAGE ────────────────────────────────────────────
img_path = pick_image(TEST_DIR)
img_bgr  = cv2.imread(img_path)
if img_bgr is None:
    print("Failed to load image."); sys.exit(1)

# ─── 2) RUN YOLOv5 ────────────────────────────────────────────
torch.hub.set_dir(Y5_REPO)
y5 = torch.hub.load(Y5_REPO, "custom", path=Y5_WEIGHTS, source="local")
y5.conf = CONF_THRESH

t0 = time.time()
r5 = y5(img_path).xyxy[0].cpu().numpy()
print(f"YOLOv5: {len(r5)} boxes in {time.time()-t0:.3f}s")
b5, s5, l5 = r5[:,:4], r5[:,4], r5[:,5].astype(int)

# ─── 3) RUN YOLOv8 ────────────────────────────────────────────
y8 = YOLO(Y8_WEIGHTS)
y8.conf = CONF_THRESH

t0 = time.time()
o8 = y8.predict(source=img_path, save=False)[0]
print(f"YOLOv8: {len(o8.boxes)} boxes in {time.time()-t0:.3f}s")
b8 = o8.boxes.xyxy.cpu().numpy()
s8 = o8.boxes.conf.cpu().numpy()
l8 = o8.boxes.cls.cpu().numpy().astype(int)

# ─── 4) RUN Faster R-CNN ─────────────────────────────────────
orig_ids = sorted({
    int(float(line.split()[0]))
    for fn in os.listdir(TRAIN_LBLS) if fn.endswith(".txt")
    for line in open(os.path.join(TRAIN_LBLS, fn))
})
num_classes = len(orig_ids) + 1

fr = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_ch = fr.roi_heads.box_predictor.cls_score.in_features
fr.roi_heads.box_predictor = FastRCNNPredictor(in_ch, num_classes)
fr.load_state_dict(torch.load(FRCNN_WEIGHTS, map_location="cpu"))
fr.eval()

t0 = time.time()
rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
t  = transforms.ToTensor()(rgb).unsqueeze(0)
with torch.no_grad():
    out = fr(t)[0]
bR = out['boxes'].cpu().numpy()
sR = out['scores'].cpu().numpy()
lR = out['labels'].cpu().numpy().astype(int)
mask = sR >= CONF_THRESH
bR, sR, lR = bR[mask], sR[mask], lR[mask]
print(f"FasterRCNN: {len(bR)} boxes in {time.time()-t0:.3f}s")

# ─── Colors ──────────────────────────────────────────────────
COLORS = {
    'y5': (0,255,0),    # green
    'y8': (255,0,0),    # blue
    'fr': (0,0,255),    # red
    'en': (0,255,255)   # yellow
}

vis = img_bgr.copy()

# YOLOv5
for (x1,y1,x2,y2), cls, sc in zip(b5, l5, s5):
    x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
    cv2.rectangle(vis,(x1,y1),(x2,y2), COLORS['y5'], 2)
    cv2.putText(vis,f"Y5:{cls}:{sc:.2f}",(x1, y2+15),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,COLORS['y5'],1)

# YOLOv8
for (x1,y1,x2,y2), cls, sc in zip(b8, l8, s8):
    x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
    cv2.rectangle(vis,(x1,y1),(x2,y2), COLORS['y8'], 2)
    cv2.putText(vis,f"Y8:{cls}:{sc:.2f}",(x1, y2+30),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,COLORS['y8'],1)

# Faster R-CNN
for (x1,y1,x2,y2), cls, sc in zip(bR, lR, sR):
    x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
    cv2.rectangle(vis,(x1,y1),(x2,y2), COLORS['fr'], 2)
    cv2.putText(vis,f"FR:{cls}:{sc:.2f}",(x1, y2+45),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,COLORS['fr'],1)

# Ensemble
fb, fs, fl = ensemble([b5,b8,bR], [s5,s8,sR], [l5,l8,lR], iou=0.5)
for (x1,y1,x2,y2), cls, sc in zip(fb, fl, fs):
    x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
    cv2.rectangle(vis,(x1,y1),(x2,y2), COLORS['en'], 2)
    cv2.putText(vis,f"EN:{cls}:{sc:.2f}",(x1, y2+60),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,COLORS['en'],1)

# ─── Legend Overlay ─────────────────────────────────────────
legend_entries = [
    ("YOLOv5", COLORS['y5']),
    ("YOLOv8", COLORS['y8']),
    ("Faster R-CNN", COLORS['fr']),
    ("Ensemble", COLORS['en']),
]
y0 = 20
for name, col in legend_entries:
    cv2.rectangle(vis, (10, y0-12), (30, y0+12), col, -1)
    cv2.putText(vis, name, (35, y0+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y0 += 30


cv2.imshow("All Models (color-coded)", vis)
print("Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
