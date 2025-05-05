import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog
import cv2
import torch
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def main():
    
    ROOT         = r"C:/Users/Yash Nautiyal/Desktop/AIML Project"
    test_folder  = os.path.join(ROOT, 'test')
    train_lbls   = os.path.join(ROOT, 'train', 'labels')
    weights_path = os.path.join(ROOT, 'fasterrcnn_res50.pth')
    conf_thresh  = 0.25
    project_dir  = os.path.join(ROOT, 'runs', 'detect')
    exp_name     = "custom"
    out_dir      = os.path.join(project_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        initialdir = test_folder,
        title      = "Select a test image",
        filetypes  = [("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    if not selected:
        print("No image selected. Exiting.")
        return

    orig_ids = set()
    for fname in os.listdir(train_lbls):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(train_lbls, fname), 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    orig_ids.add(int(float(parts[0])))
    orig_ids = sorted(orig_ids)
    cls_map    = {o:i+1 for i,o in enumerate(orig_ids)}
    inv_map    = {v:k   for k,v in cls_map.items()}
    num_classes= len(orig_ids) + 1

   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_ch   = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_ch, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    
    img_bgr = cv2.imread(selected)
    if img_bgr is None:
        print("Failed to load image. Exiting.")
        return
    h, w    = img_bgr.shape[:2]
    tensor  = transforms.ToTensor()(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(device)

    with torch.no_grad():
        outputs = model([tensor])[0]

    for box, lbl, scr in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if scr < conf_thresh:
            continue
        x1, y1, x2, y2 = box.int().cpu().numpy()
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
        orig_id = inv_map[int(lbl.item())]
        cv2.putText(img_bgr, f"{orig_id}:{scr:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

   
    result_path = os.path.join(out_dir, os.path.basename(selected))
    cv2.imwrite(result_path, img_bgr)
    print(f"Detections saved to: {out_dir}")

    img = cv2.imread(result_path)
    cv2.imshow("Faster R-CNN Detection", img)
    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
