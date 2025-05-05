
import os
import multiprocessing
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class YOLODataset(Dataset):
    def __init__(self, imgs_dir, lbls_dir, cls_map, transform=None):
        print(f"[Dataset] Initializing with images: {imgs_dir}, labels: {lbls_dir}")
        self.img_dir, self.lbl_dir = imgs_dir, lbls_dir
        self.transform = transform or transforms.ToTensor()
        self.files = sorted([
            f for f in os.listdir(imgs_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ])
        print(f"[Dataset] Found {len(self.files)} images")
        self.cls_map = cls_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img_path = os.path.join(self.img_dir, fn)
        txt_path = os.path.join(self.lbl_dir, fn.rsplit('.',1)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes, labels = [], []
        if os.path.isfile(txt_path):
            for line in open(txt_path):
                parts = line.split()
                if not parts:
                    continue
                orig_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])
                xc, yc, bw, bh = xc*w, yc*h, bw*w, bh*h
                x1, y1 = xc - bw/2, yc - bh/2
                x2, y2 = xc + bw/2, yc + bh/2
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cls_map[orig_id])

        target = {
            'boxes':  torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            img = self.transform(img)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    
    ROOT       = r"C:/Users/Yash Nautiyal/Desktop/AIML Project"
    train_imgs = os.path.join(ROOT, 'train', 'images')
    train_lbls = os.path.join(ROOT, 'train', 'labels')
    valid_imgs = os.path.join(ROOT, 'test',  'images')
    valid_lbls = os.path.join(ROOT, 'test',  'labels')
 

    print(f"[Main] Dataset root: {ROOT}")
    print(f"[Main] Train images: {train_imgs}  labels: {train_lbls}")
    print(f"[Main] Valid images: {valid_imgs}  labels: {valid_lbls}")

    
    txt_files = [
        os.path.join(train_lbls, f) for f in os.listdir(train_lbls)
        if f.endswith('.txt')
    ]
    orig_ids = set()
    for path in txt_files:
        with open(path) as fp:
            for line in fp:
                parts = line.split()
                if parts:
                    orig_ids.add(int(float(parts[0])))
    orig_ids = sorted(orig_ids)
    print(f"[Main] Found YOLO class IDs: {orig_ids}")

   
    cls_map = {orig: i+1 for i, orig in enumerate(orig_ids)}
    num_classes = len(orig_ids) + 1  # +1 for background
    print(f"[Main] remap: {cls_map}, num_classes={num_classes}")

   
    tf = transforms.ToTensor()
    print("[Main] Creating datasets & loaders…")
    train_ds = YOLODataset(train_imgs, train_lbls, cls_map, transform=tf)
    valid_ds = YOLODataset(valid_imgs, valid_lbls, cls_map, transform=tf)
    print(f"[Main] train size: {len(train_ds)}, valid size: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Main] Using device: {device}")
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_ch = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_ch, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("[Main] Model & optimizer ready.")

    
    epochs = 5
    print(f"[Main] Starting training for {epochs} epochs…")
    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for i, (imgs, targets) in enumerate(train_loader, 1):
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, tgts)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

          
            if i % 10 == 0 or i == len(train_loader):
                avg = total_loss / i
                print(f"  [Epoch {epoch}] batch {i}/{len(train_loader)}  "
                      f"avg loss={avg:.4f}")

        epoch_time = time.time() - epoch_start
        epoch_avg = total_loss / len(train_loader)
        print(f"[Epoch {epoch}/{epochs}]  avg loss={epoch_avg:.4f}  "
              f"time: {epoch_time:.1f}s")

 
    save_path = os.path.join(ROOT, 'fasterrcnn_res50.pth')
    torch.save(model.state_dict(), save_path)
    print(f"[Main] Training complete. Weights saved to: {save_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
