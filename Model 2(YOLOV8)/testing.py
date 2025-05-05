import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog
import cv2

def main():
    
    yolov8_dir  = r"C:/Users/Yash Nautiyal/Desktop/AIML Project/yolov8"
    sys.path.insert(0, yolov8_dir)
    test_folder = r"C:/Users/Yash Nautiyal/Desktop/AIML Project/test"
    weights_path= os.path.join(yolov8_dir, "runs", "train", "exp", "weights", "best.pt")
    img_size    = 640
    conf_thresh = 0.25
    project_dir = os.path.join(yolov8_dir, "runs", "detect")
    exp_name    = "custom"

   
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

    
    from ultralytics import YOLO
    model = YOLO(weights_path)
    model.predict(
        source   = selected,
        imgsz    = img_size,
        conf     = conf_thresh,
        save     = True,
        save_txt = True,
        project  = project_dir,
        name     = exp_name,
        exist_ok = True
    )
    out_dir = os.path.join(project_dir, exp_name)
    print(f"Detections saved to: {out_dir}")

    base    = os.path.basename(selected)
    out_img = os.path.join(out_dir, base)
    if not os.path.exists(out_img):
        stem    = os.path.splitext(base)[0]
        matches = glob.glob(os.path.join(out_dir, f"{stem}*"))
        out_img = matches[0] if matches else None
    if not out_img:
        print("Could not locate result image.")
        return

    img = cv2.imread(out_img)
    if img is None:
        print("Failed to load result image.")
        return

    cv2.imshow("YOLOv8 Detection", img)
    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
