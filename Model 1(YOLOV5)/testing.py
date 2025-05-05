import subprocess
import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog
import cv2

yolov5_dir      = r"C:/Users/Yash Nautiyal/Desktop/AIML Project/yolov5"
test_folder     = r"C:/Users/Yash Nautiyal/Desktop/AIML Project/test"
weights_path    = os.path.join(yolov5_dir, "runs", "train", "exp", "weights", "best.pt")
img_size        = "640"
project_name    = "runs/detect"
experiment_name = "custom"


root = tk.Tk()
root.attributes('-topmost', True)   # force window to front
root.withdraw()
root.update()                      # ensure dialog appears
selected_file = filedialog.askopenfilename(
    initialdir = test_folder,
    title      = "Select an image for object detection",
    filetypes  = [("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)
root.destroy()

if not selected_file:
    print("No file selected. Exiting.")
    sys.exit(0)


detect_script = os.path.join(yolov5_dir, "detect.py")
command = [
    sys.executable, detect_script,
    "--weights", weights_path,
    "--img",     img_size,
    "--source",  selected_file,
    "--project", project_name,
    "--name",    experiment_name,
    "--exist-ok"
]
print("Running detection with command:")
print(" ", " ".join(command))
subprocess.run(command, cwd=yolov5_dir)

output_folder      = os.path.join(yolov5_dir, project_name, experiment_name)
selected_basename  = os.path.basename(selected_file)
result_image_path  = os.path.join(output_folder, selected_basename)
if not os.path.exists(result_image_path):
    stem    = os.path.splitext(selected_basename)[0]
    matches = glob.glob(os.path.join(output_folder, f"{stem}*"))
    if matches:
        result_image_path = matches[0]
    else:
        print("Could not locate detection result in:", output_folder)
        sys.exit(1)
print("Result image found:", result_image_path)


image = cv2.imread(result_image_path)
if image is None:
    print("Failed to load image:", result_image_path)
    sys.exit(1)

cv2.imshow("YOLOv5 Detection", image)
print("Press any key in the image window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
