import cv2
import os
import xml.etree.ElementTree as ET

input_img_dir = "/home/khang/Term/Uniforms Dectection.v1-uni-v1.yolov11/train/images"
input_ann_dir = "/home/khang/Term/Uniforms Dectection.v1-uni-v1.yolov11/noneUni"
output_dir = "/home/khang/Term/noneUniFig"
os.makedirs(output_dir, exist_ok=True)

for txt_file in os.listdir(input_ann_dir):
    if not txt_file.endswith(".txt"):
        continue
    img_file = txt_file.replace(".txt", ".jpg") 
    img_path = os.path.join(input_img_dir, img_file)
    label_path = os.path.join(input_ann_dir, txt_file)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        for i, line in enumerate(f):
            cls, x, y, bw, bh = map(float, line.split())
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            crop = img[y1:y2, x1:x2]
            save_path = os.path.join(output_dir, f"{txt_file[:-4]}_{i}.jpg")
            cv2.imwrite(save_path, crop)
