import os
import cv2
import albumentations as A

input_dir = "/home/khang/UniformDetect/Dataset/test/non_uniform"

existing_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
existing_files.sort(key=lambda x: int(os.path.splitext(x)[0])) 
next_index = len(existing_files) + 1  

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.3)
])

num_aug = 3  

for filename in existing_files:
    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)

    for i in range(num_aug):
        augmented = transform(image=image)["image"]
        new_name = f"{next_index}.jpg"
        cv2.imwrite(os.path.join(input_dir, new_name), augmented)
        next_index += 1

print("Done!")
