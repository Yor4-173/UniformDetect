import os

folder = "/home/khang/UniformDetect/dataset/train/male_uniform"

images = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
images.sort()  

for i, img in enumerate(images, start=1):
    old_path = os.path.join(folder, img)
    new_name = f"{i:04}.jpg"  
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)

print(f"Change {len(images)} pictures in {folder}")

