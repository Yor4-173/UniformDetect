import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import pickle

# Load ResNet50
model = models.resnet50(weights=None)
model.fc = nn.Identity()
state_dict = torch.load("/home/khang/UniformDetect/models/resnet50_preTrain.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(path, model):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor).squeeze().numpy()
    # normalize để cosine similarity hoạt động tốt
    feat = feat / np.linalg.norm(feat)
    return feat

gallery_dir = "/home/khang/UniformDetect/dataset/train"

embeddings = []
labels = []

for cls in os.listdir(gallery_dir):
    cls_path = os.path.join(gallery_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        try:
            emb = get_embedding(img_path, model)
            embeddings.append(emb)
            labels.append(cls)  # gán nhãn theo folder
        except Exception as e:
            print(f"Skip {img_path}: {e}")

embeddings = np.array(embeddings).astype("float32")

# Tạo FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Lưu index + labels
faiss.write_index(index, "/home/khang/UniformDetect/models/gallery.index")
with open("/home/khang/UniformDetect/models/gallery_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("Gallery index built:", len(labels), "images")
