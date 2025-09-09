import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import pickle

# Model & Transform
def load_model(weight_path, device="cpu"):
    model = models.resnet50(weights=None)
    model.fc = nn.Identity()
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(path, model, device="cpu"):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze().cpu().numpy()
    feat = feat / np.linalg.norm(feat)  # normalize
    return feat.astype("float32")

# Build gallery

def build_gallery(gallery_dir, model, index_path, label_path, device):
    embeddings = []
    labels = []

    for cls in os.listdir(gallery_dir):
        cls_path = os.path.join(gallery_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                emb = get_embedding(img_path, model, device)
                embeddings.append(emb)
                labels.append(cls)
            except Exception as e:
                print(f"Skip {img_path}: {e}")

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(label_path, "wb") as f:
        pickle.dump(labels, f)

    print(" Gallery index built:", len(labels), "images")

# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS Gallery for Uniform Detection")
    parser.add_argument("--gallery-dir", type=str, default="dataset/train",
                        help="Path to gallery dataset (train set)")
    parser.add_argument("--weight", type=str, default="models/resnet50_preTrain.pth",
                        help="Path to pretrained model weights")
    parser.add_argument("--index-path", type=str, default="models/gallery.index",
                        help="Path to save FAISS index")
    parser.add_argument("--label-path", type=str, default="models/gallery_labels.pkl",
                        help="Path to save labels")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")

    args = parser.parse_args()

    model = load_model(args.weight, args.device)
    build_gallery(args.gallery_dir, model, args.index_path, args.label_path, args.device)
