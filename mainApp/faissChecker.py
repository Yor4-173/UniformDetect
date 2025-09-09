import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import pickle

# Arguments
parser = argparse.ArgumentParser(description="Uniform Detection Query")
parser.add_argument("--model", type=str, required=True, help="Path to  model")
parser.add_argument("--img", type=str, required=True, help="Path to query image")
parser.add_argument("--gallery", type=str, required=True, help="Path to gallery index")
parser.add_argument("--k", type=int, default=1, help="Top-k nearest neighbors to retrieve")
parser.add_argument("--threshold", type=float, default=0.4, help="Threshold for uniform check")
args = parser.parse_args()

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model 
model = models.resnet50(weights=None)
model.fc = nn.Identity()

state_dict = torch.load(
    args.model,
    map_location=device
)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# ---- Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(path, model, device):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).squeeze().cpu().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat.astype("float32")

# Load FAISS index & labels 
index = faiss.read_index(args.gallery)

if faiss.get_num_gpus() > 0:
    print("Using FAISS GPU")
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
else:
    print("No GPU found, using FAISS CPU")

with open("/home/khang/UniformDetect/models/gallery_labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Query 
query_emb = get_embedding(args.img, model, device)
D, I = index.search(np.array([query_emb]), k=args.k)

print("\nQuery Image:", args.img)
for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
    label = labels[idx]
    print(f"Rank {rank}: Label={label}, Distance={dist:.4f}")

#  Threshold check (only top-1) 
top1_dist = D[0][0]
if top1_dist < args.threshold:
    print("\n Correct Uniform")
else:
    print("\n Wrong Uniform")
