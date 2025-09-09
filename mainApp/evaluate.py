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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Transform 
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

def evaluate_dataset(test_dir, model, index, labels, device, threshold=0.4, k=1):
    y_true, y_pred = [], []

    for cls in os.listdir(test_dir):
        cls_path = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                emb = get_embedding(img_path, model, device)
                D, I = index.search(np.array([emb]), k=k)
                pred_label = labels[I[0][0]]

                # threshold check
                if D[0][0] < threshold:
                    y_pred.append(pred_label)
                else:
                    y_pred.append("Unknown")

                y_true.append(cls)
            except Exception as e:
                print(f"Skip {img_path}: {e}")

    print("\n=== Evaluation Results ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true + y_pred)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(set(y_true + y_pred)),
                yticklabels=list(set(y_true + y_pred)))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Uniform Detection Model with FAISS")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model .pth")
    parser.add_argument("--index_path", type=str, required=True, help="Path to FAISS gallery index")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to gallery labels pickle")
    parser.add_argument("--threshold", type=float, default=0.4, help="Threshold for similarity check")
    parser.add_argument("--k", type=int, default=1, help="Top-k nearest neighbors to search")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load Model
    model = models.resnet50(weights=None)
    model.fc = nn.Identity()
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Load FAISS index
    index = faiss.read_index(args.index_path)
    if faiss.get_num_gpus() > 0:
        print("Using FAISS GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("Using FAISS CPU")

    # Load labels
    with open(args.labels_path, "rb") as f:
        labels = pickle.load(f)

    # Run evaluation
    evaluate_dataset(args.test_dir, model, index, labels, device, threshold=args.threshold, k=args.k)

if __name__ == "__main__":
    main()
