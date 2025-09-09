import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = models.resnet50(weights=None)  
model.fc = nn.Identity()  # remove fully-connected to get embedding
state_dict = torch.load("/home/khang/UniformDetect/models/resnet50_preTrain.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # batch 1
    with torch.no_grad():
        feat = model(img_tensor)  # [1, 2048, 1, 1]
    return feat.squeeze().numpy()

# Query
gallery = [
    get_embedding("/home/khang/UniformDetect/dataset/valid/female_uniform/0012.jpg"),
    get_embedding("/home/khang/UniformDetect/dataset/valid/male_uniform/0009.jpg"),
]

gallery = np.array(gallery)
query = get_embedding("testing/TestImg1.jpg")

# Count cosine similarity
sims = cosine_similarity([query], gallery)[0]
print("Similarity with gallery:", sims)

if max(sims) > 0.8:  # threshold 
    print("Correct Uniform")
else:
    print("Wrong Uniform")
