import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

MODEL_PATH = "/home/khang/UniformDetect/models/resnet50_preTrain.pth"

# preprocessing 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_model():
    # load ResNet50
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Identity()  # Remove fully-connected to get feature vector

    # load trained weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

def get_embedding(image_path, model):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(img_t).squeeze().numpy()

    return embedding
