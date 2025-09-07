import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.dataset import SiameseUniformDataset
from src.model import SiameseNetwork
from src.loss import ContrastiveLoss

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = SiameseUniformDataset("dataset/train", transform=transform)
val_dataset   = SiameseUniformDataset("dataset/valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss = 0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()
        
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "models/siamese_uniform.pth")
