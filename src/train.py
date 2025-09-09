import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.dataset import SiameseUniformDataset
from src.model import SiameseNetwork
from src.loss import ContrastiveLoss


def train(args):
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset & Dataloader
    train_dataset = SiameseUniformDataset(args.train_dir, transform=transform)
    val_dataset   = SiameseUniformDataset(args.val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved at {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network for Uniform Detection")

    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/siamese_uniform.pth", help="Path to save model")

    args = parser.parse_args()
    train(args)
