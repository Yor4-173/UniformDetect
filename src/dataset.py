import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SiameseUniformDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Save (img_path, class_name)
        self.image_list = []
        self.class_to_images = {}

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_images[class_name] = []
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_path):
                    self.image_list.append((img_path, class_name))
                    self.class_to_images[class_name].append(img_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img1_path, cls1 = self.image_list[idx]
        img1 = Image.open(img1_path).convert("RGB")

        if random.random() < 0.5:
            # Same class
            img2_path = random.choice(self.class_to_images[cls1])
            label = 1
        else:
            # Diference class
            other_classes = [c for c in self.class_to_images.keys() if c != cls1]
            cls2 = random.choice(other_classes)
            img2_path = random.choice(self.class_to_images[cls2])
            label = 0

        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
