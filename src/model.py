import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        base = models.resnet34(weights=None)
        base.fc = nn.Linear(base.fc.in_features, embedding_dim)
        self.base = base
    
    def forward_one(self, x):
        return self.base(x)
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2
