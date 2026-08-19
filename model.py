import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # Load ResNet18 with weights pretrained on ImageNet (1.2M+ images, 1000 classes)
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')

        for param in self.resnet.parameters():
            param.requires_grad = False
            
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.resnet(x)