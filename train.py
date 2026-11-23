import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, 2)

print("ResNet50 Crack Classification Model")
print(model)
