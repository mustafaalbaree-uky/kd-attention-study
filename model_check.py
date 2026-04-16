import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

seed = cfg["training"]["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

num_classes = cfg["training"]["num_classes"]
image_size = cfg["dataset"]["image_size"]


def load_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


teacher = load_resnet50(num_classes)
student_kd = load_resnet18(num_classes)
student_baseline = load_resnet18(num_classes)

x = torch.randn(1, 3, image_size, image_size)

for name, model in [
    ("teacher (ResNet-50)", teacher),
    ("student_kd (ResNet-18)", student_kd),
    ("student_baseline (ResNet-18)", student_baseline),
]:
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"{name}: output shape = {out.shape}")
