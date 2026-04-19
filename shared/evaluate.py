"""
Evaluate teacher and both student checkpoints on the ImageNette validation split.
Downloads ImageNette from fast.ai if not already present (avoids HuggingFace
datasets library, which is incompatible with Python 3.14).
Writes results/resnet18_accuracy.csv with top-1 accuracy per model.
"""
import csv
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED        = cfg["training"]["seed"]
NUM_CLASSES = cfg["training"]["num_classes"]
IMG_SIZE    = cfg["dataset"]["image_size"]
BATCH_SIZE  = cfg["dataset"]["batch_size"]

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 320px source has enough resolution after Resize(256)+CenterCrop(224)
_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
_DATA_DIR       = Path("data")
_IMAGENETTE_DIR = _DATA_DIR / "imagenette2-320"

# ── Transform (no augmentation) ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def download_imagenette():
    if _IMAGENETTE_DIR.exists():
        return
    _DATA_DIR.mkdir(exist_ok=True)
    tgz_path = _DATA_DIR / "imagenette2-320.tgz"
    print(f"Downloading ImageNette from fast.ai (~330 MB) …")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(_IMAGENETTE_URL, tgz_path, reporthook=_progress)
    print()
    print("Extracting …")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(_DATA_DIR)
    tgz_path.unlink()
    print("Done.\n")


# ── Model loaders ──────────────────────────────────────────────────────────────
def load_resnet50(ckpt_path, num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


def load_resnet18(ckpt_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = seen = 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        correct += (model(images).argmax(1) == labels).sum().item()
        seen    += labels.size(0)
    return correct / seen


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    download_imagenette()

    val_dir = _IMAGENETTE_DIR / "val"
    test_ds = tv_datasets.ImageFolder(val_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    print(f"Test samples: {len(test_ds):,}  |  Classes: {len(test_ds.classes)}\n")

    models_cfg = [
        ("teacher_resnet50",         "checkpoints/teacher_finetuned.pth", load_resnet50),
        ("student_kd_resnet18",      "checkpoints/resnet18_kd.pth",       load_resnet18),
        ("student_baseline_resnet18","checkpoints/resnet18_baseline.pth", load_resnet18),
    ]

    Path("results").mkdir(exist_ok=True)
    csv_path = Path("results/resnet18_accuracy.csv")
    rows = []

    for model_name, ckpt_path, loader_fn in models_cfg:
        print(f"Evaluating {model_name} …")
        model = loader_fn(ckpt_path, NUM_CLASSES).to(DEVICE)
        acc   = evaluate(model, test_loader)
        print(f"  {model_name}: test_accuracy = {acc:.4f}")
        rows.append((model_name, ckpt_path, acc))

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "checkpoint", "test_accuracy"])
        for model_name, ckpt_path, acc in rows:
            w.writerow([model_name, ckpt_path, f"{acc:.4f}"])

    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
