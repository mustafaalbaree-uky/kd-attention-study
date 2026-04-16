"""
Baseline training: MobileViT-small fine-tuned on CIFAR-10 with hard labels only.

Loss = CrossEntropy(student_logits, hard_labels)   — no teacher, no soft labels.

Identical architecture, dataset split, image size, optimizer, LR schedule, and
number of epochs as the KD run (train_kd.py) so the two results are directly
comparable.

All hyperparameters are read from config.yaml.
"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── Config ─────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED         = cfg["training"]["random_seed"]
EPOCHS       = cfg["training"]["epochs"]
LR           = cfg["training"]["lr"]
WEIGHT_DECAY = cfg["training"]["weight_decay"]
VAL_SPLIT    = cfg["training"]["val_split"]
ACCUM_STEPS  = cfg["training"]["grad_accum_steps"]
IMG_SIZE     = cfg["dataset"]["image_size"]
BATCH_SIZE   = cfg["dataset"]["batch_size"]
MICRO_BATCH  = BATCH_SIZE // ACCUM_STEPS
NUM_CLASSES  = 10

# ── Seed everywhere ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device         : {DEVICE}")
print(f"lr={LR}  epochs={EPOCHS}")
print(f"Effective batch={BATCH_SIZE}  micro_batch={MICRO_BATCH}  accum={ACCUM_STEPS}")

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

full_train_aug = datasets.CIFAR10(root="data", train=True, download=True,  transform=train_tf)
full_train_val = datasets.CIFAR10(root="data", train=True, download=False, transform=val_tf)

rng = torch.Generator().manual_seed(SEED)
indices   = torch.randperm(len(full_train_aug), generator=rng).tolist()
n_val     = int(len(indices) * VAL_SPLIT)
val_idx   = indices[:n_val]
train_idx = indices[n_val:]

train_set = Subset(full_train_aug, train_idx)
val_set   = Subset(full_train_val, val_idx)

print(f"Train samples  : {len(train_set):,}  |  Val samples: {len(val_set):,}")

train_loader = DataLoader(train_set, batch_size=MICRO_BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=MICRO_BATCH, shuffle=False, num_workers=0)

# ── Model ──────────────────────────────────────────────────────────────────────
from transformers import MobileViTForImageClassification

model_name = cfg["models"]["student_baseline"]
print(f"\nLoading baseline : {model_name}")
student = MobileViTForImageClassification.from_pretrained(model_name)
in_features = student.classifier.in_features    # 640 for mobilevit-small
student.classifier = nn.Linear(in_features, NUM_CLASSES)
student.to(DEVICE)
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"  Baseline params: {trainable:,} (all trainable)")

# ── Loss, optimizer, scheduler ────────────────────────────────────────────────
ce_fn     = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── CSV logger ─────────────────────────────────────────────────────────────────
csv_path = Path("results/baseline_training_log.csv")
with csv_path.open("w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_acc"])

# ── Training loop ──────────────────────────────────────────────────────────────
print("\n" + "─" * 52)
print(f"{'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValAcc':>8}")
print("─" * 52)

for epoch in range(1, EPOCHS + 1):
    student.train()
    sum_loss = correct = n = 0
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_loader):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = student(pixel_values=imgs).logits          # (B, 10)
        loss   = ce_fn(logits, labels)
        (loss / ACCUM_STEPS).backward()

        bs        = labels.size(0)
        sum_loss += loss.item() * bs
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += bs

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    train_loss = sum_loss / n
    train_acc  = correct  / n

    # ── Validation ──────────────────────────────────────────────────────────
    student.eval()
    val_correct = val_n = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            labels = labels.to(DEVICE)
            preds  = student(pixel_values=imgs.to(DEVICE)).logits.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_n       += labels.size(0)
    val_acc = val_correct / val_n

    print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:9.4f}  {val_acc:8.4f}", flush=True)

    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_acc:.6f}",
        ])

# ── Save weights ───────────────────────────────────────────────────────────────
ckpt_path = Path("checkpoints/student_baseline.pth")
torch.save(student.state_dict(), ckpt_path)
print("─" * 52)
print(f"\nTraining complete.")
print(f"  Weights → {ckpt_path}")
print(f"  CSV log → {csv_path}")
