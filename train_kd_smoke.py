"""
Knowledge Distillation training: MobileViT-small student ← frozen ViT teacher.

KD loss = alpha * T^2 * KL(student_soft || teacher_soft)
        + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

All hyperparameters are read from config.yaml.
"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── Config ─────────────────────────────────────────────────────────────────────
with open("config_smoke.yaml") as f:
    cfg = yaml.safe_load(f)

SEED          = cfg["training"]["random_seed"]
EPOCHS        = cfg["training"]["epochs"]
LR            = cfg["training"]["lr"]
WEIGHT_DECAY  = cfg["training"]["weight_decay"]
VAL_SPLIT     = cfg["training"]["val_split"]
T             = cfg["training"]["kd_temperature"]
ALPHA         = cfg["training"]["kd_alpha"]
IMG_SIZE      = cfg["dataset"]["image_size"]
BATCH_SIZE    = cfg["dataset"]["batch_size"]
NUM_CLASSES   = 10

# ── Seed everywhere ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"T={T}, alpha={ALPHA}, lr={LR}, epochs={EPOCHS}")

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
# ImageNet statistics — all three models are pretrained on ImageNet
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

# Build train/val split with deterministic indices, applying the correct
# transform to each split (val must not use augmentation).
full_train_aug = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
full_train_val = datasets.CIFAR10(root="data", train=True, download=False, transform=val_tf)

rng = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(full_train_aug), generator=rng).tolist()
n_val   = int(len(indices) * VAL_SPLIT)
val_idx, train_idx = indices[:n_val], indices[n_val:]

train_set = Subset(full_train_aug, train_idx)
val_set   = Subset(full_train_val, val_idx)

# num_workers=0 avoids multiprocessing issues on macOS with Python 3.14
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"Train samples: {len(train_set):,}  |  Val samples: {len(val_set):,}")

# ── Models ─────────────────────────────────────────────────────────────────────
from transformers import MobileViTForImageClassification, ViTForImageClassification

# --- Teacher (ViT-base): backbone frozen, head replaced, no grad anywhere ---
print(f"\nLoading teacher  : {cfg['models']['teacher']}")
teacher = ViTForImageClassification.from_pretrained(cfg["models"]["teacher"])
teacher.classifier = nn.Linear(teacher.config.hidden_size, NUM_CLASSES)
# Freeze every parameter — teacher is never updated
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval().to(DEVICE)
print(f"  Teacher params  : {sum(p.numel() for p in teacher.parameters()):,} (all frozen)")

# --- Student (MobileViT-small): head replaced, backbone + head both trained ---
print(f"Loading student  : {cfg['models']['student_mobilevit']}")
student = MobileViTForImageClassification.from_pretrained(cfg["models"]["student_mobilevit"])
in_features = student.classifier.in_features          # 640 for mobilevit-small
student.classifier = nn.Linear(in_features, NUM_CLASSES)
student.to(DEVICE)
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"  Student params  : {trainable:,} (all trainable)")

# ── KD loss ────────────────────────────────────────────────────────────────────
ce_fn = nn.CrossEntropyLoss()

def kd_loss_fn(s_logits: torch.Tensor, t_logits: torch.Tensor,
               labels: torch.Tensor):
    """Returns (total_loss, loss_kd, loss_hard)."""
    # Hard-label cross-entropy
    loss_hard = ce_fn(s_logits, labels)

    # Soft-label KL divergence scaled by T^2 (Hinton et al. 2015)
    s_log_soft = F.log_softmax(s_logits / T, dim=-1)
    t_soft     = F.softmax(t_logits  / T, dim=-1)
    loss_kd    = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T ** 2)

    total = ALPHA * loss_kd + (1.0 - ALPHA) * loss_hard
    return total, loss_kd, loss_hard

# ── Optimizer & scheduler ──────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── CSV logger ─────────────────────────────────────────────────────────────────
csv_path = Path("results/kd_training_log.csv")
with csv_path.open("w", newline="") as f:
    csv.writer(f).writerow(
        ["epoch", "train_loss_total", "train_loss_kd", "train_loss_hard", "train_acc", "val_acc"]
    )

# ── Training loop ──────────────────────────────────────────────────────────────
print("\n" + "─" * 75)
print(f"{'Epoch':>5}  {'TotalLoss':>10}  {'KD Loss':>9}  {'CE Loss':>9}  {'TrainAcc':>9}  {'ValAcc':>8}")
print("─" * 75)

for epoch in range(1, EPOCHS + 1):
    student.train()
    sum_total = sum_kd = sum_hard = correct = n = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # Teacher forward — no grad, no weight updates
        with torch.no_grad():
            t_logits = teacher(pixel_values=imgs).logits     # (B, 10)

        # Student forward
        s_logits = student(pixel_values=imgs).logits         # (B, 10)

        loss, l_kd, l_hard = kd_loss_fn(s_logits, t_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        sum_total += loss.item()  * bs
        sum_kd    += l_kd.item()  * bs
        sum_hard  += l_hard.item()* bs
        correct   += (s_logits.argmax(1) == labels).sum().item()
        n         += bs

    scheduler.step()

    train_loss  = sum_total / n
    train_kd    = sum_kd    / n
    train_hard  = sum_hard  / n
    train_acc   = correct   / n

    # ── Validation ────────────────────────────────────────────────────────────
    student.eval()
    val_correct = val_n = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = student(pixel_values=imgs).logits.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_n       += labels.size(0)
    val_acc = val_correct / val_n

    print(f"{epoch:5d}  {train_loss:10.4f}  {train_kd:9.4f}  {train_hard:9.4f}  {train_acc:9.4f}  {val_acc:8.4f}")

    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_kd:.6f}",
            f"{train_hard:.6f}", f"{train_acc:.6f}",
            f"{val_acc:.6f}",
        ])

# ── Save weights ───────────────────────────────────────────────────────────────
ckpt_path = Path("checkpoints/student_kd.pth")
torch.save(student.state_dict(), ckpt_path)
print("─" * 75)
print(f"\nTraining complete.")
print(f"  Weights  → {ckpt_path}")
print(f"  CSV log  → {csv_path}")
