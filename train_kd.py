"""
Knowledge Distillation training: MobileViT-small student ← frozen ViT teacher.

KD loss = alpha * T^2 * KL(student_soft || teacher_soft)
        + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

Teacher logits are pre-computed once over the full training set and cached in
memory (45k × 10 × float32 = ~1.8 MB). This avoids re-running the 85M-param
teacher on every batch during training, reducing per-epoch time by ~20×.

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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ── Config ─────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED         = cfg["training"]["random_seed"]
EPOCHS       = cfg["training"]["epochs"]
LR           = cfg["training"]["lr"]
WEIGHT_DECAY = cfg["training"]["weight_decay"]
VAL_SPLIT    = cfg["training"]["val_split"]
T            = cfg["training"]["kd_temperature"]
ALPHA        = cfg["training"]["kd_alpha"]
ACCUM_STEPS  = cfg["training"]["grad_accum_steps"]
IMG_SIZE     = cfg["dataset"]["image_size"]
BATCH_SIZE   = cfg["dataset"]["batch_size"]
MICRO_BATCH  = BATCH_SIZE // ACCUM_STEPS    # actual DataLoader batch size
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
print(f"T={T}  alpha={ALPHA}  lr={LR}  epochs={EPOCHS}")
print(f"Effective batch={BATCH_SIZE}  micro_batch={MICRO_BATCH}  accum={ACCUM_STEPS}")

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
# No-aug version of train set used for teacher pre-computation
plain_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Build train/val index split with deterministic indices
full_train_aug   = datasets.CIFAR10(root="data", train=True, download=True,  transform=train_tf)
full_train_plain = datasets.CIFAR10(root="data", train=True, download=False, transform=plain_tf)
full_train_val   = datasets.CIFAR10(root="data", train=True, download=False, transform=val_tf)

rng = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(full_train_aug), generator=rng).tolist()
n_val      = int(len(indices) * VAL_SPLIT)
val_idx    = indices[:n_val]
train_idx  = indices[n_val:]

val_set         = Subset(full_train_val,   val_idx)    # no augmentation
train_set_plain = Subset(full_train_plain, train_idx)  # no aug  — for teacher inference
train_set_aug   = Subset(full_train_aug,   train_idx)  # augmented — for student training

print(f"Train samples  : {len(train_set_aug):,}  |  Val samples: {len(val_set):,}")


# ── Dataset wrapper that injects pre-computed teacher logits ───────────────────
class KDDataset(Dataset):
    """Wraps an augmented Subset and adds per-sample teacher logits."""
    def __init__(self, aug_subset: Subset, teacher_logits: torch.Tensor):
        self.subset         = aug_subset
        self.teacher_logits = teacher_logits   # shape (N, num_classes), on CPU

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, label = self.subset[i]
        return img, label, self.teacher_logits[i]


# ── Models ─────────────────────────────────────────────────────────────────────
from transformers import MobileViTForImageClassification, ViTForImageClassification

print(f"\nLoading teacher  : {cfg['models']['teacher']}")
teacher = ViTForImageClassification.from_pretrained(cfg["models"]["teacher"])
teacher.classifier = nn.Linear(teacher.config.hidden_size, NUM_CLASSES)
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()
print(f"  Teacher params : {sum(p.numel() for p in teacher.parameters()):,} (all frozen)")

print(f"Loading student  : {cfg['models']['student_mobilevit']}")
student = MobileViTForImageClassification.from_pretrained(cfg["models"]["student_mobilevit"])
in_features = student.classifier.in_features    # 640 for mobilevit-small
student.classifier = nn.Linear(in_features, NUM_CLASSES)
student.to(DEVICE)
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"  Student params : {trainable:,} (all trainable)")


# ── Pre-compute teacher logits once over the full training set ─────────────────
# Runs teacher on CPU with large batches (no backprop → no memory pressure).
# Produces a (N_train, NUM_CLASSES) float32 tensor stored in RAM.
def precompute_teacher_logits(model, dataset, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits = []
    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            logits = model(pixel_values=imgs).logits   # CPU
            all_logits.append(logits)
            if (i + 1) % 50 == 0:
                print(f"  teacher inference: {(i+1)*batch_size}/{len(dataset)} samples", flush=True)
    return torch.cat(all_logits, dim=0)   # (N_train, NUM_CLASSES)

print("\nPre-computing teacher logits on train set (runs once)…")
teacher_logits_train = precompute_teacher_logits(teacher, train_set_plain, batch_size=128)
print(f"  Cached logits shape : {tuple(teacher_logits_train.shape)}  "
      f"({teacher_logits_train.numel()*4/1e6:.1f} MB)")

# Teacher no longer needed in GPU/MPS memory — student has full device budget
del teacher

train_kd_set = KDDataset(train_set_aug, teacher_logits_train)
train_loader = DataLoader(train_kd_set, batch_size=MICRO_BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,      batch_size=MICRO_BATCH, shuffle=False, num_workers=0)


# ── KD loss ────────────────────────────────────────────────────────────────────
ce_fn = nn.CrossEntropyLoss()

def kd_loss_fn(s_logits: torch.Tensor, t_logits: torch.Tensor,
               labels: torch.Tensor):
    """Returns (total, loss_kd, loss_hard)."""
    loss_hard = ce_fn(s_logits, labels)
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
        ["epoch", "train_loss_total", "train_loss_kd", "train_loss_hard",
         "train_acc", "val_acc"]
    )


# ── Training loop ──────────────────────────────────────────────────────────────
print("\n" + "─" * 78)
print(f"{'Epoch':>5}  {'TotalLoss':>10}  {'KD Loss':>9}  {'CE Loss':>9}  "
      f"{'TrainAcc':>9}  {'ValAcc':>8}")
print("─" * 78)

for epoch in range(1, EPOCHS + 1):
    student.train()
    sum_total = sum_kd = sum_hard = correct = n = 0
    optimizer.zero_grad()

    for step, (imgs, labels, t_logits) in enumerate(train_loader):
        labels   = labels.to(DEVICE)
        t_logits = t_logits.to(DEVICE)

        s_logits = student(pixel_values=imgs.to(DEVICE)).logits   # (B, 10)

        loss, l_kd, l_hard = kd_loss_fn(s_logits, t_logits, labels)
        (loss / ACCUM_STEPS).backward()

        bs = labels.size(0)
        sum_total += loss.item() * bs
        sum_kd    += l_kd.item() * bs
        sum_hard  += l_hard.item()* bs
        correct   += (s_logits.argmax(1) == labels).sum().item()
        n         += bs

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    train_loss = sum_total / n
    train_kd   = sum_kd    / n
    train_hard = sum_hard  / n
    train_acc  = correct   / n

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

    print(f"{epoch:5d}  {train_loss:10.4f}  {train_kd:9.4f}  {train_hard:9.4f}  "
          f"{train_acc:9.4f}  {val_acc:8.4f}", flush=True)

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
print("─" * 78)
print(f"\nTraining complete.")
print(f"  Weights → {ckpt_path}")
print(f"  CSV log → {csv_path}")
