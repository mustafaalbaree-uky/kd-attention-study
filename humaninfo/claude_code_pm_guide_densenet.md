# Claude Code PM Guide — DenseNet-121 Student

This file is the student-specific companion to `claude_code_pm_guide.md`. Read the main guide first for project overview, research question, and general workflow. This file contains everything specific to the DenseNet-121 student experiment.

---

## Student identity

| Role | Model | Weights |
|---|---|---|
| Teacher | ResNet-50 | `teacher/checkpoints/teacher_finetuned.pth` |
| KD Student | DenseNet-121 | `students/densenet/checkpoints/densenet_kd.pth` |
| Baseline Student | DenseNet-121 | `students/densenet/checkpoints/densenet_baseline.pth` |

**Grad-CAM target layer:** `model.features.denseblock4` (last dense block, before norm + global pooling)

**Head replacement:** `model.classifier = nn.Linear(model.classifier.in_features, num_classes)`

---

## Why DenseNet-121

DenseNet-121 uses dense connections where each layer receives feature maps from all preceding layers. It has ~8M parameters — between ResNet-18 (~11.7M) and MobileNetV2 (~3.4M). What makes it interesting is that its feature reuse pattern is structurally very different from both ResNet-50 (teacher) and the other two students: the Grad-CAM maps may differ in character because the last dense block accumulates information across all prior layers, not just one residual skip connection. This makes the alignment question more nuanced.

---

## Directory layout

```
students/densenet/
├── train_baseline.py            ← Step 1a
├── train_kd.py                  ← Step 1b
├── generate_gradcam.py          ← Step 2 (200-image sample, local)
├── generate_gradcam_full.py     ← Step 3 (full 3925, Kaggle)
├── checkpoints/
│   ├── densenet_baseline.pth    ← output of Step 1a (Kaggle → download here)
│   └── densenet_kd.pth          ← output of Step 1b (Kaggle → download here)
└── results/
    ├── accuracy.csv
    ├── densenet_baseline_training_log.csv
    ├── densenet_kd_training_log.csv
    ├── divergence_scores.csv
    ├── summary_stats.json
    ├── figures/
    └── gradcam_full/
        ├── arrays/
        └── figures/
```

---

## Implementation status

| Step | Script | Status | Notes |
|---|---|---|---|
| 1 — Teacher training | `teacher/train_teacher.py` | **DONE** (shared) | Do not retrain |
| 1a — Baseline training | `students/densenet/train_baseline.py` | not started | |
| 1b — KD training | `students/densenet/train_kd.py` | not started | |
| 2 — Grad-CAM (200) | `students/densenet/generate_gradcam.py` | not started | |
| 3 — Grad-CAM (full) | `students/densenet/generate_gradcam_full.py` | not started | |
| 4 — Score divergence | `shared/score_divergence.py` | not started | run after Step 3 |
| 5 — Summarize | `shared/summarize.py` | not started | run after Step 4 |

---

## Kaggle checkpoint dataset name

`kd-attention-checkpoints-densenet`

Upload `densenet_baseline.pth` and `densenet_kd.pth` here after training. The teacher checkpoint lives in the existing `kd-attention-checkpoints` dataset (already uploaded from the ResNet-18 experiment).

Kaggle input paths used by `generate_gradcam_full.py`:
```
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints/teacher_finetuned.pth
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-densenet/densenet_kd.pth
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-densenet/densenet_baseline.pth
```

---

## How to run each step (Kaggle boilerplate cell)

```python
!git clone https://github.com/mustafaalbaree/kd-gradcam.git
!pip install -q pytorch-grad-cam datasets tqdm
import os; os.chdir("kd-gradcam")
```

Then run the target script:
```python
!python students/densenet/train_baseline.py
!python students/densenet/train_kd.py
!python students/densenet/generate_gradcam_full.py
```

---

## Architecture notes for Claude Code

When generating new scripts for this student, the model loader always looks like:

```python
import torchvision.models as models
import torch.nn as nn

def load_densenet(path, num_classes):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()
```

And the GradCAM target layer is always:
```python
GradCAM(model=m, target_layers=[m.features.denseblock4])
```

Note: DenseNet uses batch norm after the last dense block (`features.norm5`), then global average pooling, then the classifier. The last spatially-resolved feature map is at `denseblock4`.

---

## Key results (fill in after each run)

| Metric | Value |
|---|---|
| Teacher val_acc | (from ResNet-18 experiment) |
| Baseline val_acc | |
| KD student val_acc | |
| Mean JS divergence — KD vs teacher | |
| Mean JS divergence — Baseline vs teacher | |
| Mann-Whitney U p-value (correct vs incorrect) | |

---

## Prompt template for next Claude Code task

```
## Context
DenseNet-121 student experiment. Teacher (ResNet-50) is already trained at
teacher/checkpoints/teacher_finetuned.pth.
[Describe what has been completed so far for this student.]

## Task
[One clearly scoped task.]

## Inputs
[Exact file paths this task depends on.]

## Expected outputs
[Exact output paths and formats — all under students/densenet/]

## Constraints
- All student outputs go under students/densenet/
- GradCAM target: model.features.denseblock4
- Head: model.classifier = nn.Linear(...)
- Reproducibility: seed=42 everywhere
- Do not retrain the teacher
```
