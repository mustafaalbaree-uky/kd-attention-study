# Claude Code PM Guide — MobileNetV2 Student

This file is the student-specific companion to `claude_code_pm_guide.md`. Read the main guide first for project overview, research question, and general workflow. This file contains everything specific to the MobileNetV2 student experiment.

---

## Student identity

| Role | Model | Weights |
|---|---|---|
| Teacher | ResNet-50 | `teacher/checkpoints/teacher_finetuned.pth` |
| KD Student | MobileNetV2 | `students/mobilenet/checkpoints/mobilenet_kd.pth` |
| Baseline Student | MobileNetV2 | `students/mobilenet/checkpoints/mobilenet_baseline.pth` |

**Grad-CAM target layer:** `model.features[-1]` (last ConvBNActivation block before global average pooling)

**Head replacement:** `model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)`

---

## Why MobileNetV2

MobileNetV2 uses inverted residual blocks with depthwise separable convolutions — it is roughly 4× smaller than ResNet-18 by parameter count (~3.4M vs ~11.7M). Testing it as a student lets us ask: does a structurally different, more compressed student still benefit from soft teacher labels in terms of Grad-CAM alignment? Architectural distance from the teacher is larger here than with ResNet-18.

---

## Directory layout

```
students/mobilenet/
├── train_baseline.py            ← Step 1a
├── train_kd.py                  ← Step 1b
├── generate_gradcam.py          ← Step 2 (200-image sample, local)
├── generate_gradcam_full.py     ← Step 3 (full 3925, Kaggle)
├── checkpoints/
│   ├── mobilenet_baseline.pth   ← output of Step 1a (Kaggle → download here)
│   └── mobilenet_kd.pth         ← output of Step 1b (Kaggle → download here)
└── results/
    ├── accuracy.csv
    ├── mobilenet_baseline_training_log.csv
    ├── mobilenet_kd_training_log.csv
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
| 1a — Baseline training | `students/mobilenet/train_baseline.py` | not started | |
| 1b — KD training | `students/mobilenet/train_kd.py` | not started | |
| 2 — Grad-CAM (200) | `students/mobilenet/generate_gradcam.py` | not started | |
| 3 — Grad-CAM (full) | `students/mobilenet/generate_gradcam_full.py` | not started | |
| 4 — Score divergence | `shared/score_divergence.py` | not started | run after Step 3 |
| 5 — Summarize | `shared/summarize.py` | not started | run after Step 4 |

---

## Kaggle checkpoint dataset name

`kd-attention-checkpoints-mobilenet`

Upload `mobilenet_baseline.pth` and `mobilenet_kd.pth` here after training. The teacher checkpoint lives in the existing `kd-attention-checkpoints` dataset (already uploaded from the ResNet-18 experiment).

Kaggle input paths used by `generate_gradcam_full.py`:
```
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints/teacher_finetuned.pth
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-mobilenet/mobilenet_kd.pth
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-mobilenet/mobilenet_baseline.pth
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
!python students/mobilenet/train_baseline.py
!python students/mobilenet/train_kd.py
!python students/mobilenet/generate_gradcam_full.py
```

---

## Architecture notes for Claude Code

When generating new scripts for this student, the model loader always looks like:

```python
import torchvision.models as models
import torch.nn as nn

def load_mobilenet(path, num_classes):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()
```

And the GradCAM target layer is always:
```python
GradCAM(model=m, target_layers=[m.features[-1]])
```

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
MobileNetV2 student experiment. Teacher (ResNet-50) is already trained at
teacher/checkpoints/teacher_finetuned.pth.
[Describe what has been completed so far for this student.]

## Task
[One clearly scoped task.]

## Inputs
[Exact file paths this task depends on.]

## Expected outputs
[Exact output paths and formats — all under students/mobilenet/]

## Constraints
- All student outputs go under students/mobilenet/
- GradCAM target: model.features[-1]
- Head: model.classifier[1] = nn.Linear(...)
- Reproducibility: seed=42 everywhere
- Do not retrain the teacher
```
