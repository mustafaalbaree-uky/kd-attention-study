# Claude Code PM Guide — ResNet-18 Student

This file is the student-specific companion to `claude_code_pm_guide.md`. Read the main guide first for project overview, research question, and general workflow. This file contains everything specific to the ResNet-18 student experiment.

---

## Student identity

| Role | Model | Weights |
|---|---|---|
| Teacher | ResNet-50 | `teacher/checkpoints/teacher_finetuned.pth` |
| KD Student | ResNet-18 | `students/resnet18/checkpoints/resnet18_kd.pth` |
| Baseline Student | ResNet-18 | `students/resnet18/checkpoints/resnet18_baseline.pth` |

**Grad-CAM target layer:** `model.layer4[-1]` (last residual block, both student variants)

---

## Directory layout

```
students/resnet18/
├── train_baseline.py          ← Step 1a
├── train_kd.py                ← Step 1b
├── generate_gradcam.py        ← Step 2 (200-image sample, local)
├── generate_gradcam_full.py   ← Step 3 (full 3925, Kaggle)
├── checkpoints/
│   ├── resnet18_baseline.pth  ← output of Step 1a (Kaggle → download here)
│   └── resnet18_kd.pth        ← output of Step 1b (Kaggle → download here)
└── results/
    ├── accuracy.csv
    ├── resnet18_baseline_training_log.csv
    ├── resnet18_kd_training_log.csv
    ├── divergence_scores.csv
    ├── summary_stats.json
    ├── figures/               ← publication figures (3 PNGs)
    └── gradcam_full/
        ├── arrays/            ← 3925 .npz files
        └── figures/           ← 10 sample PNGs
```

---

## Implementation status

| Step | Script | Status | Notes |
|---|---|---|---|
| 1 — Teacher training | `teacher/train_teacher.py` | **DONE** | `teacher_finetuned.pth` exists |
| 1a — Baseline training | `students/resnet18/train_baseline.py` | **DONE** | val_acc logged |
| 1b — KD training | `students/resnet18/train_kd.py` | **DONE** | val_acc logged |
| 2 — Grad-CAM (200) | `students/resnet18/generate_gradcam.py` | **DONE** | arrays + figures exist |
| 3 — Grad-CAM (full) | `students/resnet18/generate_gradcam_full.py` | **DONE** | 3925 arrays in `gradcam_full/` |
| 4 — Score divergence | `shared/score_divergence.py` | **DONE** | `divergence_scores.csv` exists |
| 5 — Summarize | `shared/summarize.py` | **DONE** | `summary_stats.json` + figures exist |

---

## Kaggle checkpoint dataset name

`kd-attention-checkpoints`

Upload `resnet18_baseline.pth` and `resnet18_kd.pth` here after training. Teacher checkpoint lives in the same dataset.

Kaggle input paths used by `generate_gradcam_full.py`:
```
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints/teacher_finetuned.pth
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints/resnet18_kd.pth
/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints/resnet18_baseline.pth
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
!python students/resnet18/train_baseline.py
!python students/resnet18/train_kd.py
!python students/resnet18/generate_gradcam_full.py
```

---

## Key results (fill in after each run)

| Metric | Value |
|---|---|
| Teacher val_acc | |
| Baseline val_acc | |
| KD student val_acc | |
| Mean JS divergence — KD vs teacher | |
| Mean JS divergence — Baseline vs teacher | |
| Mann-Whitney U p-value (correct vs incorrect) | |

---

## Prompt template for next Claude Code task

```
## Context
ResNet-18 student experiment. Teacher (ResNet-50) is already trained.
[Describe what has been completed so far for this student.]

## Task
[One clearly scoped task.]

## Inputs
[Exact file paths this task depends on.]

## Expected outputs
[Exact output paths and formats.]

## Constraints
- All paths relative to students/resnet18/ or project root
- Reproducibility: seed=42 everywhere
- Do not retrain the teacher
```
