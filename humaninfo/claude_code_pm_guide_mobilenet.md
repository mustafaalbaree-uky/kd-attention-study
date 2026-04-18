# Claude Code project management guide — MobileNetV2 student

This file lives in the project. Any new Claude chat working on the MobileNetV2 experiment should read this file first before doing anything. It contains the full project specification, the current implementation state, and the exact format for every prompt sent to Claude Code.

---

## Project overview (read this first)

**What we are building:**
An empirical study comparing the visual attention patterns of a teacher model and a student model trained via knowledge distillation (KD). We use Grad-CAM to visualize where each model "looks" on the same test images, then measure whether divergence in attention predicts student failure.

**Central research question:**
Does training with soft teacher labels produce greater Grad-CAM saliency alignment with the teacher than training on hard labels alone — and does saliency divergence predict student failure?

**Models (this experiment):**
- Teacher: ResNet-50, fine-tuned on ImageNette — checkpoint at `teacher/checkpoints/teacher_finetuned.pth`. Do not retrain.
- Student: MobileNetV2 — trained via knowledge distillation against the frozen teacher
- Baseline student: MobileNetV2 — identical architecture, trained on hard labels only (no teacher)

**Why MobileNetV2:**
MobileNetV2 uses inverted residual blocks with depthwise separable convolutions — roughly 4× smaller than ResNet-18 by parameter count (~3.4M vs ~11.7M). Testing it as a student lets us ask whether a structurally different, more compressed student still benefits from soft teacher labels in terms of Grad-CAM alignment.

**Architecture details:**
- Head replacement: `model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)`
- Grad-CAM target layer: `model.features[-1]` (last ConvBNActivation block before global average pooling)

**Dataset:**
ImageNette — a 10-class subset of ImageNet (224×224, ~1.4GB). Available on Hugging Face (`frgfm/imagenette`). Standard train/val split.

**Compute environment:**
Claude Code runs locally and owns the full codebase. Heavy training runs execute on Kaggle GPU. GitHub is the bridge.

Workflow per training step:
1. Claude Code writes the script locally
2. Human pushes to GitHub
3. Kaggle notebook clones the repo and runs the script on GPU
4. Outputs (checkpoints, CSVs, logs) are downloaded from Kaggle's output panel
5. Human places downloaded files into the correct local directory under `students/mobilenet/`
6. Claude Code reads them and proceeds

Claude Code does not write Kaggle-specific notebook cells. It writes standard Python scripts that run anywhere. One boilerplate Kaggle cell (git clone + pip install) is added manually by the human at the top of each notebook.

**Kaggle boilerplate cell:**
```python
!git clone https://github.com/mustafaalbaree-uky/kd-attention-study.git
!pip install -q pytorch-grad-cam datasets tqdm
import os; os.chdir("kd-attention-study")
```

**Kaggle checkpoint dataset names:**
- Teacher: `kd-attention-checkpoints` (already uploaded — do not re-upload)
- MobileNetV2 students: `kd-attention-checkpoints-mobilenet`

**Deliverables from this experiment:**
- `students/mobilenet/results/accuracy.csv` — top-1 accuracy for teacher, KD student, baseline
- `students/mobilenet/results/gradcam_full/arrays/` — 3,925 .npz files (full ImageNette val set), keys: teacher, kd_student, baseline — 7×7 maps summing to 1.0 plus metadata
- `students/mobilenet/results/gradcam_full/figures/` — 10 sample PNGs: 1×4 grid (original | teacher | KD student | baseline)
- `students/mobilenet/results/divergence_scores.csv` — per-image JS divergence and Spearman r for both students vs teacher, correctness flags
- `students/mobilenet/results/summary_stats.json` — mean JS divergence by outcome group, Mann-Whitney U results, accuracy numbers
- `students/mobilenet/results/figures/` — publication-ready figures (300 DPI)

**Key constraints:**
- All code must be reproducible (fixed random seed: 42 everywhere)
- All student outputs go under `students/mobilenet/`
- Teacher checkpoint is shared — never retrain it
- Scripts use `Path(__file__).parent` for self-contained paths

---

## How this project chat works

**This chat (the project management chat)** is where the human and the senior Claude think together. We decide what the next implementation task is, we write the prompt for Claude Code, we review what Claude Code produced, and we decide what comes next. We do not write code here.

**Claude Code** (a separate terminal session) receives one prompt at a time. It implements, runs, and returns output. It does not make strategic decisions. It does not decide what to build next.

**The handoff:** Every Claude Code prompt is written here first, reviewed here, and only then sent. After Claude Code responds, the output comes back here for review before the next prompt is written.

---

## Prompt format for Claude Code

Every prompt sent to Claude Code follows this exact template. Do not deviate from it — consistency is what makes this workflow reliable.

```
## Context
[1–3 sentences describing where we are in the project and what has already been done.]

## Task
[A single, clearly scoped task. One thing. If it feels like two things, it is two prompts.]

## Inputs
[List every file, model, or dataset this task depends on. Be explicit about paths.]

## Expected outputs
[List every file this task should produce, with the exact path and format.]

## Constraints
- Fixed random seed: 42 everywhere
- Use [specific library] for [specific thing]
- Save config to config.yaml, not hardcoded
- [Any other constraint specific to this task]

## Verify by
[How we will know this worked. E.g.: "Run the script and confirm it prints final val accuracy and saves the checkpoint."]
```

---

## State tracker — MobileNetV2 student

[x] Step 1a — Baseline training (MobileNetV2, hard labels only)
    Files produced:
      - students/mobilenet/train_baseline.py
      - students/mobilenet/checkpoints/mobilenet_baseline.pth
      - students/mobilenet/results/mobilenet_baseline_training_log.csv

[x] Step 1b — KD training (MobileNetV2 student ← frozen ResNet-50 teacher)
    Files produced:
      - students/mobilenet/train_kd.py
      - students/mobilenet/checkpoints/mobilenet_kd.pth
      - students/mobilenet/results/mobilenet_kd_training_log.csv

[x] Step 2 — Evaluation (accuracy, all three models on ImageNette val set)
    Files produced:
      - students/mobilenet/results/accuracy.csv

[x] Step 3 — Grad-CAM generation (full ImageNette validation set, Kaggle)
    Files produced:
      - students/mobilenet/generate_gradcam_full.py
      - students/mobilenet/results/gradcam_full/arrays/ (3,925 .npz files)
      - students/mobilenet/results/gradcam_full/figures/ (10 sample PNGs)

[x] Step 4 — Divergence scoring (JS divergence + Spearman + SSIM, full validation set)
    Files produced:
      - students/mobilenet/results/divergence_scores.csv
        Columns: filename, true_label, teacher_pred, kd_pred, baseline_pred,
                 teacher_correct, kd_correct, baseline_correct,
                 js_teacher_kd, js_teacher_baseline,
                 spearman_teacher_kd, spearman_teacher_baseline,
                 ssim_teacher_kd, ssim_teacher_baseline

[x] Step 5 — Summary stats, figures, and statistical significance test
    Files produced:
      - students/mobilenet/results/summary_stats.json
      - students/mobilenet/results/figures/figure1_js_divergence_bar.png
      - students/mobilenet/results/figures/figure2_js_by_outcome.png
      - students/mobilenet/results/figures/figure3_spearman_distribution.png
      - students/mobilenet/results/figures/figure4_ssim_by_outcome.png

[ ] Step 6 — Floor computation (seed-43 baseline training, Grad-CAM, scoring)
    Files to produce:
      - students/mobilenet/checkpoints/mobilenet_baseline_seed43.pth
      - students/mobilenet/results/mobilenet_baseline_seed43_training_log.csv
      - students/mobilenet/results/gradcam_full/arrays_seed43/ (3,925 .npz files, keys: baseline_seed42, baseline_seed43)
      - students/mobilenet/results/floor_scores.csv
      - students/mobilenet/results/summary_stats.json (updated with floor keys)

---

## Kaggle input paths (for generate_gradcam_full.py)

```
CKPT_TEACHER  = "/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints/teacher_finetuned.pth"
CKPT_KD       = "/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-mobilenet/mobilenet_kd.pth"
CKPT_BASELINE = "/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-mobilenet/mobilenet_baseline.pth"
```

---

## Key results (fill in after each run)

| Metric | Value |
|---|---|
| Teacher val_acc | 0.9936 |
| Baseline val_acc | 0.9819 |
| KD student val_acc | 0.9829 |
| Mean JS divergence — KD vs teacher | 0.1366 (std 0.0518) |
| Mean JS divergence — Baseline vs teacher | 0.1478 (std 0.0557) |
| Mean Spearman r — KD vs teacher | 0.8390 (std 0.1592) |
| Mean Spearman r — Baseline vs teacher | 0.8179 (std 0.1680) |
| Mean SSIM — KD vs teacher | 0.9519 (std 0.0525) |
| Mean SSIM — Baseline vs teacher | 0.9449 (std 0.0567) |
| Mann-Whitney U statistic | 6,534,331.0 |
| Mann-Whitney p-value | 1.31e-31 |

---

## Rules for writing future prompts

1. **One task per prompt.** If the task description has the word "and" more than once, split it.
2. **Always specify exact file paths** for inputs and outputs. Claude Code does not guess — it does what it is told.
3. **Always include the verify step.** We do not move to the next step until the verify step passes.
4. **Never send a prompt without reading the state tracker.** The tracker prevents duplicate work and catches dependency errors.
5. **If Claude Code produces something unexpected**, bring it back to this chat before deciding how to proceed. Do not improvise instructions inside the Claude Code session.
6. **All outputs go under `students/mobilenet/`** — never write to the root results/ or shared checkpoints/.
7. **Do not retrain the teacher** — `teacher/checkpoints/teacher_finetuned.pth` is shared and already trained.

---

## Formatting rule for prompt blocks — non-negotiable

Any time a new Claude instance in this project writes a new prompt, it must wrap the entire prompt block using **four backticks**, not three. This is because the prompt body itself contains triple-backtick fences for code, and a three-backtick outer fence will break rendering.

---

## What to do when Claude Code produces an error

1. Copy the full error message back into this chat.
2. Do not ask Claude Code to fix it immediately.
3. Read the error here first. Understand what it is saying.
4. Write a corrective prompt using the template above.
5. Send the corrective prompt to Claude Code.

---

## What to do when a step is complete

1. Mark it in the state tracker (change `[ ]` to `[x]`).
2. Note the files produced and any important observations (val_acc, mean JS, etc.).
3. Come back to this chat and ask: "Step N is done. What is the next prompt?"
4. This chat will generate the next prompt in the template format above.
5. Review it, then send it to Claude Code.

---

## Operating rules for this chat (read this first in any new session)

These rules apply to every Claude session opened in this project. Any new chat must read this file before doing anything else, then operate according to these rules without being reminded.

**When the human sends Claude Code output:**
- First determine: does this output complete the current step, or does it need a follow-up?
- If complete: update the state tracker and provide the next prompt.
- If incomplete: provide a follow-up prompt to continue the current step. Do not advance the tracker yet.

**Update rule:**
Never rewrite entire files. Only rewrite the specific subsections that changed. Tell the human exactly which section to replace and with what.

**Step completion rule:**
A step is complete when its verify condition passes. Do not mark a step complete until the human confirms the verify output matches what the prompt specified.
