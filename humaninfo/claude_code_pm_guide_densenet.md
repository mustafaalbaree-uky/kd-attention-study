# Claude Code project management guide — DenseNet-121 student

This file lives in the project. Any new Claude chat working on the DenseNet-121 experiment should read this file first before doing anything. It contains the full project specification, the current implementation state, and the exact format for every prompt sent to Claude Code.

---

## Project overview (read this first)

**What we are building:**
An empirical study comparing the visual attention patterns of a teacher model and a student model trained via knowledge distillation (KD). We use Grad-CAM to visualize where each model "looks" on the same test images, then measure whether divergence in attention predicts student failure.

**Central research question:**
Does training with soft teacher labels produce greater Grad-CAM saliency alignment with the teacher than training on hard labels alone — and does saliency divergence predict student failure?

**Models (this experiment):**
- Teacher: ResNet-50, fine-tuned on ImageNette — checkpoint at `teacher/checkpoints/teacher_finetuned.pth`. Do not retrain.
- Student: DenseNet-121 — trained via knowledge distillation against the frozen teacher
- Baseline student: DenseNet-121 — identical architecture, trained on hard labels only (no teacher)

**Why DenseNet-121:**
DenseNet-121 uses dense connections where each layer receives feature maps from all preceding layers (~8M parameters). Its feature reuse pattern is structurally very different from both ResNet-50 (teacher) and the other students: the last dense block accumulates information across all prior layers, not just one residual skip. This makes the Grad-CAM alignment question more nuanced — the activation maps have different spatial character than ResNet maps.

**Architecture details:**
- Head replacement: `model.classifier = nn.Linear(model.classifier.in_features, num_classes)`
- Grad-CAM target layer: `model.features.denseblock4` (last dense block, before norm5 + global pooling)

**Dataset:**
ImageNette — a 10-class subset of ImageNet (224×224, ~1.4GB). Available on Hugging Face (`frgfm/imagenette`). Standard train/val split.

**Compute environment:**
Claude Code runs locally and owns the full codebase. Heavy training runs execute on Kaggle GPU. GitHub is the bridge.

Workflow per training step:
1. Claude Code writes the script locally
2. Human pushes to GitHub
3. Kaggle notebook clones the repo and runs the script on GPU
4. Outputs (checkpoints, CSVs, logs) are downloaded from Kaggle's output panel
5. Human places downloaded files into the correct local directory under `students/densenet/`
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
- DenseNet-121 students: `kd-attention-checkpoints-densenet`

**Deliverables from this experiment:**
- `students/densenet/results/accuracy.csv` — top-1 accuracy for teacher, KD student, baseline
- `students/densenet/results/gradcam_full/arrays/` — 3,925 .npz files (full ImageNette val set), keys: teacher, kd_student, baseline — maps summing to 1.0 plus metadata
- `students/densenet/results/gradcam_full/figures/` — 10 sample PNGs: 1×4 grid (original | teacher | KD student | baseline)
- `students/densenet/results/divergence_scores.csv` — per-image JS divergence and Spearman r for both students vs teacher, correctness flags
- `students/densenet/results/summary_stats.json` — mean JS divergence by outcome group, Mann-Whitney U results, accuracy numbers
- `students/densenet/results/figures/` — publication-ready figures (300 DPI)

**Key constraints:**
- All code must be reproducible (fixed random seed: 42 everywhere)
- All student outputs go under `students/densenet/`
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

## State tracker — DenseNet-121 student

[ ] Step 1a — Baseline training (DenseNet-121, hard labels only)
    Files to produce:
      - students/densenet/train_baseline.py (already written — run it on Kaggle)
      - students/densenet/checkpoints/densenet_baseline.pth
      - students/densenet/results/densenet_baseline_training_log.csv

[ ] Step 1b — KD training (DenseNet-121 student ← frozen ResNet-50 teacher)
    Files to produce:
      - students/densenet/train_kd.py (already written — run it on Kaggle)
      - students/densenet/checkpoints/densenet_kd.pth
      - students/densenet/results/densenet_kd_training_log.csv

[ ] Step 2 — Evaluation (accuracy, all three models on ImageNette val set)
    Files to produce:
      - students/densenet/results/accuracy.csv (teacher, kd student, baseline)

[ ] Step 3 — Grad-CAM generation (full ImageNette validation set, Kaggle)
    Files to produce:
      - students/densenet/generate_gradcam_full.py (already written — run it on Kaggle)
      - students/densenet/results/gradcam_full/arrays/ (3,925 .npz files)
      - students/densenet/results/gradcam_full/figures/ (10 sample PNGs)

[ ] Step 4 — Divergence scoring (JS divergence + Spearman, full validation set)
    Files to produce:
      - shared/score_divergence.py (may need path update for this student)
      - students/densenet/results/divergence_scores.csv

[ ] Step 5 — Summary stats, figures, and statistical significance test
    Files to produce:
      - students/densenet/results/summary_stats.json
      - students/densenet/results/figures/figure1_js_divergence_bar.png
      - students/densenet/results/figures/figure2_js_by_outcome.png
      - students/densenet/results/figures/figure3_spearman_distribution.png

---

## Kaggle input paths (for generate_gradcam_full.py)

```
CKPT_TEACHER  = "/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints/teacher_finetuned.pth"
CKPT_KD       = "/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-densenet/densenet_kd.pth"
CKPT_BASELINE = "/kaggle/input/datasets/mustafaalbaree/kd-attention-checkpoints-densenet/densenet_baseline.pth"
```

---

## Key results (fill in after each run)

| Metric | Value |
|---|---|
| Teacher val_acc | 0.9936 (from ResNet-18 experiment) |
| Baseline val_acc | |
| KD student val_acc | |
| Mean JS divergence — KD vs teacher | |
| Mean JS divergence — Baseline vs teacher | |
| Mann-Whitney U statistic | |
| Mann-Whitney p-value | |

---

## Rules for writing future prompts

1. **One task per prompt.** If the task description has the word "and" more than once, split it.
2. **Always specify exact file paths** for inputs and outputs. Claude Code does not guess — it does what it is told.
3. **Always include the verify step.** We do not move to the next step until the verify step passes.
4. **Never send a prompt without reading the state tracker.** The tracker prevents duplicate work and catches dependency errors.
5. **If Claude Code produces something unexpected**, bring it back to this chat before deciding how to proceed. Do not improvise instructions inside the Claude Code session.
6. **All outputs go under `students/densenet/`** — never write to the root results/ or shared checkpoints/.
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
