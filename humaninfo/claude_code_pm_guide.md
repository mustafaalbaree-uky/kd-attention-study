# Claude Code project management guide

This file lives in the project. Any new Claude chat in this project should read this file first before doing anything. It contains the full project specification, the current implementation state, and the exact format for every prompt sent to Claude Code.

---

## Project overview (read this first)

**What we are building:**
An empirical study comparing the visual attention patterns of a teacher model and a student model trained via knowledge distillation (KD). We use Grad-CAM to visualize where each model "looks" on the same test images, then measure whether divergence in attention predicts student failure.

**Central research question:**
Does knowledge distillation transfer visual reasoning strategy, or only classification accuracy?

**Models:**
- Teacher: ResNet-50, pre-trained on ImageNet (torchvision.models or Hugging Face)
- Student: ResNet-18 — trained via knowledge distillation against the frozen teacher
- Baseline student: ResNet-18 — identical architecture, trained on hard labels only (no teacher)

**Dataset:**
ImageNette — a 10-class subset of ImageNet (224×224, ~1.4GB). Available on Hugging Face (`frgfm/imagenette`). Standard train/val/test split.

**Compute environment:**
Claude Code runs locally and owns the full codebase. Heavy training runs execute on Kaggle GPU. GitHub is the bridge.

Workflow per training step:
1. Claude Code writes the script locally
2. Human pushes to GitHub
3. Kaggle notebook clones the repo and runs the script on GPU
4. Outputs (checkpoints, CSVs, logs) are downloaded from Kaggle's output panel
5. Human places downloaded files into the correct local results/ or checkpoints/ directory
6. Claude Code reads them and proceeds

Claude Code does not write Kaggle-specific notebook cells. It writes standard Python scripts that run anywhere. One boilerplate Kaggle cell (git clone + pip install) is added manually by the human at the top of each notebook.

**Deliverables from implementation:**
- `results/accuracy.csv` — top-1 accuracy for all three models
- `results/gradcam/` — side-by-side Grad-CAM figures (teacher | KD student | baseline student) for 200 test images
- `results/similarity_scores.csv` — per-image SSIM scores, predicted labels, true labels, correctness flags
- `results/summary_stats.json` — mean SSIM by outcome group, accuracy numbers
- All code, config files, and a README with exact reproduction steps

**Key constraints:**
- All code must be reproducible (fixed random seed everywhere)
- No private datasets — CIFAR-10 or Tiny ImageNet only (CIFAR-10 chosen for first experiment)
- Use Hugging Face `transformers` and/or `timm` for models — do not train from scratch
- Save config as a separate YAML or JSON file, not hardcoded in scripts

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
[How we will know this worked. E.g.: "Run the script and confirm it prints final test accuracy and saves the output file at results/accuracy.csv."]
```

---

## Implementation state tracker

Update this section after every Claude Code session. This is the single source of truth for where the project stands.

```
## Deprecated — Phase 1 (ViT/MobileViT/CIFAR-10)
See DEPRECATED_PHASE1.md for full record.

## Phase 2 (ResNet-50 → ResNet-18 / ImageNette)

[ ] Step 1 — Environment setup and model loading
    Files to produce:
      - requirements.txt (updated for Phase 2)
      - config.yaml (ResNet-50, ResNet-18, ImageNette, image_size=224, batch_size=64, seed=42)
      - model_check.py (updated for new models)
    Notes: —

[ ] Step 2 — KD training (ResNet-18 student vs. frozen ResNet-50 teacher)
    Files to produce:
      - train_kd.py
      - checkpoints/student_kd.pth
      - results/kd_training_log.csv
    Notes: —

[ ] Step 3 — Baseline student training (ResNet-18, hard labels only)
    Files to produce:
      - train_baseline.py
      - checkpoints/student_baseline.pth
      - results/baseline_training_log.csv
    Notes: —

[ ] Step 4 — Evaluation (accuracy, all three models)
    Files to produce:
      - results/accuracy.csv
    Notes: —

[ ] Step 5 — Grad-CAM generation (full 10k test set)
    Files to produce:
      - results/gradcam/ (teacher | KD student | baseline side-by-side figures)
    Notes: —

[ ] Step 6 — Divergence scoring (JS divergence + Spearman)
    Files to produce:
      - results/divergence_scores.csv (per-image JS divergence, Spearman r, labels, correctness flags)
    Notes: JS divergence requires normalizing Grad-CAM maps to sum to 1 before computing.
           Spearman computed on flattened, ranked map values.

[ ] Step 7 — Summary stats and figures
    Files to produce:
      - results/summary_stats.json
      - results/figures/ (group comparison plots)
    Notes: —
```

---
## Prompt 1

```
## Context
We are starting Phase 2 of a knowledge distillation research project. Phase 1 (ViT/MobileViT/CIFAR-10) is deprecated. This phase uses ResNet-50 as the teacher and ResNet-18 as the student and baseline, trained on ImageNette (10-class ImageNet subset, 224×224). No code exists yet for Phase 2.

## Task
Set up the project environment and verify that all three models load and run a forward pass without errors.

The three models are:
1. ResNet-50 — the teacher (pre-trained on ImageNet, from torchvision)
2. ResNet-18 — the KD student (same weights source, will be fine-tuned later)
3. ResNet-18 — the baseline student (identical to model 2 — load it the same way)

Replace each model's final fully connected layer with a 10-class head before the forward pass. Do not train anything. Just load, replace the head, run one forward pass on a random 224×224 tensor, and print the output shape of each model.

## Inputs
None — all models pulled from torchvision.

## Expected outputs
- `requirements.txt` — pinned versions for all dependencies (torch, torchvision, timm, datasets, Pillow, PyYAML, grad-cam, scikit-image, scipy, pandas, matplotlib, tqdm)
- `config.yaml` — must include: teacher model name, student model name, dataset name (`frgfm/imagenette`), image_size (224), batch_size (64), seed (42), num_classes (10), KD temperature T (4), KD loss weight alpha (0.7)
- `model_check.py` — loads all three models, replaces heads, runs one forward pass each, prints output shape

## Constraints
- Random seed: 42 everywhere (torch.manual_seed, numpy.random.seed, random.seed)
- Use torchvision.models for ResNet-50 and ResNet-18 (pretrained=True for teacher, pretrained=True for both students as starting weights)
- All hyperparameters read from config.yaml — nothing hardcoded in the script
- No training in this step

## Verify by
Run `python model_check.py` and confirm it prints three lines, one per model, each showing output shape `torch.Size([1, 10])`. No errors.
```


---
> ⚠️ Prompts 1 and 2 below are Phase 1 artifacts (ViT/MobileViT/CIFAR-10). They are kept for reference. Phase 2 prompts begin after the state tracker above.

## Prompt 1 (legacy)

```
## Context
We are starting a deep learning research project from scratch. There is no existing code. The project studies whether knowledge distillation (KD) preserves a teacher model's visual attention patterns, measured using Grad-CAM.

## Task
Set up the project environment and verify that we can load all three models we need:
1. A pre-trained ViT teacher from Hugging Face (google/vit-base-patch16-224 or the closest available)
2. A MobileViT-small student (from Hugging Face, apple/mobilevit-small)
3. A second instance of MobileViT-small — same architecture as the student, used as the baseline. The baseline will be trained on hard labels only (no teacher). Load it identically to model 2.

Run a single forward pass through each model on a random 224x224 tensor to confirm they load and run without errors. Print the output shape of each model. Do not train anything yet.

## Inputs
None — pulling all models from Hugging Face / timm.

## Expected outputs
- requirements.txt listing all dependencies with pinned versions
- config.yaml with model names, dataset choice (CIFAR-10), image size (224), batch size (64), random seed (42)
- setup.py or a single setup script that installs dependencies
- model_check.py that loads all three models and prints output shapes

## Constraints
- Random seed: 42 everywhere (torch.manual_seed, numpy, random)
- Use Hugging Face transformers for ViT
- Use timm for MobileViT and ResNet-18
- No training in this step — just loading and a forward pass
- Pin all dependency versions in requirements.txt

## Verify by
Run model_check.py and confirm it prints three lines, one per model, each showing the output tensor shape. No errors.
```

---

## Prompt 2 (legacy)

```
## Context
Step 1 is complete. All three models load and run cleanly. MobileViT-small is loaded from Hugging Face (apple/mobilevit-small), not timm. All models currently output (1, 1000) logits and will need their classifier heads replaced with 10-class heads for CIFAR-10. No training has been done yet.

## Task
Implement the knowledge distillation (KD) training loop. Replace each model's classifier head with a 10-class head, load CIFAR-10, and train the MobileViT-small student using soft labels from the frozen ViT teacher. Log accuracy and loss per epoch. Save the trained student weights.

## Inputs
- config.yaml (model names, dataset, image_size=224, batch_size=64, seed=42)
- model_check.py (reference for how models are loaded)
- All three models loaded exactly as in Step 1

## Expected outputs
- train_kd.py — the full KD training script
- checkpoints/student_kd.pth — saved weights of the KD-trained MobileViT-small student
- results/kd_training_log.csv — per-epoch columns: epoch, train_loss_total, train_loss_kd, train_loss_hard, train_acc, val_acc

## Constraints
- Random seed: 42 everywhere (torch.manual_seed, numpy, random, torch.cuda.manual_seed_all)
- Teacher is frozen — no gradients, no weight updates on ViT
- KD loss: weighted sum of cross-entropy on hard labels + KL divergence on soft labels (teacher temperature T=4, read from config.yaml)
- Add KD temperature T and KD loss weight alpha to config.yaml (start with T=4, alpha=0.7)
- Use Hugging Face transformers for ViT and MobileViT; replace classifier heads in-place before training
- Use standard CIFAR-10 train/val split (50k train, 10k test — use 10% of train as val)
- Images resized to 224x224 to match model expectations
- All hyperparameters read from config.yaml, not hardcoded

## Verify by
Run train_kd.py and confirm: (1) it prints per-epoch train loss and val accuracy, (2) val accuracy is climbing by epoch 5, (3) checkpoints/student_kd.pth exists at end of run, (4) results/kd_training_log.csv exists with one row per epoch.
```

---

## Rules for writing future prompts

1. **One task per prompt.** If the task description has the word "and" more than once, split it.
2. **Always specify exact file paths** for inputs and outputs. Claude Code does not guess — it does what it is told.
3. **Always include the verify step.** We do not move to the next step until the verify step passes.
4. **Never send a prompt without reading the state tracker.** The tracker prevents duplicate work and catches dependency errors.
5. **If Claude Code produces something unexpected**, bring it back to this chat before deciding how to proceed. Do not improvise instructions inside the Claude Code session.
6. **Config changes go in config.yaml, not in the prompt.** If we decide to change the temperature, batch size, or dataset, update the config file — do not hardcode it in the script.

---

## What to do when Claude Code produces an error

1. Copy the full error message back into this chat.
2. Do not ask Claude Code to fix it immediately.
3. Read the error here first. Understand what it is saying.
4. Write a corrective prompt using the template above.
5. Send the corrective prompt to Claude Code.

This prevents cascading errors where Claude Code "fixes" something in a way that breaks something else.

---

## What to do when a step is complete

1. Mark it in the state tracker (change `[ ]` to `[x]`).
2. Note the files produced and any important observations.
3. Come back to this chat and ask: "Step N is done. What is the next prompt?"
4. This chat will generate the next prompt in the template format above.
5. Review it, then send it to Claude Code.

## How this project chat works

**This chat (the project management chat)** is where the human and the senior Claude think together. We decide what the next implementation task is, we write the prompt for Claude Code, we review what Claude Code produced, and we decide what comes next. We do not write code here.

**Claude Code** (a separate terminal session) receives one prompt at a time. It implements, runs, and returns output. It does not make strategic decisions. It does not decide what to build next.

**The handoff:** Every Claude Code prompt is written here first, reviewed here, and only then sent. After Claude Code responds, the output comes back here for review before the next prompt is written.

---

## Operating rules for this chat (read this first in any new session)

These rules apply to every Claude session opened in this project that is related to coding using Claude code. Any new chat must read this file before doing anything else, then operate according to these rules without being reminded.

**When the human sends Claude Code output:**
- First determine: does this output complete the current step, or does it need a follow-up?
- If complete: update the state tracker and provide the next prompt.
- If incomplete: provide a follow-up prompt to continue the current step. Do not advance the tracker yet.

**Format rule — non-negotiable:**
All `.md` content provided for copy-pasting must be wrapped in a fenced block using four backticks (` ```` `) so that inner triple-backtick fences render correctly and the block remains copy-pasteable. Never let inner markdown syntax break out of the outer fence.

**Update rule:**
Never rewrite entire files. Only rewrite the specific subsections that changed. Tell the human exactly which section to replace and with what. Do not invent implementation choices — only record what Claude Code's output confirms. If a detail is not yet known, leave it ambiguous.

**Step completion rule:**
A step is complete when its verify condition passes. Do not mark a step complete until the human confirms the verify output matches what the prompt specified.

**Current step tracking:**
Always check the state tracker before writing any prompt. Never write a prompt for a step whose dependencies are not yet marked complete.
