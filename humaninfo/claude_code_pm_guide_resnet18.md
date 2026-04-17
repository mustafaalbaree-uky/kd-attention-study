# Claude Code project management guide

This file lives in the project. Any new Claude chat in this project should read this file first before doing anything. It contains the full project specification, the current implementation state, and the exact format for every prompt sent to Claude Code.

---

## Project overview (read this first)

**What we are building:**
An empirical study comparing the visual attention patterns of a teacher model and a student model trained via knowledge distillation (KD). We use Grad-CAM to visualize where each model "looks" on the same test images, then measure whether divergence in attention predicts student failure.

**Central research question:**
Does training with soft teacher labels produce greater Grad-CAM saliency alignment with the teacher than training on hard labels alone — and does saliency divergence predict student failure?

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
- `results/gradcam_full/arrays/` — 3,925 .npz files (full ImageNette validation set), one per image, keys: teacher, kd_student, baseline — 7×7 maps summing to 1.0 plus metadata
- `results/gradcam_full/figures/` — 3,925 PNGs: 1×4 grid (original | teacher | KD student | baseline)
- `results/divergence_scores.csv` — per-image JS divergence and Spearman r for both students vs teacher, correctness flags, 3,925 rows
- `results/summary_stats.json` — mean JS divergence by outcome group, Mann-Whitney U test results, accuracy numbers
- `results/figures/` — publication-ready figures (300 DPI)
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

## Phase 2 (ResNet-50 → ResNet-18 / ImageNette)

[x] Step 1 — Environment setup and model loading
    Files produced:
      - requirements.txt (pinned: torch, torchvision, timm, datasets, Pillow, PyYAML, grad-cam, scikit-image, scipy, pandas, matplotlib, tqdm)
      - config.yaml (resnet50 teacher, resnet18 student/baseline, frgfm/imagenette, image_size=224, batch_size=64, seed=42, num_classes=10, T=4, alpha=0.7)
      - model_check.py (loads all three from torchvision, replaces FC head with 10-class linear, forward pass confirmed)
      - .gitignore (data/ excluded)
    Notes: All three models print torch.Size([1, 10]). data/ removed from git tracking and gitignored. Repo pushed to GitHub successfully.

[x] Step 2 — KD training (ResNet-18 student vs. frozen ResNet-50 teacher)
    Files produced:
      - train_resnet18_kd.py (ResNet-50 frozen teacher, ResNet-18 student, KD loss: alpha*T²*KL + (1-alpha)*CE, SGD + CosineAnnealingLR, 30 epochs, tqdm, best-checkpoint saving)
      - checkpoints/resnet18_kd.pth — best val_acc: 0.9822 (saved locally)
      - results/resnet18_kd_training_log.csv — 30 epochs of training logs (saved locally)
    Notes: Training completed on Kaggle GPU T4 x2. Best checkpoint at epoch with val_acc=0.9822.

[x] Step 3 — Baseline student training (ResNet-18, hard labels only)
    Files produced:
      - train_resnet18_baseline.py
      - checkpoints/resnet18_baseline.pth — best val_acc: 0.9837
      - results/resnet18_baseline_training_log.csv
    Notes: val_acc (0.9837) is marginally higher than KD student (0.9822) — difference is
           within noise at this scale. Worth noting in the report as a baseline comparison.

[x] Step 3.5 — Teacher fine-tuning (ResNet-50 head on ImageNette) (and kd re-train)
    Files produced:
      - train_teacher.py
      - checkpoints/teacher_finetuned.pth
    Notes: Fine-tunes only the FC head (or a few top layers) of the ImageNet-pretrained
           ResNet-50 on ImageNette for a small number of epochs. train_kd.py updated to
           load teacher_finetuned.pth instead of raw torchvision weights.

[x] Step 4 — Evaluation (accuracy, all three models)
    Files produced:
      - evaluate.py
      - results/accuracy.csv
    Notes: teacher=0.9936, KD student=0.9819, baseline=0.9822. KD and baseline are
           essentially tied (~0.3pp difference). This is analytically useful — any saliency
           divergence found in Step 6 cannot be attributed to accuracy differences between
           the two students.

[DEPRECATED] Step 5 — Grad-CAM generation (200 stratified test images)
    Notes: Superseded by Step 5b. results/gradcam_200/ has been deleted locally.
           generate_gradcam.py retained in repo for reference only.

[x] Step 5b — Grad-CAM generation (full ImageNette validation set, Kaggle)
    Files produced:
      - generate_gradcam_full.py (pushed to main: efb46ae)
      - results/gradcam_full/arrays/ (3,925 .npz files: keys teacher, kd_student, baseline — 7×7 maps summing to 1.0 — plus scalar metadata true_label, teacher_pred, kd_pred, baseline_pred)
      - results/gradcam_full/figures/ (3,925 PNGs: 1×4 grid — original | teacher | KD student | baseline)
    Notes: Runs on the ImageNette val split (3,925 images — the correct evaluation split;
           training images are excluded from Grad-CAM analysis). Standalone script, all paths
           hardcoded for Kaggle. Checkpoints read from /kaggle/input/kd-attention-checkpoints/.
           Arrays and figures downloaded and placed locally. This is the primary Grad-CAM output.

[x] Step 6 — Divergence scoring (JS divergence + Spearman, full validation set)
    Files produced:
      - score_divergence.py (input path updated to results/gradcam_full/arrays/)
      - results/divergence_scores.csv (3,925 rows, no NaNs, JS in [0,1], Spearman in [-1,1])
    Notes: KD student mean JS distance from teacher: 0.1223. Baseline mean JS distance from
           teacher: 0.1359. KD student shows consistently lower divergence from teacher —
           consistent with hypothesis. Progress printed every 200 images.
    ⚠️ NEEDS UPDATE: SSIM columns (ssim_teacher_kd, ssim_teacher_baseline) not yet added.
           divergence_scores.csv must be regenerated after Prompt 9.

[ ] Step 6b — Add SSIM to divergence scoring and regenerate summary stats
    Files to produce:
      - score_divergence.py (updated — add ssim_teacher_kd, ssim_teacher_baseline columns)
      - results/divergence_scores.csv (3,925 rows — now with SSIM columns)
      - summarize.py (updated — include SSIM means in summary_stats.json)
      - results/summary_stats.json (regenerated)
      - results/figures/ (figures 1–3 regenerated; figure4_ssim_by_outcome.png added)

[x] Step 7 — Summary stats, figures, and statistical significance test
    Files produced:
      - summarize.py (updated with Mann-Whitney U test, alternative='less')
      - results/summary_stats.json (JS means, stds, outcome-group breakdown,
        mann_whitney_u_statistic=6162446, mann_whitney_p_value=1.97e-53)
      - results/figures/figure1_js_divergence_bar.png (regenerated, 300 DPI)
      - results/figures/figure2_js_by_outcome.png (regenerated, 300 DPI)
      - results/figures/figure3_spearman_distribution.png (regenerated, 300 DPI)
    Notes: Mann-Whitney U=6,162,446, p=1.97×10⁻⁵³ — extremely strong evidence that KD
           student JS divergence is statistically lower than baseline (one-tailed test in
           direction of hypothesis). This is the headline statistical result for the paper.
           Figures regenerated against full 3,925-image validation set.
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
## Prompt 2

```
## Context
Phase 2, Step 1 is complete. All three models (ResNet-50 teacher, ResNet-18 KD student, ResNet-18 baseline student) load and run forward passes correctly. config.yaml exists with all hyperparameters. The repo is on GitHub at https://github.com/mustafaalbaree-uky/kd-attention-study.git. This script will run on Kaggle GPU — ImageNette will be downloaded from Hugging Face at runtime.

## Task
Implement the knowledge distillation training loop. Train the ResNet-18 KD student using soft labels from the frozen ResNet-50 teacher on ImageNette. Log accuracy and loss per epoch. Save the trained student weights.

## Inputs
- config.yaml (teacher: resnet50, student: resnet18, dataset: frgfm/imagenette, image_size=224, batch_size=64, seed=42, num_classes=10, T=4, alpha=0.7)
- model_check.py (reference for how models are loaded and heads are replaced)

## Expected outputs
- `train_kd.py` — the full KD training script
- `checkpoints/student_kd.pth` — saved weights of the KD-trained ResNet-18 student
- `results/kd_training_log.csv` — per-epoch columns: epoch, train_loss_total, train_loss_kd, train_loss_hard, train_acc, val_acc

## Constraints
- Random seed: 42 everywhere (torch.manual_seed, numpy.random.seed, random.seed, torch.cuda.manual_seed_all)
- Teacher is fully frozen — no gradients, no weight updates on ResNet-50
- KD loss: weighted sum of (1) cross-entropy on hard labels and (2) KL divergence between student softened output and teacher softened output, both computed at temperature T=4. Alpha and T read from config.yaml
- Use torchvision.models for both models, replace FC heads with 10-class linear layers in-place before training
- Load ImageNette via Hugging Face datasets (`frgfm/imagenette`), standard train/validation split
- Apply standard ImageNet preprocessing: Resize(224), CenterCrop(224), ToTensor(), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- All hyperparameters read from config.yaml — nothing hardcoded
- Use tqdm for epoch progress
- Save checkpoint only if val_acc improves (best model checkpoint)

## Verify by
Confirm `train_kd.py` exists and that it imports without errors by running `python -c "import train_kd"` — no training should execute. Then push to GitHub and run on Kaggle.
```

---
## Prompt 3

```
## Context
Phase 2, Step 2 is complete. The KD-trained ResNet-18 student (student_kd.pth) achieved val_acc=0.9822 on ImageNette after 30 epochs. We now need a baseline student — an identical ResNet-18 architecture trained on hard labels only, with no teacher. This baseline isolates KD-specific effects from effects of reduced model capacity.

## Task
Implement the baseline student training loop. Train a ResNet-18 on ImageNette using only cross-entropy loss against hard labels. No teacher, no KD loss. Everything else — architecture, dataset, preprocessing, optimizer, scheduler, epochs — must be identical to train_kd.py so the two runs are directly comparable.

## Inputs
- config.yaml (teacher: resnet50, student: resnet18, dataset: frgfm/imagenette, image_size=224, batch_size=64, seed=42, num_classes=10, T=4, alpha=0.7)
- train_kd.py (reference for model loading, dataset loading, preprocessing, optimizer, scheduler)

## Expected outputs
- `train_baseline.py` — the full baseline training script
- `checkpoints/student_baseline.pth` — best checkpoint by val_acc
- `results/baseline_training_log.csv` — per-epoch columns: epoch, train_loss, train_acc, val_acc

## Constraints
- Random seed: 42 everywhere (torch.manual_seed, numpy.random.seed, random.seed, torch.cuda.manual_seed_all)
- Loss: cross-entropy on hard labels only — no KD loss, no teacher involved at all
- Same optimizer: SGD with same hyperparameters as train_kd.py
- Same scheduler: CosineAnnealingLR with same hyperparameters as train_kd.py
- Same architecture: ResNet-18, FC head replaced with 10-class linear layer
- Same dataset: frgfm/imagenette, same train/val split, same preprocessing
- Same number of epochs: 30
- All hyperparameters read from config.yaml — nothing hardcoded
- Use tqdm for epoch progress
- Save checkpoint only if val_acc improves (best model checkpoint)

## Verify by
Run `python -c "import train_baseline"` and confirm it imports without errors. Then push to GitHub and run on Kaggle.
```
---
## Prompt 4

```
## Context
Phase 2, Steps 1–3.5 are complete. We have three trained checkpoints: checkpoints/teacher_finetuned.pth (ResNet-50, fine-tuned on ImageNette), checkpoints/resnet18_kd.pth (ResNet-18, KD-trained, best val_acc=0.9822), and checkpoints/resnet18_baseline.pth (ResNet-18, hard labels only, best val_acc=0.9837). We now need a formal evaluation of all three models on the held-out test set to produce the accuracy row of our results table.

## Task
Write an evaluation script that loads all three checkpoints, runs each on the ImageNette test split, and records top-1 accuracy per model. This is evaluation only — no training, no Grad-CAM yet.

## Inputs
- config.yaml (resnet50 teacher, resnet18 student/baseline, frgfm/imagenette, image_size=224, batch_size=64, seed=42, num_classes=10)
- checkpoints/teacher_finetuned.pth
- checkpoints/resnet18_kd.pth
- checkpoints/resnet18_baseline.pth

## Expected outputs
- `evaluate.py` — the evaluation script
- `results/accuracy.csv` — three rows, one per model, with columns: model_name, checkpoint, test_accuracy

The CSV should look exactly like this:
```
model_name,checkpoint,test_accuracy
teacher_resnet50,checkpoints/teacher_finetuned.pth,0.XXXX
student_kd_resnet18,checkpoints/resnet18_kd.pth,0.XXXX
student_baseline_resnet18,checkpoints/resnet18_baseline.pth,0.XXXX
```
---
## Prompt 5

```
## Context
Phase 2, Step 5 is complete. We have 200 .npz files in results/gradcam/arrays/, each containing three 7×7 Grad-CAM maps (teacher, kd_student, baseline) normalized to sum to 1.0, plus per-image prediction metadata (true_label, teacher_pred, kd_pred, baseline_pred). We now need to compute divergence scores between teacher and each student for every image, and attach correctness flags so we can correlate divergence with failure in Step 7.

## Task
Write a scoring script that loads all 200 .npz files, computes two divergence metrics between the teacher map and each student map, and saves a single CSV with one row per image.

## Inputs
- results/gradcam/arrays/ — 200 .npz files from Step 5
- config.yaml

## Expected outputs
- `score_divergence.py` — the scoring script
- `results/divergence_scores.csv` — one row per image, with columns:
    filename, true_label,
    teacher_pred, kd_pred, baseline_pred,
    teacher_correct, kd_correct, baseline_correct,
    js_teacher_kd, js_teacher_baseline,
    spearman_teacher_kd, spearman_teacher_baseline

## Constraints
- Jensen-Shannon divergence: use scipy.spatial.distance.jensenshannon — it returns the JS distance (square root of JS divergence), which is fine; just be consistent. Do not reimplement it manually.
- Spearman rank correlation: use scipy.stats.spearmanr on the flattened (49-element) teacher and student map arrays
- correctness flags: 1 if predicted label == true label, 0 otherwise
- If a map is all zeros (should not happen after Step 5 normalization, but guard anyway), skip the JS computation and record NaN for that row
- No GPU needed — this is pure numpy/scipy
- All file I/O uses the arrays/ directory path read from config.yaml if present, otherwise hardcoded default of results/gradcam/arrays/
- Print progress every 50 images

## Verify by
Run `python score_divergence.py` and confirm: (1) results/divergence_scores.csv exists with exactly 200 rows plus a header, (2) all js_ columns are between 0 and 1, (3) all spearman_ columns are between -1 and 1, (4) print the mean js_teacher_kd and mean js_teacher_baseline to stdout as a sanity check.

## Constraints
- Random seed: 42 everywhere (torch.manual_seed, numpy.random.seed, random.seed)
- Use the ImageNette test split from frgfm/imagenette — NOT the validation split used during training
- Same preprocessing as training: Resize(256), CenterCrop(224), ToTensor(), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- Load each model from torchvision.models with the same head-replacement logic as the training scripts — then load the checkpoint weights
- All models evaluated in eval mode (model.eval(), torch.no_grad())
- All hyperparameters read from config.yaml — nothing hardcoded
- Print each model's accuracy to stdout as it completes, then save the CSV

## Verify by
Run `python evaluate.py` and confirm: (1) it prints three accuracy lines to stdout, (2) results/accuracy.csv exists with exactly three rows plus a header, (3) teacher accuracy is noticeably higher than or competitive with both students (sanity check that the checkpoints loaded correctly — if teacher accuracy is below 0.90, something is wrong).
```

---
## Prompt 6

```
## Context
Phase 2 Steps 1-6 are complete. We have divergence_scores.csv with 200 rows containing per-image JS divergence and Spearman r for both KD student and baseline against the teacher, plus correctness flags and labels. Key numbers: KD student mean JS = 0.121, baseline mean JS = 0.136. Step 7 produces the summary stats and figures that go directly into the paper.

## Task
Write a script that reads divergence_scores.csv and produces two things: a JSON file of summary statistics and a set of figures for the paper.

Summary statistics to compute and save to results/summary_stats.json:
- Mean and std JS divergence for KD student vs teacher, across all 200 images
- Mean and std JS divergence for baseline vs teacher, across all 200 images
- Mean and std Spearman r for both, across all 200 images
- For the KD student: mean JS divergence split by outcome group — (1) both correct, (2) student wrong + teacher correct, (3) both wrong
- Same outcome group split for baseline
- Top-1 accuracy for all three models (read from results/accuracy.csv)

Figures to produce and save to results/figures/:
- figure1_js_divergence_bar.png — grouped bar chart: mean JS divergence (KD student vs baseline) with error bars (std). Two bars side by side, one group labeled "vs Teacher." Clean, paper-ready.
- figure2_js_by_outcome.png — grouped bar chart: mean JS divergence for KD student split by outcome group (both correct / student wrong+teacher correct / both wrong). Error bars. This is the correlation-with-failure plot.
- figure3_spearman_distribution.png — overlaid histogram or KDE: distribution of per-image Spearman r for KD student (blue) and baseline (orange). Shows spread, not just mean.

## Inputs
- results/divergence_scores.csv
- results/accuracy.csv

## Expected outputs
- summarize.py
- results/summary_stats.json
- results/figures/figure1_js_divergence_bar.png
- results/figures/figure2_js_by_outcome.png
- results/figures/figure3_spearman_distribution.png

## Constraints
- Random seed: 42 everywhere
- All hyperparameters and file paths read from config.yaml — nothing hardcoded
- Figures must be publication quality: 300 DPI, no chart junk, axis labels, legend, title
- Use matplotlib only — no seaborn
- All figures saved with tight_layout()

## Verify by
Run python summarize.py and confirm: (1) results/summary_stats.json exists and contains all expected keys, (2) all three figures exist in results/figures/ and are non-empty files, (3) the mean JS numbers in the JSON match 0.121 for KD student and 0.136 for baseline within rounding.
```

---
## Prompt 7

```
## Context
Phase 2 Step 5b is complete. We have 3,925 .npz files in results/gradcam_full/arrays/, each containing three 7×7 Grad-CAM maps (teacher, kd_student, baseline) normalized to sum to 1.0, plus per-image prediction metadata. The previous divergence_scores.csv was computed on only 200 images and is now retired. We need to rerun scoring on the full 3,925-image validation set.

## Task
Update score_divergence.py to read from results/gradcam_full/arrays/ instead of results/gradcam_200/arrays/, then rerun it to produce a new divergence_scores.csv with 3,925 rows.

## Inputs
- results/gradcam_full/arrays/ — 3,925 .npz files
- config.yaml
- score_divergence.py (existing script — change the input path only, touch nothing else)

## Expected outputs
- score_divergence.py (updated input path)
- results/divergence_scores.csv — 3,925 rows, columns: filename, true_label, teacher_pred, kd_pred, baseline_pred, teacher_correct, kd_correct, baseline_correct, js_teacher_kd, js_teacher_baseline, spearman_teacher_kd, spearman_teacher_baseline

## Constraints
- Random seed: 42 everywhere
- Do not change the scoring logic — only the input directory path
- Print progress every 200 images
- Print mean js_teacher_kd and mean js_teacher_baseline to stdout when done

## Verify by
Run python score_divergence.py and confirm: (1) results/divergence_scores.csv has exactly 3,925 rows plus header, (2) all js_ columns are in [0,1], (3) all spearman_ columns are in [-1,1], (4) mean JS values print to stdout.
```

---
## Prompt 8

```
## Context
Phase 2 Step 6 is complete. results/divergence_scores.csv now has 3,925 rows. We need to regenerate all summary statistics and figures against the full validation set, and add a Mann-Whitney U test comparing the KD and baseline JS distributions to give us a defensible p-value for the headline result.

## Task
Update summarize.py to (1) read from the 3,925-row divergence_scores.csv, (2) add a Mann-Whitney U test between js_teacher_kd and js_teacher_baseline columns, and (3) regenerate all three figures and summary_stats.json.

## Inputs
- results/divergence_scores.csv (3,925 rows)
- results/accuracy.csv

## Expected outputs
- summarize.py (updated)
- results/summary_stats.json — same structure as before, plus new keys: mann_whitney_u_statistic, mann_whitney_p_value
- results/figures/figure1_js_divergence_bar.png (regenerated)
- results/figures/figure2_js_by_outcome.png (regenerated)
- results/figures/figure3_spearman_distribution.png (regenerated)

## Constraints
- Random seed: 42 everywhere
- Mann-Whitney U test: use scipy.stats.mannwhitneyu with alternative='less' (one-tailed — testing that KD JS is lower than baseline JS, the direction of the hypothesis)
- Print the U statistic and p-value to stdout
- All figures 300 DPI, tight_layout(), matplotlib only

## Verify by
Run python summarize.py and confirm: (1) results/summary_stats.json contains mann_whitney_u_statistic and mann_whitney_p_value keys, (2) all three figures are regenerated, (3) U statistic and p-value print to stdout.
```

---
## Prompt 9

```
## Context
Steps 1–7 are complete for the ResNet-18 experiment. results/divergence_scores.csv has 3,925 rows with JS divergence and Spearman r columns. We identified a gap: SSIM (Structural Similarity Index) was specified in the research design as a spatially-aware complement to JS divergence but was never added to the scoring script. We need to add it now and regenerate all downstream outputs.

## Task
Two changes in sequence:

1. Update score_divergence.py to add SSIM between teacher and each student map for every image, appending two new columns to divergence_scores.csv.

2. Update summarize.py to include SSIM means in summary_stats.json and produce one new figure (SSIM by outcome group, matching the structure of figure2_js_by_outcome.png). Regenerate all outputs.

## Inputs
- students/resnet18/results/gradcam_full/arrays/ — 3,925 .npz files (keys: teacher, kd_student, baseline — 7×7 maps summing to 1.0)
- students/resnet18/results/divergence_scores.csv (existing — will be overwritten with SSIM columns added)
- students/resnet18/results/accuracy.csv

## Expected outputs
- shared/score_divergence.py (updated)
- students/resnet18/results/divergence_scores.csv — 3,925 rows, same columns as before plus: ssim_teacher_kd, ssim_teacher_baseline
- shared/summarize.py (updated)
- students/resnet18/results/summary_stats.json — same structure as before plus: mean_ssim_kd, std_ssim_kd, mean_ssim_baseline, std_ssim_baseline, and ssim broken out by outcome group for both students
- students/resnet18/results/figures/figure1_js_divergence_bar.png (regenerated — unchanged)
- students/resnet18/results/figures/figure2_js_by_outcome.png (regenerated — unchanged)
- students/resnet18/results/figures/figure3_spearman_distribution.png (regenerated — unchanged)
- students/resnet18/results/figures/figure4_ssim_by_outcome.png (new — grouped bar chart: mean SSIM for KD student and baseline split by outcome group, same structure as figure2)

## Constraints
- Random seed: 42 everywhere
- SSIM implementation: use skimage.metrics.structural_similarity
- Maps are 7×7 and normalized to sum to 1.0 — reshape from flat array to (7,7) before passing to SSIM
- Use data_range=1.0 and win_size=7 (maps are exactly 7×7 — this is the maximum window size and is correct)
- SSIM is bounded −1 to 1; higher = more similar
- Do not change the JS or Spearman logic — add SSIM as additional columns only
- Print mean ssim_teacher_kd and mean ssim_teacher_baseline to stdout when score_divergence.py finishes
- All figures 300 DPI, tight_layout(), matplotlib only
- Progress printed every 200 images in score_divergence.py

## Verify by
Run python shared/score_divergence.py then python shared/summarize.py and confirm: (1) divergence_scores.csv has the two new SSIM columns with values in [−1, 1], (2) summary_stats.json contains mean_ssim_kd and mean_ssim_baseline, (3) figure4_ssim_by_outcome.png exists in students/resnet18/results/figures/.
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

## Formatting rule for prompt blocks — non-negotiable

Any time a new Claude instance in this project writes a new prompt (whether Prompt N, a corrective follow-up, or a revised version of an existing prompt), it must wrap the entire prompt block using **four backticks**, not three. This is because the prompt body itself contains triple-backtick fences for code, and a three-backtick outer fence will break rendering and make the block un-copy-pasteable.

The correct format is always:


---

Prompt [N]
```

## Context

## Task

## Inputs

## Expected outputs

## Constraints

## Verify by
```
When adding a new prompt to the state tracker section, the format is:

`---` on its own line, then `## Prompt [N]` as a heading, then the four-backtick fenced block. Never use three backticks for prompt blocks. Never wrap a prompt in a code block labeled `markdown` or any other language tag.

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
