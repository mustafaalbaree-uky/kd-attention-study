# Research design — first draft
**Project:** Does knowledge distillation preserve visual reasoning?
**Architecture:** ResNet-50 (teacher) → ResNet-18 (student) — pure CNN pipeline
**Dataset:** ImageNette (10-class ImageNet subset, 224×224)
**Course:** Deep learning (599 section) · 2-person team
**Status:** Pre-implementation draft — to be revised after experiments complete

---

## 1. The problem

Knowledge distillation (KD) is a widely used technique for compressing large neural networks into smaller, faster ones. The field measures its success almost exclusively through **accuracy transfer** — does the student model match the teacher's classification performance?

But accuracy is a blunt instrument. A model can get the right answer for the wrong reason. Two models can agree on a label while attending to completely different parts of an image to reach that decision.

**The gap:** We do not know whether KD transfers *how* a teacher model reasons visually, or merely *what* it concludes. Prior work does not systematically compare the internal visual focus of teacher and student models on matched inputs.

---

## 2. Research question

> Does a student model trained via knowledge distillation attend to the same image regions as its teacher — and when it diverges, does that divergence predict when the student will fail?

This breaks into two sub-questions:

- **2a.** Do teacher and student produce similar Grad-CAM attention maps on images they both classify correctly?
- **2b.** Is attention divergence between teacher and student correlated with student classification errors?

---

## 3. Hypothesis and null hypothesis

In traditional science, a hypothesis is a testable prediction. In AI research, this is sometimes called a **claim** — what you assert your experiment will demonstrate. The null hypothesis is what you assume is true until your results give you reason to reject it.

### Hypothesis (our claim)
Knowledge distillation transfers output accuracy but does **not** reliably transfer visual reasoning strategy. Specifically:

- The student will achieve competitive top-1 accuracy relative to the teacher (within ~5 percentage points).
- On images both models classify correctly, a meaningful proportion (~30–50%) will show **substantial divergence** in Grad-CAM activation maps.
- On images the student misclassifies, Grad-CAM maps will show higher divergence from the teacher than on correctly classified images — suggesting that reasoning misalignment is predictive of failure.

### Null hypothesis
There is no systematic difference in Grad-CAM activation patterns between teacher and student beyond random variation. Any observed divergence is not correlated with classification outcome.

*We aim to reject the null. If we cannot, that is also a valid and interesting finding — it would suggest KD transfers more than just labels.*

---

## 4. Variables

### 4.1 Independent variables
These are what we deliberately change or compare across conditions.

| Variable | Values |
|---|---|
| Model type | Teacher (ResNet-50) vs. Student (ResNet-18) |
| Training method | KD-trained student vs. baseline student (trained without teacher) |

The baseline student is important: it lets us ask whether any differences we see are due to KD specifically, or just due to model size difference.

### 4.2 Dependent variables
These are what we measure as outcomes.

| Variable | How we measure it |
|---|---|
| Classification accuracy | Top-1 accuracy on test set |
| Attention map similarity | Jensen-Shannon divergence and Spearman rank correlation between normalized Grad-CAM maps |
| Attention divergence vs. failure correlation | Compare mean SSIM on correct vs. incorrect student predictions |

### 4.3 Controlled variables
These are held constant so they do not confound our results.

- Same dataset for all models (CIFAR-10 or Tiny ImageNet)
- Same test images used for all comparisons
- Same Grad-CAM implementation and layer selection applied to all models
- Same image preprocessing pipeline
- Fixed random seed for reproducibility

---

## 5. Methodology

### Step 1 — Model selection
- **Teacher:** ResNet-50, pre-trained on ImageNet (`torchvision.models.resnet50(pretrained=True)`)
- **Student:** ResNet-18 — same architectural family as the teacher, meaningfully smaller
- **Baseline student:** ResNet-18, identical to the student, trained on hard labels only (no teacher). Isolates KD-specific effects from effects of reduced model capacity.

Rationale for architecture choice: Using two models from the same CNN family eliminates architectural confounding. Any observed divergence in Grad-CAM maps is attributable to the distillation process itself, not to fundamental differences in how each model represents space (e.g., attention patches vs. convolutions).

### Step 2 — Dataset
- ImageNette: 10-class ImageNet subset at native 224×224 resolution (~1.4GB, available via Hugging Face `frgfm/imagenette`)
- Use the standard train/val/test split
- All models trained and evaluated on identical splits

Rationale for dataset choice: ResNet's final convolutional layer on a 224×224 input produces a 7×7 feature map — the standard resolution for interpretable Grad-CAM heatmaps. CIFAR-10's native 32×32 content produces spatially uninformative maps regardless of resize.

### Step 3 — Knowledge distillation training
- Follow Hinton et al. (2015) KD procedure: student trained on a weighted combination of hard label loss and soft label loss (teacher's softened probability outputs)
- Temperature T = 4 (standard starting point)
- Log: accuracy per epoch, KD loss, hard label loss

### Step 4 — Grad-CAM generation
- Run Grad-CAM on the full 10,000-image test set (stratified across all 10 classes)- Run Grad-CAM on teacher and student for each image
- Store: predicted label, true label, confidence score, Grad-CAM map (as heatmap overlay)

### Step 5 — Quantitative analysis
- Normalize each Grad-CAM map by its sum so it functions as a spatial probability distribution
- Compute Jensen-Shannon divergence between teacher and student maps for each image (symmetric, bounded 0–1)
- Compute Spearman rank correlation between flattened teacher and student maps for each image- Separate pairs into: both correct / student wrong+teacher correct / both wrong
- Compare mean SSIM across groups — does student failure correlate with lower map similarity?

### Step 6 — Qualitative case study analysis
- Manually select 10 high-agreement pairs, 10 high-divergence pairs, 5 failure cases
- For each: describe what the teacher attends to, what the student attends to, and what explains the difference
- This is the core of the experiments & discussion section

---

## 6. Expected outcomes

### What we expect to see
- Student achieves within 3–7% of teacher accuracy — close enough to call KD "successful" by standard metrics
- A substantial fraction of images show meaningful attention divergence even when both models are correct
- Divergence is noticeably higher on student failures than on student successes
- Qualitatively: teacher tends to focus on the object; student sometimes focuses on background cues or texture

### What would surprise us
- Perfect attention map alignment between teacher and student — would suggest KD transfers reasoning strategy completely
- No correlation between divergence and failure — would suggest attention maps are not informative of confidence or correctness
- Student outperforming teacher — unlikely but possible on small datasets

### What would be a non-result
- Random divergence with no pattern — would mean Grad-CAM is not sensitive enough to detect what we're looking for, not necessarily that the difference doesn't exist

---

## 7. Evaluation criteria

| Criterion | How we assess it |
|---|---|
| Accuracy gap (teacher vs. student) | Numeric: top-1 accuracy difference |
| Attention similarity (correct cases) | Mean SSIM score on jointly correct predictions |
| Attention divergence (failure cases) | Mean SSIM score on student-error cases |
| Effect size | Is the SSIM difference between correct/failure cases large enough to be meaningful? |
| Qualitative coherence | Do our case studies tell a consistent story? |

---

## 8. Anticipated limitations

- **Grad-CAM is imperfect.** It approximates where a model "looks" but does not capture the full internal reasoning. High SSIM does not guarantee the same reasoning; low SSIM does not guarantee different reasoning.
- **ImageNette is not full ImageNet.** Results on a 10-class subset may not generalize to the full 1000-class setting. The 10 ImageNette classes are also visually distinct, which may make the teacher's task easier than a more fine-grained benchmark would.
- **Generalizability.** Results on one teacher–student pair and one dataset may not generalize to other architectures or domains.
- **Correlation vs. causation.** We can show that attention divergence correlates with failure; we cannot prove it causes failure.

---

## 9. What this is not claiming

We are not claiming KD is flawed or that accuracy is a bad metric. We are asking a narrower, prior question: *what does KD actually transfer?* Understanding that is a prerequisite for knowing when to trust a distilled model.

---

*This document is a living draft. Section 5 (methodology) and Section 6 (expected outcomes) will be revised after implementation to reflect what was actually done and found.*
