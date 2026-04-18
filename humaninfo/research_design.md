# Research design — first draft
**Project:** Does knowledge distillation produce greater saliency alignment than hard-label training?
**Architecture:** ResNet-50 (teacher) → ResNet-18 (student) — pure CNN pipeline
**Dataset:** ImageNette (10-class ImageNet subset, 224×224)
**Course:** Deep learning (599 section) · 2-person team
**Status:** Pre-implementation draft — to be revised after experiments complete

---

## 1. The problem

Knowledge distillation (KD) is a widely used technique for compressing large neural networks into smaller, faster ones. The field measures its success almost exclusively through **accuracy transfer** — does the student model match the teacher's classification performance?

But accuracy is a blunt instrument. A model can get the right answer for the wrong reason. Two models can agree on a label while attending to completely different parts of an image to reach that decision.

**The gap:** KD only constrains the student to match the teacher's output distribution — it never touches the teacher's internal representations. So the question is not whether KD transfers reasoning (the mechanism makes that unlikely by design), but whether output-matching pressure *incidentally* produces greater saliency alignment than standard hard-label training. Prior work does not systematically test this.
---

### 2. Research question

> Does training with soft teacher labels produce greater Grad-CAM saliency alignment with the teacher than training with hard labels alone — and when the KD student's saliency diverges from the teacher's, does that divergence predict student failure?

This breaks into two sub-questions:

- **2a.** Does the KD student produce Grad-CAM maps more similar to the teacher than the baseline student does, on images all three models classify correctly?
- **2b.** Within the KD student, is saliency divergence from the teacher correlated with student classification errors?

---

### Hypothesis (our claim)
Training with soft teacher labels produces greater Grad-CAM saliency alignment with the teacher than training on hard labels alone. Specifically:

- Both students will achieve competitive top-1 accuracy relative to the teacher (within ~5 percentage points).
- The KD student's Grad-CAM maps will show higher mean similarity to the teacher than the baseline student's maps, on images all three classify correctly.
- Within the KD student, saliency divergence from the teacher will be higher on images the student misclassifies than on images it gets right.

### Null hypothesis
There is no systematic difference in Grad-CAM saliency alignment between the KD student and the baseline student, relative to the teacher. Any observed difference is not correlated with classification outcome.

*We aim to reject the null. If we cannot, that is also a meaningful finding — it would suggest that output-matching pressure does not incidentally produce saliency alignment, and that the two phenomena are independent.*

---

## 4. Variables

### 4.1 Independent variables
These are what we deliberately change or compare across conditions.

| Variable | Values |
|---|---|
| Model type | Teacher (ResNet-50) vs. Student (ResNet-18) |
| Training method | KD-trained student vs. baseline student (trained without teacher) |

The baseline student is the primary comparison point: it lets us isolate whether greater saliency alignment (if observed) is due to KD specifically, or just a consequence of reduced model capacity. Without it, we cannot attribute anything to the training procedure.

### 4.2 Dependent variables
These are what we measure as outcomes.

| Variable | How we measure it |
|---|---|
| Classification accuracy | Top-1 accuracy on test set |
| Attention map similarity | Jensen-Shannon divergence and Spearman rank correlation between normalized Grad-CAM maps |
| Attention divergence vs. failure correlation | Compare mean SSIM on correct vs. incorrect student predictions |

### 4.3 Controlled variables
These are held constant so they do not confound our results.

| Variable | How controlled |
|---|---|
| Dataset | Same ImageNette split for all models |
| Test images | Same 3,925 validation images for all comparisons |
| Grad-CAM implementation | Same library, same target layer selection applied consistently |
| Image preprocessing | Same pipeline for all models |
| Random seed | Fixed at 42 for all primary models; seed 43 used exclusively for floor baseline |
| Floor baseline seed | Same seed (43) used across all three student architectures |

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
- For each image, compute three metrics between the teacher map and each student map:
  - **Jensen-Shannon divergence** (symmetric, bounded 0–1): treats the normalized map as a probability distribution and measures distributional difference. Does not preserve spatial structure.
  - **Spearman rank correlation** (bounded −1 to 1): measures monotonic spatial agreement between two flattened maps. Non-parametric and robust to outlier activations.
  - **SSIM** (Structural Similarity Index, bounded −1 to 1): preserves local spatial structure by comparing luminance, contrast, and structure in local windows. Primary spatially-aware complement to JS divergence.
- Separate image pairs into outcome groups: both correct / student wrong + teacher correct / both wrong
- Compare mean JS divergence, mean Spearman r, and mean SSIM across groups for each student
- **Floor reference:** For each architecture, a second baseline model trained with seed 43 (all else identical to the seed-42 baseline) is used to establish a reference divergence floor. JS, Spearman, and SSIM are computed between the two baseline models on the same images. This anchors the scale: it represents the minimum expected divergence between two models with no reason to attend differently beyond random initialization. The KD and baseline student divergences from the teacher are interpreted relative to this floor.

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
- The KD student and baseline student showing equivalent saliency alignment to the teacher — would suggest KD's output-matching pressure produces no incidental alignment effect
- No correlation between divergence and failure — would suggest attention maps are not informative of confidence or correctness
- Student outperforming teacher — unlikely but possible on small datasets

### What would be a non-result
- Random divergence with no pattern — would mean Grad-CAM is not sensitive enough to detect what we're looking for, not necessarily that the difference doesn't exist

---

## 7. Evaluation criteria

| Criterion | Metric(s) |
|---|---|
| Accuracy gap (teacher vs. student) | Top-1 accuracy difference |
| Overall saliency alignment — KD vs. baseline | Mean JS divergence, mean Spearman r, mean SSIM (lower JS / higher Spearman+SSIM = more aligned) |
| Saliency alignment on correct predictions | Mean JS, Spearman r, SSIM on jointly correct images |
| Saliency divergence on failure cases | Mean JS, Spearman r, SSIM on student-error images |
| Effect size | Is the difference between correct/failure groups consistent across all three metrics? |
| Statistical significance | Mann-Whitney U test between KD and baseline JS distributions |
| Reference floor (scale anchor) | Mean JS, Spearman r, SSIM between seed-42 and seed-43 baselines — establishes minimum expected divergence |
| Qualitative coherence | Do case studies tell a consistent story with the quantitative findings? |

---

## 8. Anticipated limitations

- **Grad-CAM is imperfect and unstable.** It is a coarse, first-order approximation — its resolution is bounded by the spatial size of the chosen feature map (7×7 for ResNet on 224×224 inputs), and it can be sensitive to gradient noise. More importantly, observed differences in saliency maps between teacher and student may reflect architectural differences or gradient instability rather than any meaningful divergence in how the two models process images. We interpret map similarity as a measurable proxy for saliency alignment, not as evidence of shared or divergent internal reasoning.
- **ViT excluded by design.** A natural extension of this study would be to use a ViT teacher and examine whether the same alignment pattern holds when attention and saliency are more closely coupled. We excluded ViT for a principled reason: Grad-CAM does not transfer cleanly to transformer architectures. Applying it to a ViT would require a different method entirely (e.g., attention rollout or Chefer et al.'s transformer attribution), and the resulting maps would not be directly comparable to CNN Grad-CAM maps. We note this as a meaningful direction for future work.
- **Floor baseline is architecture-specific.** The seed-43 baseline used for floor computation shares the same architecture as the seed-42 baseline. This means the floor value differs across architectures (ResNet-18, MobileNet, DenseNet) and cannot be used to compare absolute alignment levels across experiments — only to contextualize results within each architecture's own scale.
- **ImageNette is not full ImageNet.** Results on a 10-class subset may not generalize to the full 1000-class setting. The 10 ImageNette classes are also visually distinct, which may make the teacher's task easier than a more fine-grained benchmark would.
- **Generalizability.** Results on one teacher–student pair and one dataset may not generalize to other architectures or domains.
- **Correlation vs. causation.** We can show that attention divergence correlates with failure; we cannot prove it causes failure.

---

## 9. What this is not claiming

We are not claiming KD is flawed or that accuracy is a bad metric. We are also not claiming saliency maps reveal internal reasoning — the professor's feedback is explicit that similar heatmaps do not guarantee similar reasoning. We are asking a narrower question: does the pressure to match a teacher's output distribution produce any measurable incidental effect on saliency alignment, compared to a model trained without that pressure? The results, whatever they are, are informative about what KD actually does beyond the accuracy numbers.

---

*This document is a living draft. Section 5 (methodology) and Section 6 (expected outcomes) will be revised after implementation to reflect what was actually done and found.*
