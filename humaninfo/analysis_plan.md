# Analysis plan — KD saliency alignment study

**Project:** Does knowledge distillation produce greater saliency alignment than hard-label training?  
**Architectures:** ResNet-18, MobileNetV2, DenseNet-121 (all students) ← ResNet-50 (teacher)  
**Dataset:** ImageNette (10-class ImageNet subset, 224×224)  
**Status:** Implementation complete. This file governs the analysis phase.

---

## How this phase works

Each step has a clear question it answers, the data it draws on, the decision criteria that determine what the finding is, and a findings box to fill in after we work through the results together. Steps are completed in order. No writing begins until all eight steps are complete and their findings boxes are filled.

The findings recorded here become the raw material for the paper's Experiments & Discussion section. The qualitative case study (Step 7) becomes the analytical spine of that section.

---

## State tracker

- [ ] Step 1 — Accuracy validity check
- [ ] Step 2 — Headline alignment result (RQ2a)
- [ ] Step 3 — Floor-relative interpretation
- [ ] Step 4 — Failure correlation (RQ2b)
- [ ] Step 5 — Metric agreement check
- [ ] Step 6 — Cross-architecture consistency
- [ ] Step 7 — Qualitative case study
- [ ] Step 8 — Synthesis

---

## Step 1 — Accuracy validity check

**The question this step answers:**
Are the two students close enough in accuracy to each other — and to the teacher — that any alignment difference we find can be attributed to the training procedure (KD vs. hard labels) rather than to one model simply being better?

This step is the precondition for everything that follows. If the KD student is substantially more accurate than the baseline, it would be confounded: a better model naturally looks more like a better teacher. We need to rule this out before we can attribute alignment differences to KD itself.

**Data needed:**
- `students/resnet18/results/accuracy.csv`
- `students/mobilenet/results/accuracy.csv`
- `students/densenet/results/accuracy.csv`

**What to look for:**

1. **KD vs. baseline accuracy gap** — is it small? The target threshold is ≤2 percentage points difference between the KD student and baseline student for each architecture. If the gap is larger, the alignment comparison is compromised and must be flagged as a limitation.

2. **Both students vs. teacher accuracy gap** — are both students within ~5pp of the teacher? This is the standard benchmark for "successful distillation." If either student falls far below the teacher, note it.

3. **Consistency across architectures** — does the accuracy pattern hold for all three? Note any architecture where the pattern breaks.

**Decision criteria:**

| Outcome | Interpretation |
|---|---|
| KD and baseline within ≤2pp of each other (all architectures) | Validity condition met — alignment differences can be attributed to training procedure |
| KD and baseline within 2–5pp | Flag as partial confound — discuss in limitations, but proceed |
| KD vs. baseline gap >5pp for any architecture | Major confound — any alignment result for that architecture must be interpreted with heavy caution |
| Both students within ~5pp of teacher | Distillation is "successful" by standard accuracy metric |

**Findings (fill in after analysis):**

| Architecture | Teacher acc | KD student acc | Baseline acc | KD–Baseline gap | Validity? |
|---|---|---|---|---|---|
| ResNet-18 | | | | | |
| MobileNetV2 | | | | | |
| DenseNet-121 | | | | | |

**Narrative finding:**

> [Fill in after analysis — e.g. "Across all three architectures, the KD student and baseline student achieve top-1 accuracy within X–Ypp of each other, with the teacher at 99.4%. The accuracy differences between the two students are small enough that they do not constitute a confound for the alignment analysis."]

---

## Step 2 — Headline alignment result (RQ2a)

**The question this step answers:**
Does training with soft teacher labels produce greater Grad-CAM saliency alignment with the teacher than training on hard labels alone?

This is the primary empirical claim of the paper. We test it across three architectures and three metrics. The more consistently the result replicates, the stronger the claim.

**Data needed:**
- `summary_stats.json` for all three architectures (mean JS, Spearman, SSIM for KD vs. teacher and baseline vs. teacher; Mann-Whitney U statistic and p-value)

**What to look for:**

1. **Direction of effect** — for each architecture and each metric, is KD student more aligned with teacher than baseline? 
   - Lower JS divergence = more aligned
   - Higher Spearman r = more aligned
   - Higher SSIM = more aligned

2. **Statistical significance** — is the Mann-Whitney p-value < 0.05 for each architecture? (The test is one-tailed: KD JS < baseline JS)

3. **Effect magnitude** — how large is the difference in absolute terms? (e.g., mean JS of 0.122 vs. 0.136 is a difference of 0.014)

4. **Consistency** — does the direction of effect hold across all three architectures and all three metrics, or are there exceptions?

**Decision criteria:**

| Outcome | Interpretation |
|---|---|
| KD more aligned than baseline, p < 0.05, all three architectures, all three metrics | Strong evidence — null hypothesis rejected cleanly |
| KD more aligned than baseline, p < 0.05, all three architectures, 2/3 metrics | Moderate-strong evidence — note which metric disagrees and why |
| KD more aligned than baseline, p < 0.05, 2/3 architectures | Moderate evidence — discuss what's different about the third architecture |
| Effect present but p > 0.05 for any architecture | Weak evidence for that architecture — do not overstate |
| Effect absent or reversed for any architecture | Null holds there — this is a real finding, not a failure |

**Findings (fill in after analysis):**

**JS divergence (lower = more aligned):**

| Architecture | KD mean JS (std) | Baseline mean JS (std) | Difference | Mann-Whitney p |
|---|---|---|---|---|
| ResNet-18 | | | | |
| MobileNetV2 | | | | |
| DenseNet-121 | | | | |

**Spearman r (higher = more aligned):**

| Architecture | KD mean r (std) | Baseline mean r (std) | Difference | |
|---|---|---|---|---|
| ResNet-18 | | | | |
| MobileNetV2 | | | | |
| DenseNet-121 | | | | |

**SSIM (higher = more aligned):**

| Architecture | KD mean SSIM (std) | Baseline mean SSIM (std) | Difference | |
|---|---|---|---|---|
| ResNet-18 | | | | |
| MobileNetV2 | | | | |
| DenseNet-121 | | | | |

**Narrative finding:**

> [Fill in after analysis]

---

## Step 3 — Floor-relative interpretation

**The question this step answers:**
Is the alignment advantage of the KD student practically meaningful — or is it smaller than the divergence you would expect between any two models trained identically with different random seeds?

Statistical significance (Step 2) tells us the effect is real. This step tells us whether it is large enough to matter. The floor score is the key reference: it measures JS divergence between two baseline models trained identically except for their random seed. This represents the minimum expected divergence between any two reasonable models of the same architecture — pure initialization variance, no meaningful difference in what they learned.

If the KD advantage is smaller than the floor divergence, the effect is real but small. If it is larger, the effect is practically meaningful — KD is moving the student's attention maps by more than chance would.

**Data needed:**
- `summary_stats.json` for all three architectures — specifically the floor keys: `floor_js_mean`, `floor_spearman_mean`, `floor_ssim_mean`
- Mean JS/Spearman/SSIM for KD and baseline (from Step 2)

**What to compute:**

For each architecture, compute the following quantities:

- **KD advantage (JS):** `baseline_js_mean − kd_js_mean` (how much more aligned is KD vs. baseline)
- **KD advantage as fraction of floor:** `(baseline_js_mean − kd_js_mean) / floor_js_mean` (is the advantage large relative to initialization noise?)
- **KD distance to floor (JS):** `kd_js_mean − floor_js_mean` (how much further from the floor is the KD student?)
- **Baseline distance to floor (JS):** `baseline_js_mean − floor_js_mean`

Repeat for Spearman and SSIM.

**What to look for:**

The most important question: is the KD student closer to the floor than the baseline? The floor represents the minimum expected divergence when two models have no reason to differ. If the KD student is substantially closer to the floor than the baseline, it means KD is pulling the student's attention patterns toward the teacher — not just marginally.

Conversely: if KD advantage ÷ floor JS < 0.10, then the KD advantage is less than 10% of initialization variance. That is a real but very small effect.

**Decision criteria:**

| KD advantage ÷ floor JS | Interpretation |
|---|---|
| > 0.50 | Large effect — KD is moving the student's maps by more than half of initialization variance |
| 0.20 – 0.50 | Moderate effect — meaningful but modest |
| 0.10 – 0.20 | Small effect — real but limited practical significance |
| < 0.10 | Negligible practical effect — state this honestly |

**Findings (fill in after analysis):**

| Architecture | Floor JS | KD adv (JS) | KD adv ÷ floor | Interpretation |
|---|---|---|---|---|
| ResNet-18 | | | | |
| MobileNetV2 | | | | |
| DenseNet-121 | | | | |

**Narrative finding:**

> [Fill in after analysis — this is where honesty about effect size lives. If the effect is small relative to floor, say so. That is still a finding.]

---

## Step 4 — Failure correlation (RQ2b)

**The question this step answers:**
Within the KD student, is saliency divergence from the teacher correlated with student classification errors? When the student's attention maps diverge most from the teacher's, is that also when it is most likely to get the answer wrong?

This is the second research sub-question. It is distinct from RQ2a: RQ2a asks whether KD produces more alignment on average; RQ2b asks whether the alignment signal is informative about individual prediction quality.

**Data needed:**
- `summary_stats.json` for all three architectures — specifically the outcome-group breakdown:
  - Mean JS/Spearman/SSIM when both models are correct
  - Mean JS/Spearman/SSIM when student is wrong and teacher is correct
  - Mean JS/Spearman/SSIM when both models are wrong
- Note: this breakdown should exist for both KD student and baseline

**What to look for:**

The expected ordering for the KD student (if hypothesis holds):
- **Both correct** → lowest divergence (student attending to same regions as teacher, getting right answer)
- **Student wrong, teacher right** → higher divergence (student attending elsewhere, failing)
- **Both wrong** → highest or intermediate divergence (both models missing something — potentially attending to same wrong region)

Check this ordering for all three metrics and all three architectures. Also check whether the baseline shows the same pattern — if it does, the failure correlation is a property of model size difference, not of KD specifically.

**Counts matter:** How many images fall into each outcome group? If there are very few images where the student is wrong and the teacher is right, the mean for that group is unreliable. Note the sample sizes.

**Decision criteria:**

| Outcome | Interpretation |
|---|---|
| Monotonic ordering holds for KD student, all architectures, all metrics | Strong evidence — divergence predicts failure |
| Ordering holds for KD student but not baseline | KD creates a tighter coupling between alignment and correctness |
| Ordering holds for both KD and baseline | Failure correlation is a property of model capacity, not KD |
| No clear ordering | Divergence does not predict failure for these architectures/dataset |

**Findings (fill in after analysis):**

**KD student — JS divergence by outcome group:**

| Architecture | Both correct | Student wrong, teacher right | Both wrong | Ordering holds? |
|---|---|---|---|---|
| ResNet-18 | | | | |
| MobileNetV2 | | | | |
| DenseNet-121 | | | | |

**Baseline — JS divergence by outcome group:**

| Architecture | Both correct | Student wrong, teacher right | Both wrong | Ordering holds? |
|---|---|---|---|---|
| ResNet-18 | | | | |
| MobileNetV2 | | | | |
| DenseNet-121 | | | | |

**Approximate image counts per group (from any architecture):**

> [Fill in — how many images fall in each group? This determines how much weight to give the group means.]

**Narrative finding:**

> [Fill in after analysis]

---

## Step 5 — Metric agreement check

**The question this step answers:**
Do Jensen-Shannon divergence, Spearman rank correlation, and SSIM all tell the same story — or does one of them disagree, and if so, why?

Each metric captures a different aspect of map similarity. JS divergence treats the normalized map as a probability distribution and measures distributional agreement, ignoring spatial layout. Spearman rank correlation measures monotonic agreement between map values when laid out flat — it captures whether the same pixels are ranked as high-activation in both maps, but does not weight by spatial proximity. SSIM compares local patches, preserving spatial structure and measuring how similar the two maps are neighborhood-by-neighborhood.

If all three agree, our conclusion is robust to measurement choice. If one disagrees, it tells us something specific about the nature of the alignment — or misalignment.

**Data needed:**
- Summary stats for all three architectures — all three metrics, both students
- This step uses numbers already collected in Steps 2 and 4; no new data needed

**What to look for:**

Go through each finding from Steps 2 and 4 and check whether all three metrics point in the same direction. Specifically:

1. Does the direction of the headline effect (KD more aligned than baseline) hold across all three metrics for each architecture?
2. Does the failure correlation ordering hold across all three metrics, or does one metric show a different pattern?
3. Are there any cases where JS and SSIM disagree? (This would suggest the maps agree on which pixels are active but disagree on spatial arrangement — or vice versa.)

**Plausible disagreements and their meanings:**

- **JS agrees but SSIM disagrees:** The maps have similar overall distributions (the same pixels tend to be active) but the spatial arrangement differs. Could happen if the student's activations are shifted or blurred relative to the teacher's.
- **Spearman high but JS high (divergent):** The ranking of pixels agrees but the magnitudes differ — one map is more peaked than the other. Temperature effects in the Grad-CAM normalization could cause this.
- **All three agree:** The finding is robust and does not depend on how you measure alignment.

**Findings (fill in after analysis):**

| Comparison | JS direction | Spearman direction | SSIM direction | All agree? |
|---|---|---|---|---|
| KD vs baseline alignment (ResNet-18) | | | | |
| KD vs baseline alignment (MobileNetV2) | | | | |
| KD vs baseline alignment (DenseNet-121) | | | | |
| Failure correlation — KD student (ResNet-18) | | | | |
| Failure correlation — KD student (MobileNetV2) | | | | |
| Failure correlation — KD student (DenseNet-121) | | | | |

**Narrative finding:**

> [Fill in after analysis — if all metrics agree, say so clearly and note what it means for the robustness of the conclusion. If one disagrees, explain why and what it tells us.]

---

## Step 6 — Cross-architecture consistency

**The question this step answers:**
Does the alignment advantage from KD vary by student architecture — and if so, does architectural similarity to the teacher explain the variation?

The three student architectures are structurally very different. ResNet-18 is in the same family as the teacher (ResNet-50) — both use skip connections and the same style of convolutional blocks. MobileNetV2 uses inverted residuals and depthwise separable convolutions, making it structurally unlike either ResNet. DenseNet-121 uses dense connections where every layer receives all prior feature maps — a completely different information-flow pattern.

If architectural similarity predicts alignment advantage, that would be a meaningful finding: KD works better at transferring attention when the student and teacher share structural vocabulary. If all three architectures show similar effects, the alignment benefit is robust and architecture-agnostic.

**Data needed:**
- Headline alignment numbers from Step 2 and floor-relative numbers from Step 3, organized by architecture
- This step synthesizes existing findings; no new data needed

**What to compare:**

1. **KD alignment advantage (JS)** across architectures — which architecture shows the largest advantage? Which shows the smallest?
2. **KD advantage ÷ floor JS** across architectures — is the practical effect size consistent, or does it vary?
3. **Floor JS value itself** across architectures — do different architectures have different baseline variance? (A higher floor means more inherent variability between runs, which could mask or amplify the KD effect.)
4. **Statistical significance** — is the effect significant for all three architectures, or does p-value strength vary?

**Key architectural hypothesis to test:**
ResNet-18 (same family as teacher) should show the greatest alignment advantage. MobileNetV2 (most structurally different) should show the smallest. DenseNet-121 should fall in between. If the pattern is reversed, that is an interesting non-obvious finding.

**Findings (fill in after analysis):**

| Architecture | KD adv (JS) | KD adv ÷ floor | Rank (1=most aligned) | Mann-Whitney p |
|---|---|---|---|---|
| ResNet-18 | | | | |
| MobileNetV2 | | | | |
| DenseNet-121 | | | | |

**Does architectural similarity to teacher predict alignment advantage?**

> [Fill in — yes/no/partially, with explanation]

**Narrative finding:**

> [Fill in after analysis]

---

## Step 7 — Qualitative case study

**The question this step answers:**
What does the alignment (or misalignment) actually look like? Are there consistent patterns in what the teacher attends to vs. what the student attends to? Do the failure cases have a recognizable visual signature?

This step is the analytical spine of the paper's experiments section. The quantitative results tell us that the effect exists and is statistically significant; this step tells us what the effect means. Without this, the paper reads as a benchmarking exercise. With it, it becomes an argument about what KD actually does.

**Data needed:**
- Sample Grad-CAM PNGs from `results/gradcam_full/figures/` for all three architectures (you will send these)
- `divergence_scores.csv` for one or two architectures — to identify specific images for selection if needed

**Selection criteria for case studies:**

Select images in the following categories. Aim for 3–5 images total per category, distributed across architectures.

**Category A — High agreement (KD student ≈ teacher):**
Images where both teacher and KD student clearly attend to the same region of the image. What does the teacher attend to here? Does the baseline also agree, or does the KD student align where the baseline doesn't?

**Category B — KD diverges from teacher, baseline even more so:**
Images where the KD student's attention diverges from the teacher, but the baseline diverges more. These illustrate the headline effect concretely — KD pulled the student toward the teacher, even if not all the way.

**Category C — High divergence + student failure:**
Images where the student's attention diverges from the teacher AND the student makes a wrong prediction. The goal is to see if there is a visual pattern — is the student attending to background? To texture rather than object? To a confounding object in the scene?

**Category D — Counter-examples (if any exist):**
Images where the baseline student is more aligned with the teacher than the KD student. These are the null hypothesis cases and should be acknowledged if they appear in meaningful numbers.

**What to record for each image:**

For each selected image:
- Architecture
- True label and predicted labels (teacher, KD student, baseline)
- What region does the teacher attend to?
- What region does the KD student attend to?
- What region does the baseline attend to?
- What explains the difference? (if anything can be said)
- JS divergence values for that image (from divergence_scores.csv)

**Findings (fill in during analysis session):**

> [This section will be populated image-by-image when we go through the Grad-CAM PNGs together. The goal is 2–3 sentences per image that could be dropped directly into the paper's case study discussion.]

**Patterns to watch for across all case studies:**
- Does the teacher tend to attend to the object itself, while students attend to background or texture?
- Do the failure cases show the student attending to a plausible but wrong part of the image?
- Is there any class where alignment is consistently poor? (Some ImageNette classes — e.g., gas pumps, golf balls — have more complex background distributions)
- Do the three student architectures fail in different ways visually?

---

## Step 8 — Synthesis

**The question this step answers:**
What can we actually conclude, how confidently, and why does it matter?

This step produces the paper's conclusion. It draws on all prior findings and answers four questions precisely.

**The four questions:**

**1. Did we reject the null hypothesis?**
The null: there is no systematic difference in Grad-CAM alignment between the KD student and baseline relative to the teacher. Based on Steps 2 and 5, state clearly whether we reject it, for which architectures, and with what confidence.

**2. What is the magnitude of the effect?**
Based on Step 3: is the alignment advantage practically meaningful relative to the floor, or is it statistically real but practically small? Be precise. Do not overstate.

**3. Does divergence predict failure?**
Based on Step 4: is the failure correlation present, consistent, and specific to KD — or is it a general property of model capacity difference?

**4. What can't we conclude?**
Grad-CAM is a first-order approximation. Similar heatmaps do not guarantee similar internal representations. We can show that KD produces maps that look more like the teacher's — we cannot show that the student reasons the way the teacher does. This distinction must be stated clearly.

**Findings (fill in after analysis):**

**Overall verdict on RQ2a:**
> [Reject / fail to reject null, with which architectures, which metrics, what p-values]

**Overall verdict on RQ2b:**
> [Does divergence correlate with failure, and what is the pattern]

**Effect size judgment:**
> [Is the alignment advantage large, moderate, small, or negligible relative to floor?]

**The one non-obvious insight:**
> [What is the single most interesting finding that someone reading just the abstract would not expect? This is the paper's contribution beyond confirming the hypothesis.]

**The honest limitation:**
> [What is the single most important thing we cannot conclude from this data?]

**Paper-ready one-paragraph conclusion:**
> [Fill in last — once all other findings are recorded, write the conclusion paragraph here. This goes directly into the paper.]

---

## Cross-reference: research questions and where they are answered

| Research question | Primary step | Supporting steps |
|---|---|---|
| RQ2a: Does KD produce greater Grad-CAM alignment with teacher than hard-label training? | Step 2 | Steps 3, 5, 6 |
| RQ2b: Does saliency divergence correlate with student failure? | Step 4 | Step 7 |
| Is the effect practically meaningful (not just statistically significant)? | Step 3 | Step 6 |
| Is the conclusion robust to metric choice? | Step 5 | — |
| Is the conclusion robust across student architectures? | Step 6 | Steps 2, 3 |
| What does the effect look like visually? | Step 7 | — |

---

## What comes after this plan is complete

Once all eight steps are complete and all findings boxes are filled, the analysis phase is done. What follows:

1. **Research design document update** — revise `research_design.md` to reflect what was actually done and found (it is a living draft)
2. **Paper outline** — structure the experiments & discussion section around the eight-step arc
3. **Paper writing** — prose from findings, not from memory
4. **Slide preparation** — the narrative arc of Steps 1 → 8 maps directly onto a presentation structure

Do not begin paper writing until this document is fully populated.
