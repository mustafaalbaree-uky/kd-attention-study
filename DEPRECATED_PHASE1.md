# Deprecated — Phase 1

## What was built
- Environment setup: requirements.txt, config.yaml, setup.py, model_check.py
- KD training loop: train_kd.py
- Trained KD student checkpoint (deleted)
- Baseline training run (terminated early, no checkpoint kept)

## Why this was deprecated
Architecture pivot approved before Grad-CAM analysis began.

**Old setup:** ViT teacher (google/vit-base-patch16-224) → MobileViT-small student, CIFAR-10 dataset, 200-image Grad-CAM sample, SSIM similarity metric.

**Problem 1 — Architectural confound:** ViT and MobileViT use fundamentally different spatial reasoning mechanisms. Any Grad-CAM divergence could reflect architecture, not KD. The research question requires isolating KD effects.

**Problem 2 — Grad-CAM resolution:** CIFAR-10 images have native resolution of 32×32. Even though images were resized to 224×224 for model input, the underlying content is low-information upscaled data. Grad-CAM heatmaps on such images are spatially uninformative — there is insufficient scene structure to produce meaningful attention maps for comparison.

## New setup (Phase 2)
- Teacher: ResNet-50 (pre-trained on ImageNet)
- Student: ResNet-18 (KD-trained against frozen teacher)
- Baseline: ResNet-18 (hard labels only, no teacher)
- Dataset: ImageNette (10-class ImageNet subset, 224×224)
- Metrics: KL divergence (Jensen-Shannon variant) + Spearman rank correlation on Grad-CAM maps
- Compute: Kaggle GPU (GitHub → Kaggle → local results workflow)
