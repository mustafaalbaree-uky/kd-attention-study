kd-gradcam/
├── teacher/
│   ├── train_teacher.py
│   ├── teacher_training_log.csv
│   └── checkpoints/
│       └── teacher_finetuned.pth
│
├── students/
│   │
│   ├── resnet18/
│   │   ├── train_kd.py
│   │   ├── train_baseline.py
│   │   ├── train_baseline_seed43.py
│   │   ├── generate_gradcam.py
│   │   ├── generate_gradcam_floor.py
│   │   ├── checkpoints/
│   │   │   ├── resnet18_kd.pth
│   │   │   ├── resnet18_baseline.pth
│   │   │   └── resnet18_baseline_seed43.pth
│   │   └── results/
│   │       ├── accuracy.csv
│   │       ├── resnet18_kd_training_log.csv
│   │       ├── resnet18_baseline_training_log.csv
│   │       ├── resnet18_baseline_seed43_training_log.csv
│   │       ├── divergence_scores.csv
│   │       ├── floor_scores.csv
│   │       ├── summary_stats.json
│   │       ├── figures/
│   │       │   ├── figure1_js_divergence_bar.png
│   │       │   ├── figure2_js_by_outcome.png
│   │       │   ├── figure3_spearman_distribution.png
│   │       │   └── figure4_ssim_by_outcome.png
│   │       └── gradcam_full/
│   │           ├── arrays/     ← 3,925 .npz files (one per test image)
│   │           └── figures/    ← 10 sample Grad-CAM comparison PNGs
│   │
│   ├── mobilenet/
│   │   ├── train_kd.py
│   │   ├── train_baseline.py
│   │   ├── train_baseline_seed43.py
│   │   ├── generate_gradcam.py
│   │   ├── generate_gradcam_floor.py
│   │   ├── evaluate.py
│   │   ├── zip_results.py
│   │   ├── checkpoints/
│   │   │   ├── mobilenet_kd.pth
│   │   │   ├── mobilenet_baseline.pth
│   │   │   └── mobilenet_baseline_seed43.pth
│   │   └── results/
│   │       ├── accuracy.csv
│   │       ├── mobilenet_kd_training_log.csv
│   │       ├── mobilenet_baseline_training_log.csv
│   │       ├── mobilenet_baseline_seed43_training_log.csv
│   │       ├── divergence_scores.csv
│   │       ├── floor_scores.csv
│   │       ├── summary_stats.json
│   │       ├── figures/
│   │       │   ├── figure1_js_divergence_bar.png
│   │       │   ├── figure2_js_by_outcome.png
│   │       │   ├── figure3_spearman_distribution.png
│   │       │   └── figure4_ssim_by_outcome.png
│   │       └── gradcam_full/
│   │           ├── arrays/     ← 3,925 .npz files (one per test image)
│   │           └── figures/    ← 10 sample Grad-CAM comparison PNGs
│   │
│   └── densenet/
│       ├── train_kd.py
│       ├── train_baseline.py
│       ├── train_baseline_seed43.py
│       ├── generate_gradcam.py
│       ├── generate_gradcam_floor.py
│       ├── evaluate.py
│       ├── zip_results.py
│       ├── checkpoints/
│       │   ├── densenet_kd.pth
│       │   ├── densenet_baseline.pth
│       │   └── densenet_baseline_seed43.pth
│       └── results/
│           ├── accuracy.csv
│           ├── densenet_kd_training_log.csv
│           ├── densenet_baseline_training_log.csv
│           ├── densenet_baseline_seed43_training_log.csv
│           ├── divergence_scores.csv
│           ├── floor_scores.csv
│           ├── summary_stats.json
│           ├── figures/
│           │   ├── figure1_js_divergence_bar.png
│           │   ├── figure2_js_by_outcome.png
│           │   ├── figure3_spearman_distribution.png
│           │   └── figure4_ssim_by_outcome.png
│           └── gradcam_full/
│               ├── arrays/     ← 3,925 .npz files (one per test image)
│               └── figures/    ← 10 sample Grad-CAM comparison PNGs
│
├── shared/
│   ├── evaluate.py
│   ├── model_check.py
│   ├── score_divergence.py
│   ├── score_floor.py
│   ├── summarize.py
│   └── add_floor_to_summary.py
│
├── humaninfo/
│   ├── codebase_structure.md
│   ├── claude_code_pm_guide_resnet18.md
│   ├── claude_code_pm_guide_mobilenet.md
│   ├── claude_code_pm_guide_densenet.md
│   ├── lesson_plan_guide.md
│   ├── research_design.md
│   ├── project_rubric.md
│   └── lessons/
│       ├── Lesson_1.pdf
│       ├── Lesson_2_How_Weights_Get_Adjusted.pdf
│       ├── Lesson_3_What_Output_Numbers_Mean.pdf
│       ├── Lesson_4_Full_Training_Loop.pdf
│       ├── Lesson_5_Knowledge_Distillation.pdf
│       ├── Lesson_6_Feature_Maps.pdf
│       ├── Lesson_7_GradCAM.pdf
│       └── Lesson_8_The_Project.pdf
│
├── data/
├── audit.py
├── run_floor_overnight.py
├── config.yaml
├── requirements.txt
└── setup.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT ROLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per-student scripts (live inside each students/<arch>/ directory):

  train_kd.py                 Train the KD student (soft-label distillation from teacher)
  train_baseline.py           Train the vanilla baseline student (cross-entropy only)
  train_baseline_seed43.py    Same as train_baseline.py but with seed=43 for floor computation
  generate_gradcam.py         Generate a small set of sample Grad-CAM visualizations
  generate_gradcam_floor.py   Generate full 3,925-image Grad-CAM arrays for all three models
                              (teacher, kd_student, baseline) and save as .npz
  evaluate.py                 Evaluate checkpoints and write accuracy.csv
                              (mobilenet and densenet only; resnet18 uses shared/evaluate.py)
  zip_results.py              Package results/ for download from Kaggle
                              (mobilenet and densenet only)

Shared scripts (live in shared/, used by all architectures):

  evaluate.py                 Evaluate a checkpoint against the test set; write accuracy.csv
  model_check.py              Sanity-check a checkpoint loads and runs a forward pass
  score_divergence.py         Compute JS divergence, Spearman r, and SSIM between Grad-CAM
                              maps; write divergence_scores.csv
  score_floor.py              Compute floor metrics (seed-43 baseline vs seed-0 baseline);
                              write floor_scores.csv
  summarize.py                Aggregate divergence_scores.csv into summary_stats.json
  add_floor_to_summary.py     Merge floor_scores.csv statistics into an existing
                              summary_stats.json


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULT FILE ROLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  accuracy.csv                     Top-1 accuracy for teacher, KD student, and baseline
  *_training_log.csv               Per-epoch loss/accuracy during training (30 epochs each)
  *_baseline_seed43_training_log   Same format, for the seed-43 floor run
  divergence_scores.csv            Per-image JS, Spearman r, SSIM between teacher↔kd and
                                   teacher↔baseline Grad-CAM maps (3,925 rows)
  floor_scores.csv                 Per-image JS, Spearman r, SSIM between seed-0 baseline
                                   and seed-43 baseline maps — establishes noise floor
                                   (3,925 rows)
  summary_stats.json               Aggregated statistics: means, Mann-Whitney U, floor means
  figures/figure[1-4]_*.png        Publication-quality per-architecture analysis figures
  gradcam_full/arrays/*.npz        One file per test image; keys: teacher, kd_student,
                                   baseline, true_label, *_pred (normalized 7×7 maps)
  gradcam_full/figures/*.png       10 randomly sampled side-by-side Grad-CAM comparisons


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Floor computation: The seed-43 baseline is trained identically to the seed-0 baseline
  except for the random seed. Its Grad-CAM maps are compared against the seed-0 baseline
  maps to produce floor_scores.csv. This gives a within-model noise floor for each metric,
  letting us interpret whether teacher↔kd gaps are meaningfully larger than training-seed
  variance.

.npz array format: Each file stores normalized Grad-CAM activation maps as float32 arrays
  of shape (7, 7) (final spatial resolution before global average pooling). Maps sum to 1.0.
  Labels and predictions are stored as scalars under true_label, teacher_pred, kd_pred,
  baseline_pred.

Kaggle vs local: Training and Grad-CAM generation for mobilenet and densenet were run on
  Kaggle (GPU). Checkpoints, result CSVs, and figures were downloaded locally. The accuracy
  CSVs for those architectures therefore contain absolute Kaggle paths in the checkpoint
  column — this is cosmetic and does not affect analysis.
