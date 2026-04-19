"""
Compute Jensen-Shannon divergence, Spearman rank correlation, SSIM, and mIoU
between teacher and each student Grad-CAM map for the full ImageNette validation set.
Outputs {student}_divergence_scores.csv with one row per image.

Usage (from project root):
    python shared/score_divergence.py                 # defaults to students/resnet18/
    python shared/score_divergence.py --student mobilenet
    python shared/score_divergence.py --student densenet
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim_fn

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent       # shared/
_ROOT = _HERE.parent                # project root

parser = argparse.ArgumentParser()
parser.add_argument("--student", default="resnet18",
                    help="Student subdirectory name under students/ (default: resnet18)")
args = parser.parse_args()

STUDENT_DIR = _ROOT / "students" / args.student
ARRAYS_DIR  = STUDENT_DIR / "results" / "gradcam_full" / "arrays"
OUT_CSV     = STUDENT_DIR / "results" / f"{args.student}_divergence_scores.csv"

FIELDNAMES = [
    "filename",
    "true_label",
    "teacher_pred", "kd_pred", "baseline_pred",
    "teacher_correct", "kd_correct", "baseline_correct",
    "js_teacher_kd", "js_teacher_baseline",
    "spearman_teacher_kd", "spearman_teacher_baseline",
    "ssim_teacher_kd", "ssim_teacher_baseline",
    "miou_teacher_kd", "miou_teacher_baseline",
]


def js(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(jensenshannon(a, b))


def spear(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Maps are stored flat or 2-D; ensure (7,7) before SSIM
    a2 = a.astype(np.float64).reshape(7, 7)
    b2 = b.astype(np.float64).reshape(7, 7)
    return float(ssim_fn(a2, b2, data_range=1.0, win_size=7))


def compute_miou(map1: np.ndarray, map2: np.ndarray) -> float:
    a = map1.astype(np.float64).ravel()
    b = map2.astype(np.float64).ravel()
    ious = []
    for p in range(0, 101, 5):  # 21 thresholds: 0, 5, 10, ..., 100
        t1 = np.percentile(a, p)
        t2 = np.percentile(b, p)
        mask1 = a >= t1
        mask2 = b >= t2
        intersection = float(np.sum(mask1 & mask2))
        union = float(np.sum(mask1 | mask2))
        ious.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(ious))


def process(npz_path: Path) -> dict:
    d = np.load(npz_path)

    true_label    = int(d["true_label"])
    teacher_pred  = int(d["teacher_pred"])
    kd_pred       = int(d["kd_pred"])
    baseline_pred = int(d["baseline_pred"])

    t  = d["teacher"]
    kd = d["kd_student"]
    bl = d["baseline"]

    return {
        "filename":                  npz_path.name,
        "true_label":                true_label,
        "teacher_pred":              teacher_pred,
        "kd_pred":                   kd_pred,
        "baseline_pred":             baseline_pred,
        "teacher_correct":           int(teacher_pred  == true_label),
        "kd_correct":                int(kd_pred       == true_label),
        "baseline_correct":          int(baseline_pred == true_label),
        "js_teacher_kd":             js(t, kd),
        "js_teacher_baseline":       js(t, bl),
        "spearman_teacher_kd":       spear(t, kd),
        "spearman_teacher_baseline": spear(t, bl),
        "ssim_teacher_kd":           ssim(t, kd),
        "ssim_teacher_baseline":     ssim(t, bl),
        "miou_teacher_kd":           compute_miou(t, kd),
        "miou_teacher_baseline":     compute_miou(t, bl),
    }


def main() -> None:
    if not ARRAYS_DIR.exists():
        raise FileNotFoundError(f"Arrays directory not found: {ARRAYS_DIR}")

    npz_files = sorted(ARRAYS_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {ARRAYS_DIR}")

    print(f"Student : {args.student}")
    print(f"Arrays  : {ARRAYS_DIR}  ({len(npz_files)} files)")
    print(f"Output  : {OUT_CSV}\n")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(npz_files, 1):
        rows.append(process(path))
        if i % 200 == 0:
            print(f"  Processed {i}/{len(npz_files)} …")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUT_CSV}")

    js_kd    = [r["js_teacher_kd"]       for r in rows if not math.isnan(r["js_teacher_kd"])]
    js_bl    = [r["js_teacher_baseline"] for r in rows if not math.isnan(r["js_teacher_baseline"])]
    ss_kd    = [r["ssim_teacher_kd"]     for r in rows]
    ss_bl    = [r["ssim_teacher_baseline"] for r in rows]
    miou_kd  = [r["miou_teacher_kd"]     for r in rows]
    miou_bl  = [r["miou_teacher_baseline"] for r in rows]

    print(f"  mean js_teacher_kd         = {sum(js_kd) / len(js_kd):.6f}")
    print(f"  mean js_teacher_baseline   = {sum(js_bl) / len(js_bl):.6f}")
    print(f"  mean ssim_teacher_kd       = {sum(ss_kd) / len(ss_kd):.6f}")
    print(f"  mean ssim_teacher_baseline = {sum(ss_bl) / len(ss_bl):.6f}")
    print(f"  mean miou_teacher_kd       = {sum(miou_kd) / len(miou_kd):.6f}")
    print(f"  mean miou_teacher_baseline = {sum(miou_bl) / len(miou_bl):.6f}")


if __name__ == "__main__":
    main()
