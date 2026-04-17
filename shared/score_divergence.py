"""
Compute Jensen-Shannon divergence and Spearman rank correlation between
teacher and each student Grad-CAM map for all 200 sampled images.
Outputs results/divergence_scores.csv with one row per image.
"""
import csv
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

# ── Config ──────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ARRAYS_DIR = Path(
    cfg.get("paths", {}).get("gradcam_arrays", "results/gradcam_full/arrays")
)
OUT_CSV = Path("results/divergence_scores.csv")

FIELDNAMES = [
    "filename",
    "true_label",
    "teacher_pred", "kd_pred", "baseline_pred",
    "teacher_correct", "kd_correct", "baseline_correct",
    "js_teacher_kd", "js_teacher_baseline",
    "spearman_teacher_kd", "spearman_teacher_baseline",
]


def js(a: np.ndarray, b: np.ndarray) -> float:
    """JS distance between two non-negative arrays (need not sum to 1 here,
    but after Step 5 normalization they already do)."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(jensenshannon(a, b))


def spear(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


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
        "filename":            npz_path.name,
        "true_label":          true_label,
        "teacher_pred":        teacher_pred,
        "kd_pred":             kd_pred,
        "baseline_pred":       baseline_pred,
        "teacher_correct":     int(teacher_pred  == true_label),
        "kd_correct":          int(kd_pred       == true_label),
        "baseline_correct":    int(baseline_pred == true_label),
        "js_teacher_kd":       js(t, kd),
        "js_teacher_baseline": js(t, bl),
        "spearman_teacher_kd":       spear(t, kd),
        "spearman_teacher_baseline": spear(t, bl),
    }


def main() -> None:
    npz_files = sorted(ARRAYS_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {ARRAYS_DIR}")

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

    js_kd  = [r["js_teacher_kd"]       for r in rows if not math.isnan(r["js_teacher_kd"])]
    js_bl  = [r["js_teacher_baseline"] for r in rows if not math.isnan(r["js_teacher_baseline"])]
    print(f"  mean js_teacher_kd       = {sum(js_kd)  / len(js_kd):.6f}")
    print(f"  mean js_teacher_baseline = {sum(js_bl) / len(js_bl):.6f}")


if __name__ == "__main__":
    main()
