"""
Compute summary statistics and produce paper-ready figures from divergence_scores.csv.
Outputs results/summary_stats.json and results/figures/figure{1,2,3}.png.
"""
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml
from scipy.stats import mannwhitneyu

# ── Config ──────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

random.seed(cfg["training"]["seed"])
np.random.seed(cfg["training"]["seed"])

DIVERGENCE_CSV = Path(cfg.get("paths", {}).get("divergence_csv",  "results/divergence_scores.csv"))
ACCURACY_CSV   = Path(cfg.get("paths", {}).get("accuracy_csv",    "results/accuracy.csv"))
STATS_JSON     = Path(cfg.get("paths", {}).get("summary_stats",   "results/summary_stats.json"))
FIGURES_DIR    = Path(cfg.get("paths", {}).get("figures_dir",     "results/figures"))

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STATS_JSON.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────────
def load_divergence(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))

def load_accuracy(path: Path) -> dict[str, float]:
    acc = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            acc[row["model_name"]] = float(row["test_accuracy"])
    return acc

rows = load_divergence(DIVERGENCE_CSV)
acc  = load_accuracy(ACCURACY_CSV)

# ── Extract columns ─────────────────────────────────────────────────────────────
js_kd  = np.array([float(r["js_teacher_kd"])            for r in rows])
js_bl  = np.array([float(r["js_teacher_baseline"])      for r in rows])
sp_kd  = np.array([float(r["spearman_teacher_kd"])      for r in rows])
sp_bl  = np.array([float(r["spearman_teacher_baseline"]) for r in rows])

teacher_correct  = np.array([int(r["teacher_correct"])  for r in rows])
kd_correct       = np.array([int(r["kd_correct"])       for r in rows])
baseline_correct = np.array([int(r["baseline_correct"]) for r in rows])

# ── Outcome group masks ──────────────────────────────────────────────────────────
# Group 1: both correct   Group 2: student wrong + teacher correct   Group 3: both wrong
def outcome_masks(student_correct: np.ndarray, teacher_correct: np.ndarray):
    both_correct   = (student_correct == 1) & (teacher_correct == 1)
    student_wrong  = (student_correct == 0) & (teacher_correct == 1)
    both_wrong     = (student_correct == 0) & (teacher_correct == 0)
    return both_correct, student_wrong, both_wrong

kd_bc,  kd_sw,  kd_bw  = outcome_masks(kd_correct,       teacher_correct)
bl_bc,  bl_sw,  bl_bw  = outcome_masks(baseline_correct,  teacher_correct)

def group_stats(js: np.ndarray, mask: np.ndarray) -> dict:
    subset = js[mask]
    if len(subset) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(float(subset.mean()), 6),
            "std":  round(float(subset.std()),  6),
            "n":    int(mask.sum())}

# ── Summary statistics ───────────────────────────────────────────────────────────
stats = {
    "js_divergence": {
        "kd_student": {
            "mean": round(float(js_kd.mean()), 6),
            "std":  round(float(js_kd.std()),  6),
        },
        "baseline": {
            "mean": round(float(js_bl.mean()), 6),
            "std":  round(float(js_bl.std()),  6),
        },
    },
    "spearman_r": {
        "kd_student": {
            "mean": round(float(sp_kd.mean()), 6),
            "std":  round(float(sp_kd.std()),  6),
        },
        "baseline": {
            "mean": round(float(sp_bl.mean()), 6),
            "std":  round(float(sp_bl.std()),  6),
        },
    },
    "js_by_outcome": {
        "kd_student": {
            "both_correct":           group_stats(js_kd, kd_bc),
            "student_wrong_teacher_correct": group_stats(js_kd, kd_sw),
            "both_wrong":             group_stats(js_kd, kd_bw),
        },
        "baseline": {
            "both_correct":           group_stats(js_bl, bl_bc),
            "student_wrong_teacher_correct": group_stats(js_bl, bl_sw),
            "both_wrong":             group_stats(js_bl, bl_bw),
        },
    },
    "top1_accuracy": {
        "teacher":    acc.get("teacher_resnet50"),
        "kd_student": acc.get("student_kd_resnet18"),
        "baseline":   acc.get("student_baseline_resnet18"),
    },
}

# ── Mann-Whitney U test (one-tailed: KD JS < baseline JS) ───────────────────────
mw_u, mw_p = mannwhitneyu(js_kd, js_bl, alternative="less")
stats["mann_whitney_u_statistic"] = float(mw_u)
stats["mann_whitney_p_value"]     = float(mw_p)
print(f"Mann-Whitney U = {mw_u:.1f},  p = {mw_p:.6e}")

with STATS_JSON.open("w") as f:
    json.dump(stats, f, indent=2)
print(f"Saved → {STATS_JSON}")

# ── Matplotlib style helpers ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

KD_COLOR   = "#2166ac"   # blue
BL_COLOR   = "#d6604d"   # orange-red
GROUP_COLORS = ["#4dac26", "#f4a582", "#bababa"]  # green / peach / grey

# ── Figure 1 — JS divergence bar chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4))

means = [js_kd.mean(), js_bl.mean()]
stds  = [js_kd.std(),  js_bl.std()]
x     = np.array([0, 1])
width = 0.5

bars = ax.bar(x, means, width, yerr=stds, capsize=5,
              color=[KD_COLOR, BL_COLOR],
              error_kw={"elinewidth": 1.2, "capthick": 1.2},
              zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(["KD Student\n(ResNet-18)", "Baseline Student\n(ResNet-18)"])
ax.set_ylabel("Mean JS Distance vs Teacher")
ax.set_title("Grad-CAM Spatial Divergence from Teacher", pad=10)
ax.set_ylim(0, max(means) * 1.55)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[bars.index(bar)] + 0.004,
            f"{m:.3f}", ha="center", va="bottom", fontsize=10)

ax.annotate("vs Teacher (ResNet-50)", xy=(0.5, 0.93), xycoords="axes fraction",
            ha="center", fontsize=9, color="grey")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "figure1_js_divergence_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIGURES_DIR / 'figure1_js_divergence_bar.png'}")

# ── Figure 2 — JS divergence by outcome group (KD student) ──────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4))

group_labels = ["Both Correct", "Student Wrong\n+ Teacher Correct", "Both Wrong"]
group_masks  = [kd_bc, kd_sw, kd_bw]

means_g = []
stds_g  = []
ns_g    = []
for mask in group_masks:
    sub = js_kd[mask]
    means_g.append(sub.mean() if len(sub) else 0.0)
    stds_g.append(sub.std()   if len(sub) else 0.0)
    ns_g.append(int(mask.sum()))

x = np.arange(len(group_labels))
bars = ax.bar(x, means_g, 0.55, yerr=stds_g, capsize=5,
              color=GROUP_COLORS,
              error_kw={"elinewidth": 1.2, "capthick": 1.2},
              zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylabel("Mean JS Distance vs Teacher")
ax.set_title("KD Student — JS Divergence by Prediction Outcome", pad=10)
ax.set_ylim(0, max(means_g) * 1.6)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)

for bar, m, n in zip(bars, means_g, ns_g):
    top = bar.get_height() + (bar.get_yerr() if hasattr(bar, "get_yerr") else 0)
    # Use the stds_g directly for annotation offset
    offset = stds_g[list(bars).index(bar)] + 0.006
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "figure2_js_by_outcome.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIGURES_DIR / 'figure2_js_by_outcome.png'}")

# ── Figure 3 — Spearman r distribution (overlaid histograms) ────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

bins = np.linspace(-0.5, 1.0, 31)

ax.hist(sp_kd, bins=bins, alpha=0.6, color=KD_COLOR, label="KD Student", density=True, zorder=3)
ax.hist(sp_bl, bins=bins, alpha=0.6, color=BL_COLOR,  label="Baseline Student", density=True, zorder=3)

# Vertical mean lines
ax.axvline(sp_kd.mean(), color=KD_COLOR, linestyle="--", linewidth=1.4,
           label=f"KD mean = {sp_kd.mean():.3f}")
ax.axvline(sp_bl.mean(), color=BL_COLOR,  linestyle="--", linewidth=1.4,
           label=f"Baseline mean = {sp_bl.mean():.3f}")

ax.set_xlabel("Spearman r  (Teacher vs Student Grad-CAM)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Spatial Rank Correlation with Teacher", pad=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "figure3_spearman_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIGURES_DIR / 'figure3_spearman_distribution.png'}")

# ── Sanity print ─────────────────────────────────────────────────────────────────
print(f"\nSanity check:")
print(f"  mean js_teacher_kd       = {js_kd.mean():.6f}")
print(f"  mean js_teacher_baseline = {js_bl.mean():.6f}")
