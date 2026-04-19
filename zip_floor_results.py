"""
Zip all floor results for download from Kaggle output panel.
Run after score_floor.py, summarize.py, and generate_floor_figures.py
have been run for all three architectures.

Usage (from project root):
    python zip_floor_results.py

Produces in /kaggle/working/ (or zip_output/ when run locally):
    {arch}_arrays_seed43.zip     ← 3,925 .npz files per architecture
    {arch}_figures_seed43.zip    ← 10 sample comparison PNGs
    {arch}_floor_scores.csv      ← per-image floor metrics (js, spearman, ssim, miou)
    {arch}_summary_stats.json    ← full summary including floor means
"""
import os
import shutil
from pathlib import Path

_ROOT   = Path(__file__).parent
OUT_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else _ROOT / "zip_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for arch in ["resnet18", "mobilenet", "densenet"]:
    results  = _ROOT / "students" / arch / "results"
    gradcam  = results / "gradcam_full"

    arrays_zip = OUT_DIR / f"{arch}_arrays_seed43"
    shutil.make_archive(str(arrays_zip), "zip", str(gradcam / "arrays_seed43"))
    sz = os.path.getsize(f"{arrays_zip}.zip") / 1e6
    print(f"{arch} arrays   : {arrays_zip}.zip  ({sz:.1f} MB)")

    figs_zip = OUT_DIR / f"{arch}_figures_seed43"
    shutil.make_archive(str(figs_zip), "zip", str(gradcam / "figures_seed43"))
    sz2 = os.path.getsize(f"{figs_zip}.zip") / 1e6
    print(f"{arch} figures  : {figs_zip}.zip  ({sz2:.1f} MB)")

    shutil.copy(results / f"{arch}_floor_scores.csv",   OUT_DIR / f"{arch}_floor_scores.csv")
    shutil.copy(results / f"{arch}_summary_stats.json", OUT_DIR / f"{arch}_summary_stats.json")
    print(f"{arch} csv/json : copied\n")

print(f"All outputs → {OUT_DIR}")
