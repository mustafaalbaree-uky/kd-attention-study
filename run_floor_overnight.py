"""
Master script — floor baseline computation, end-to-end.

Steps:
  1. Train seed-43 baselines (resnet18, mobilenet, densenet)
  2. Generate Grad-CAM floor arrays (arrays_seed43/ per architecture)
  3. Score floor metrics (JS, Spearman, SSIM between seed-42 and seed-43)
  4. Update summary_stats.json with floor keys
  5. Zip all outputs

Run from the repo root on Kaggle:
    python run_floor_overnight.py
"""
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

_ROOT    = Path(__file__).parent
_ZIP_OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else _ROOT / "output"
_ZIP_OUT.mkdir(exist_ok=True)

ARCHS = ["resnet18", "mobilenet", "densenet"]


def run(cmd_parts, desc):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")
    subprocess.run([sys.executable] + cmd_parts, cwd=str(_ROOT), check=True)


def main():
    # ── Step 1: Train seed-43 baselines ───────────────────────────────────────
    for arch in ARCHS:
        run([f"students/{arch}/train_baseline_seed43.py"],
            f"Training seed-43 baseline: {arch}")

    # ── Step 2: Generate Grad-CAM floor arrays ────────────────────────────────
    for arch in ARCHS:
        run([f"students/{arch}/generate_gradcam_floor.py"],
            f"Grad-CAM floor arrays: {arch}")

    # ── Step 3: Score floor metrics ───────────────────────────────────────────
    for arch in ARCHS:
        run(["shared/score_floor.py", "--student", arch],
            f"Scoring floor metrics: {arch}")

    # ── Step 4: Update summary stats ──────────────────────────────────────────
    for arch in ARCHS:
        run(["shared/add_floor_to_summary.py", "--student", arch],
            f"Updating summary_stats.json: {arch}")

    # ── Step 5: Zip outputs ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Zipping floor_results.zip")
    print(f"{'=' * 60}")
    with zipfile.ZipFile(_ZIP_OUT / "floor_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for arch in ARCHS:
            ckpt = _ROOT / "students" / arch / "checkpoints" / f"{arch}_baseline_seed43.pth"
            log  = _ROOT / "students" / arch / "results" / f"{arch}_baseline_seed43_training_log.csv"
            floor_csv  = _ROOT / "students" / arch / "results" / "floor_scores.csv"
            summary    = _ROOT / "students" / arch / "results" / "summary_stats.json"
            for p in [ckpt, log, floor_csv, summary]:
                zf.write(p, p.relative_to(_ROOT))
        for md in [
            _ROOT / "humaninfo" / "research_design.md",
            _ROOT / "humaninfo" / "claude_code_pm_guide_resnet18.md",
            _ROOT / "humaninfo" / "claude_code_pm_guide_mobilenet.md",
            _ROOT / "humaninfo" / "claude_code_pm_guide_densenet.md",
        ]:
            zf.write(md, md.relative_to(_ROOT))
    print(f"Saved -> {_ZIP_OUT / 'floor_results.zip'}")

    print(f"\n{'=' * 60}")
    print("  Zipping floor_arrays_seed43.zip")
    print(f"{'=' * 60}")
    with zipfile.ZipFile(_ZIP_OUT / "floor_arrays_seed43.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for arch in ARCHS:
            arrays_dir = _ROOT / "students" / arch / "results" / "gradcam_full" / "arrays_seed43"
            for npz in sorted(arrays_dir.glob("*.npz")):
                zf.write(npz, npz.relative_to(_ROOT))
    print(f"Saved -> {_ZIP_OUT / 'floor_arrays_seed43.zip'}")

    # ── Completion summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  COMPLETION SUMMARY")
    print(f"{'=' * 60}")
    for arch in ARCHS:
        summary_path = _ROOT / "students" / arch / "results" / "summary_stats.json"
        with open(summary_path) as f:
            stats = json.load(f)

        log_path = _ROOT / "students" / arch / "results" / f"{arch}_baseline_seed43_training_log.csv"
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))
        best_val_acc = max(float(r["val_acc"]) for r in rows)

        print(f"\n  {arch}:")
        print(f"    seed-43 val_acc     = {best_val_acc:.4f}")
        print(f"    floor_js_mean       = {stats.get('floor_js_mean', 'N/A'):.4f}")
        print(f"    floor_spearman_mean = {stats.get('floor_spearman_mean', 'N/A'):.4f}")
        print(f"    floor_ssim_mean     = {stats.get('floor_ssim_mean', 'N/A'):.4f}")

    print(f"\nAll outputs in {_ZIP_OUT}")


if __name__ == "__main__":
    main()
