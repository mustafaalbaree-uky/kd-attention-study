"""
Add floor metrics to summary_stats.json in-place.

Reads {student}_floor_scores.csv, computes mean/std for each metric, and writes
six new keys into {student}_summary_stats.json without touching existing keys.

Usage:
    python shared/add_floor_to_summary.py --student resnet18
    python shared/add_floor_to_summary.py --student mobilenet
    python shared/add_floor_to_summary.py --student densenet
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent   # project root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True,
                        choices=["resnet18", "mobilenet", "densenet"])
    args = parser.parse_args()

    results_dir  = _ROOT / "students" / args.student / "results"
    floor_csv    = results_dir / f"{args.student}_floor_scores.csv"
    summary_json = results_dir / f"{args.student}_summary_stats.json"

    with floor_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    js_vals   = np.array([float(r["js_floor"])       for r in rows])
    sp_vals   = np.array([float(r["spearman_floor"]) for r in rows])
    ssim_vals = np.array([float(r["ssim_floor"])     for r in rows])

    with summary_json.open() as f:
        stats = json.load(f)

    stats["floor_js_mean"]       = float(np.mean(js_vals))
    stats["floor_js_std"]        = float(np.std(js_vals))
    stats["floor_spearman_mean"] = float(np.mean(sp_vals))
    stats["floor_spearman_std"]  = float(np.std(sp_vals))
    stats["floor_ssim_mean"]     = float(np.mean(ssim_vals))
    stats["floor_ssim_std"]      = float(np.std(ssim_vals))

    with summary_json.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"{args.student}: floor keys added to {summary_json}")
    print(f"  floor_js_mean       = {stats['floor_js_mean']:.4f}")
    print(f"  floor_spearman_mean = {stats['floor_spearman_mean']:.4f}")
    print(f"  floor_ssim_mean     = {stats['floor_ssim_mean']:.4f}")


if __name__ == "__main__":
    main()
