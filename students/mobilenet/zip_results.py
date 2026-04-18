"""
Zip students/mobilenet/results/ for download.
On Kaggle: writes to /kaggle/working/mobilenet_results.zip (visible in Output panel).
Locally:   writes to the project root.

Usage (from any directory):
    python students/mobilenet/zip_results.py
"""
import shutil
from pathlib import Path

_HERE = Path(__file__).parent          # students/mobilenet/
_ROOT = _HERE.parent.parent            # project root

results_dir = _HERE / "results"

# Write to /kaggle/working/ on Kaggle, project root otherwise
kaggle_out = Path("/kaggle/working")
out_dir    = kaggle_out if kaggle_out.exists() else _ROOT
out_zip    = out_dir / "mobilenet_results"

shutil.make_archive(str(out_zip), "zip", str(results_dir))
print(f"Saved → {out_zip}.zip")
if kaggle_out.exists():
    print("Download mobilenet_results.zip from the Output panel.")
