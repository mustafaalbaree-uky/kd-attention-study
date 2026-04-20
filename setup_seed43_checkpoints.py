"""
Copy seed-43 baseline checkpoints from the attached Kaggle dataset to each
student's checkpoints/ directory. Run once before generate_gradcam_floor.py.

Usage (from project root):
    python setup_seed43_checkpoints.py
"""
import shutil
from pathlib import Path

_ROOT = Path(__file__).parent

def find_dataset_dir() -> Path:
    # Search all of /kaggle/input/ for any directory containing resnet18_baseline_seed43.pth
    kaggle_input = Path("/kaggle/input")
    matches = list(kaggle_input.rglob("resnet18_baseline_seed43.pth"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        "resnet18_baseline_seed43.pth not found anywhere under /kaggle/input/.\n"
        "Check that the checkpoints_seed43 dataset is attached to this notebook."
    )


def main():
    dataset_dir = find_dataset_dir()
    print(f"Dataset : {dataset_dir}\n")

    for arch in ["resnet18", "mobilenet", "densenet"]:
        filename = f"{arch}_baseline_seed43.pth"
        src = dataset_dir / filename
        dst = _ROOT / "students" / arch / "checkpoints" / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copied  : {filename}")

    print("\nAll seed-43 checkpoints in place.")


if __name__ == "__main__":
    main()
