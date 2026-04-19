"""
Copy seed-43 baseline checkpoints from the attached Kaggle dataset to each
student's checkpoints/ directory. Run once before generate_gradcam_floor.py.

Usage (from project root):
    python setup_seed43_checkpoints.py
"""
import shutil
from pathlib import Path

_ROOT = Path(__file__).parent

_CANDIDATE_DIRS = [
    Path("/kaggle/input/checkpoints-seed43"),
    Path("/kaggle/input/checkpoints_seed43"),
    Path("/kaggle/input/datasets/mustafaalbaree/checkpoints-seed43"),
    Path("/kaggle/input/datasets/mustafaalbaree/checkpoints_seed43"),
]


def find_dataset_dir() -> Path:
    for d in _CANDIDATE_DIRS:
        if d.exists():
            return d
    raise FileNotFoundError(
        "Seed-43 dataset not found. Tried:\n" +
        "\n".join(f"  {d}" for d in _CANDIDATE_DIRS) +
        "\nRun !ls /kaggle/input/ to find the correct path."
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
