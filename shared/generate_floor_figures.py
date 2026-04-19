"""
Generate 10 sample floor figures (one per ImageNette class) comparing seed-42
and seed-43 baseline Grad-CAM maps. Reads from arrays_seed43/*.npz — run
after generate_gradcam_floor.py.

Usage (from project root):
    python shared/generate_floor_figures.py --student resnet18
    python shared/generate_floor_figures.py --student mobilenet
    python shared/generate_floor_figures.py --student densenet
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
from PIL import Image as PILImage

_HERE = Path(__file__).parent    # shared/
_ROOT = _HERE.parent             # project root

parser = argparse.ArgumentParser()
parser.add_argument("--student", required=True, choices=["resnet18", "mobilenet", "densenet"])
args = parser.parse_args()

STUDENT_DIR    = _ROOT / "students" / args.student
ARRAYS_DIR     = STUDENT_DIR / "results" / "gradcam_full" / "arrays_seed43"
FIGURES_DIR    = STUDENT_DIR / "results" / "gradcam_full" / "figures_seed43"
DATA_DIR       = _ROOT / "data"
IMAGENETTE_DIR = DATA_DIR / "imagenette2-320"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

IMAGENETTE_LABELS = {
    "n01440764": "tench",           "n02102040": "english_springer",
    "n02979186": "cassette_player", "n03000684": "chain_saw",
    "n03028079": "church",          "n03394916": "french_horn",
    "n03417042": "garbage_truck",   "n03425413": "gas_pump",
    "n03445777": "golf_ball",       "n03888257": "parachute",
}

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN.tolist(), _STD.tolist()),
])


def download_imagenette():
    if IMAGENETTE_DIR.exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    tgz = DATA_DIR / "imagenette2-320.tgz"
    print("Downloading ImageNette (~330 MB) …")
    urllib.request.urlretrieve(IMAGENETTE_URL, tgz,
        reporthook=lambda n, bs, ts: print(f"\r  {min(n*bs/ts*100,100):.1f}%", end="", flush=True))
    print("\nExtracting …")
    with tarfile.open(tgz) as t:
        t.extractall(DATA_DIR)
    tgz.unlink()
    print("Done.\n")


def denormalize(tensor):
    return np.clip(tensor.numpy().transpose(1, 2, 0) * _STD + _MEAN, 0, 1)


def make_overlay(img_hw3, raw_map, alpha=0.5):
    H, W = img_hw3.shape[:2]
    u8 = (raw_map / (raw_map.max() + 1e-8) * 255).astype(np.uint8)
    resized = np.array(PILImage.fromarray(u8).resize((W, H), PILImage.BILINEAR)) / 255.0
    heatmap = plt.get_cmap("jet")(resized)[:, :, :3]
    return np.clip(alpha * heatmap + (1 - alpha) * img_hw3, 0, 1).astype(np.float32)


def main():
    download_imagenette()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = tv_datasets.ImageFolder(IMAGENETTE_DIR / "val", transform=transform)
    class_names = [IMAGENETTE_LABELS.get(c, c) for c in dataset.classes]

    # build per-class ordered index list (matches generate_gradcam_floor.py ordering)
    class_to_indices: dict[str, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(class_names[label], []).append(idx)

    print(f"Student  : {args.student}")
    print(f"Arrays   : {ARRAYS_DIR}")
    print(f"Figures  : {FIGURES_DIR}\n")

    n_saved = 0
    for cls_name in sorted(class_to_indices.keys()):
        npz_path = ARRAYS_DIR / f"{cls_name}_0001.npz"
        if not npz_path.exists():
            print(f"  Missing: {npz_path.name} — skipping")
            continue

        img_tensor, _ = dataset[class_to_indices[cls_name][0]]
        img_hw3 = denormalize(img_tensor)

        d = np.load(npz_path)
        map42 = d["baseline_seed42"].reshape(7, 7)
        map43 = d["baseline_seed43"].reshape(7, 7)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_hw3)
        axes[0].set_title(f"Original\n({cls_name})", fontsize=9)
        axes[1].imshow(make_overlay(img_hw3, map42))
        axes[1].set_title("Baseline seed-42", fontsize=9)
        axes[2].imshow(make_overlay(img_hw3, map43))
        axes[2].set_title("Baseline seed-43", fontsize=9)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"{cls_name}_0001.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1
        print(f"  {cls_name}")

    print(f"\nSaved {n_saved} figures → {FIGURES_DIR}")


if __name__ == "__main__":
    main()
