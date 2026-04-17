"""
Generate Grad-CAM heatmaps for teacher (ResNet-50), KD student (ResNet-18),
and baseline student (ResNet-18) on 200 stratified test images from ImageNette.
Outputs side-by-side figures and raw sum-normalized numpy arrays.
"""
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from PIL import Image as PILImage
from pytorch_grad_cam import GradCAM
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent          # students/resnet18/
_ROOT = _HERE.parent.parent            # project root

with open(_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED             = cfg["training"]["seed"]
NUM_CLASSES      = cfg["training"]["num_classes"]
IMG_SIZE         = cfg["dataset"]["image_size"]
IMAGES_PER_CLASS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DATA_DIR       = _ROOT / "data"
_IMAGENETTE_DIR = _DATA_DIR / "imagenette2-320"
_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

CKPT_TEACHER  = _ROOT / "teacher" / "checkpoints" / "teacher_finetuned.pth"
CKPT_KD       = _HERE / "checkpoints" / "resnet18_kd.pth"
CKPT_BASELINE = _HERE / "checkpoints" / "resnet18_baseline.pth"

OUT_FIGS   = _HERE / "results" / "gradcam" / "figures"
OUT_ARRAYS = _HERE / "results" / "gradcam" / "arrays"

IMAGENETTE_LABELS = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
])

try:
    _BILINEAR = PILImage.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = PILImage.BILINEAR


# ── Data ────────────────────────────────────────────────────────────────────────
def download_imagenette() -> None:
    if _IMAGENETTE_DIR.exists():
        return
    _DATA_DIR.mkdir(exist_ok=True)
    tgz_path = _DATA_DIR / "imagenette2-320.tgz"
    print("Downloading ImageNette (~330 MB) …")

    def _progress(n, bs, ts):
        print(f"\r  {min(n * bs / ts * 100, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(_IMAGENETTE_URL, tgz_path, reporthook=_progress)
    print()
    print("Extracting …")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(_DATA_DIR)
    tgz_path.unlink()
    print("Done.\n")


def stratified_sample(dataset: tv_datasets.ImageFolder, n_per_class: int) -> list:
    np.random.seed(SEED)
    by_class: dict = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    sampled = []
    for label in sorted(by_class):
        chosen = np.random.choice(by_class[label], size=n_per_class, replace=False)
        sampled.extend(chosen.tolist())
    return sampled


# ── Models ──────────────────────────────────────────────────────────────────────
def load_resnet50(ckpt_path) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.eval().to(DEVICE)


def load_resnet18(ckpt_path) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.eval().to(DEVICE)


# ── Grad-CAM ────────────────────────────────────────────────────────────────────
def compute_raw_gradcam(cam_obj, model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
    pred_idx = int(logits.argmax(1).item())

    cam_obj(input_tensor=input_tensor, targets=None)

    acts  = cam_obj.activations_and_grads.activations[0].cpu().numpy()
    grads = cam_obj.activations_and_grads.gradients[0].cpu().numpy()

    weights = np.mean(grads, axis=(2, 3), keepdims=True)
    raw = np.sum(weights * acts, axis=1)[0]
    raw = np.maximum(raw, 0.0)

    return raw, pred_idx


def normalize_map(raw: np.ndarray) -> np.ndarray:
    total = raw.sum()
    if total == 0.0:
        return np.ones_like(raw) / raw.size
    return raw / total


# ── Visualization ───────────────────────────────────────────────────────────────
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * _STD + _MEAN, 0.0, 1.0).astype(np.float32)


def make_overlay(img_hw3: np.ndarray, raw_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    H, W = img_hw3.shape[:2]
    map_u8 = (raw_map / (raw_map.max() + 1e-8) * 255).astype(np.uint8)
    map_resized = np.array(
        PILImage.fromarray(map_u8).resize((W, H), _BILINEAR)
    ) / 255.0
    heatmap = plt.get_cmap("jet")(map_resized)[:, :, :3]
    return np.clip(alpha * heatmap + (1.0 - alpha) * img_hw3, 0.0, 1.0).astype(np.float32)


def save_figure(img_hw3, raw_maps, pred_names, true_class_name, stem) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_hw3)
    axes[0].set_title(f"Original\n(true: {true_class_name})", fontsize=9)
    axes[0].axis("off")

    panels = [
        ("teacher",    "Teacher (ResNet-50)"),
        ("kd_student", "KD Student (ResNet-18)"),
        ("baseline",   "Baseline (ResNet-18)"),
    ]
    for ax, (key, label) in zip(axes[1:], panels):
        ax.imshow(make_overlay(img_hw3, raw_maps[key]))
        ax.set_title(f"{label}\npred: {pred_names[key]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(OUT_FIGS / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}")
    download_imagenette()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_ARRAYS.mkdir(parents=True, exist_ok=True)

    val_dir = _IMAGENETTE_DIR / "val"
    dataset = tv_datasets.ImageFolder(val_dir, transform=transform)
    class_names = [IMAGENETTE_LABELS.get(c, c) for c in dataset.classes]
    print(f"Classes: {class_names}")

    indices = stratified_sample(dataset, IMAGES_PER_CLASS)
    print(f"Sampled {len(indices)} images ({IMAGES_PER_CLASS} per class)\n")

    print("Loading models …")
    teacher    = load_resnet50(CKPT_TEACHER)
    kd_student = load_resnet18(CKPT_KD)
    baseline   = load_resnet18(CKPT_BASELINE)

    cam_teacher  = GradCAM(model=teacher,    target_layers=[teacher.layer4[-1]])
    cam_kd       = GradCAM(model=kd_student, target_layers=[kd_student.layer4[-1]])
    cam_baseline = GradCAM(model=baseline,   target_layers=[baseline.layer4[-1]])

    model_cams = [
        ("teacher",    teacher,    cam_teacher),
        ("kd_student", kd_student, cam_kd),
        ("baseline",   baseline,   cam_baseline),
    ]

    class_counter: dict = {}

    for ds_idx in tqdm(indices, desc="Generating Grad-CAMs"):
        img_tensor, true_label = dataset[ds_idx]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        true_class = class_names[true_label]
        class_counter[true_class] = class_counter.get(true_class, 0) + 1
        stem = f"{true_class}_{class_counter[true_class]:03d}"

        img_hw3 = denormalize(img_tensor)

        raw_maps:     dict = {}
        pred_names:   dict = {}
        pred_indices: dict = {}

        for key, model, cam_obj in model_cams:
            raw, pred_idx = compute_raw_gradcam(cam_obj, model, input_tensor)
            raw_maps[key]     = raw
            pred_indices[key] = pred_idx
            pred_names[key]   = class_names[pred_idx]

        norm_maps = {k: normalize_map(v) for k, v in raw_maps.items()}

        np.savez(
            OUT_ARRAYS / f"{stem}.npz",
            teacher=norm_maps["teacher"],
            kd_student=norm_maps["kd_student"],
            baseline=norm_maps["baseline"],
            true_label=np.int64(true_label),
            teacher_pred=np.int64(pred_indices["teacher"]),
            kd_pred=np.int64(pred_indices["kd_student"]),
            baseline_pred=np.int64(pred_indices["baseline"]),
        )

        save_figure(img_hw3, raw_maps, pred_names, true_class, stem)

    n_figs   = len(list(OUT_FIGS.glob("*.png")))
    n_arrays = len(list(OUT_ARRAYS.glob("*.npz")))
    print(f"\nDone.  Figures: {n_figs}  |  Arrays: {n_arrays}")


if __name__ == "__main__":
    main()
