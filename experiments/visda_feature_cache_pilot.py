import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# ============================================================
# VISDA FEATURE EXTRACTION PILOT
# ============================================================

SEED = 42

ROOT = Path("data/visda")
SOURCE_ROOT = ROOT / "train"
TARGET_ROOT = ROOT / "validation"

SOURCE_SAMPLES = 1000
TARGET_SAMPLES = 1000

BATCH_SIZE = 16

IMAGE_SIZE = 224
RESIZE_SIZE = 256


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# TRANSFORM
# ============================================================

def build_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# DATASET
# ============================================================

def build_subset(root, samples):
    dataset = datasets.ImageFolder(
        root=str(root),
        transform=build_transform(),
    )

    if samples > len(dataset):
        samples = len(dataset)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:samples].tolist()

    return Subset(dataset, indices)


# ============================================================
# MODEL
# ============================================================

def build_feature_extractor(device):
    print("Loading ImageNet-pretrained ResNet-50...")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # Remove the classification layer.
    feature_extractor = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.eval()
    feature_extractor.to(device)

    for parameter in feature_extractor.parameters():
        parameter.requires_grad_(False)

    return feature_extractor


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    loader,
    device,
    name,
):
    features = []
    labels = []

    total = len(loader.dataset)
    processed = 0

    start = time.perf_counter()

    for batch_idx, (images, batch_labels) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(device)

        batch_features = model(images)

        # ResNet-50 output:
        # [B, 2048, 1, 1]
        batch_features = batch_features.flatten(1)

        features.append(
            batch_features.cpu()
        )

        labels.append(
            batch_labels.clone()
        )

        processed += images.size(0)

        if (
            batch_idx % 10 == 0
            or processed >= total
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            images_per_second = (
                processed
                / max(elapsed, 1e-8)
            )

            remaining = total - processed

            eta = (
                remaining
                / max(
                    images_per_second,
                    1e-8,
                )
            )

            print(
                f"{name} | "
                f"{processed}/{total} "
                f"| {images_per_second:.2f} img/s "
                f"| ETA {eta:.1f}s"
            )

    features = torch.cat(
        features,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        features,
        labels,
        elapsed,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("VISDA-2017 RESNET-50 FEATURE EXTRACTION PILOT")
    print("=" * 70)

    print(f"device={device}")
    print(f"seed={SEED}")
    print(f"source_samples={SOURCE_SAMPLES}")
    print(f"target_samples={TARGET_SAMPLES}")
    print(f"batch_size={BATCH_SIZE}")

    print()
    print("Building datasets...")

    source_dataset = build_subset(
        SOURCE_ROOT,
        SOURCE_SAMPLES,
    )

    target_dataset = build_subset(
        TARGET_ROOT,
        TARGET_SAMPLES,
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(
        f"Source pilot samples: "
        f"{len(source_dataset)}"
    )

    print(
        f"Target pilot samples: "
        f"{len(target_dataset)}"
    )

    print()
    print("Building frozen feature extractor...")

    model = build_feature_extractor(
        device
    )

    print()
    print("=" * 70)
    print("SOURCE EXTRACTION")
    print("=" * 70)

    source_features, source_labels, source_time = (
        extract_features(
            model,
            source_loader,
            device,
            "SOURCE",
        )
    )

    print()
    print("=" * 70)
    print("TARGET EXTRACTION")
    print("=" * 70)

    target_features, target_labels, target_time = (
        extract_features(
            model,
            target_loader,
            device,
            "TARGET",
        )
    )

    print()
    print("=" * 70)
    print("PILOT RESULT")
    print("=" * 70)

    print(
        f"Source feature shape: "
        f"{tuple(source_features.shape)}"
    )

    print(
        f"Target feature shape: "
        f"{tuple(target_features.shape)}"
    )

    print(
        f"Source extraction time: "
        f"{source_time:.2f}s"
    )

    print(
        f"Target extraction time: "
        f"{target_time:.2f}s"
    )

    total_images = (
        SOURCE_SAMPLES
        + TARGET_SAMPLES
    )

    total_time = (
        source_time
        + target_time
    )

    images_per_second = (
        total_images
        / max(total_time, 1e-8)
    )

    print(
        f"Combined throughput: "
        f"{images_per_second:.2f} img/s"
    )

    # Approximate full dataset extraction.
    full_source = 152397
    full_target = 55388
    full_total = (
        full_source
        + full_target
    )

    estimated_full_seconds = (
        full_total
        / max(images_per_second, 1e-8)
    )

    print()
    print(
        "=" * 70
    )
    print("FULL-CACHE ESTIMATE")
    print(
        "=" * 70
    )

    print(
        f"Estimated full images: "
        f"{full_total}"
    )

    print(
        f"Estimated full extraction time: "
        f"{estimated_full_seconds / 3600:.2f} hours"
    )

    # Feature storage estimate.
    feature_elements = (
        full_total * 2048
    )

    float32_gb = (
        feature_elements
        * 4
        / (1024 ** 3)
    )

    float16_gb = (
        feature_elements
        * 2
        / (1024 ** 3)
    )

    print()
    print(
        f"Estimated float32 feature storage: "
        f"{float32_gb:.2f} GB"
    )

    print(
        f"Estimated float16 feature storage: "
        f"{float16_gb:.2f} GB"
    )

    print()
    print(
        "The pilot is complete."
    )

    print(
        "Do NOT start full extraction yet."
    )


if __name__ == "__main__":
    main()