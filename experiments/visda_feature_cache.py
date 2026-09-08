import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# VISDA-2017 FULL FEATURE CACHE
# Frozen ImageNet ResNet-50
# CPU-friendly, resumable, chunked, FP16 storage
# ============================================================

SEED = 42

ROOT = Path("data/visda")

SOURCE_ROOT = ROOT / "train"
TARGET_ROOT = ROOT / "validation"

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
TARGET_CACHE.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
CHUNK_SIZE = 5000

IMAGE_SIZE = 224
RESIZE_SIZE = 256

NUM_CLASSES = 12

CLASSES = [
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
]

EXPECTED_FEATURE_DIM = 2048


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)


# ============================================================
# DATASET WITH INDICES
# ============================================================

class IndexedImageFolder(Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(
            root=str(root),
            transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


# ============================================================
# TRANSFORM
# ============================================================

def build_transform():
    # Deterministic feature extraction.
    #
    # We intentionally do NOT use random crop/flip here.
    # The feature cache must be reproducible and resumable.
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
# MODEL
# ============================================================

def build_feature_extractor():
    print("Loading ImageNet-pretrained ResNet-50...")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # Remove the final ImageNet classifier.
    feature_extractor = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.eval()

    for parameter in feature_extractor.parameters():
        parameter.requires_grad_(False)

    return feature_extractor


# ============================================================
# ATOMIC SAVE
# ============================================================

def atomic_torch_save(obj, path):
    path = Path(path)

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        obj,
        temp_path,
    )

    temp_path.replace(path)


# ============================================================
# METADATA
# ============================================================

def metadata_path(cache_dir):
    return cache_dir / "metadata.json"


def write_metadata(
    cache_dir,
    split,
    dataset,
):
    metadata = {
        "split": split,
        "seed": SEED,
        "root": str(dataset.dataset.root),
        "num_samples": len(dataset),
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "class_to_idx": dataset.dataset.class_to_idx,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "feature_dtype": "float16",
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "resize": RESIZE_SIZE,
        "crop": IMAGE_SIZE,
        "normalization": {
            "mean": [
                0.485,
                0.456,
                0.406,
            ],
            "std": [
                0.229,
                0.224,
                0.225,
            ],
        },
        "feature_definition": (
            "Frozen ImageNet-pretrained ResNet-50 "
            "penultimate pooled representation"
        ),
        "target_labels_training": (
            "Labels are stored for evaluation only and "
            "must not be used by adaptation."
        ),
    }

    with open(
        metadata_path(cache_dir),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )


# ============================================================
# CHUNK PATH
# ============================================================

def chunk_path(cache_dir, chunk_index):
    return cache_dir / (
        f"chunk_{chunk_index:05d}.pt"
    )


# ============================================================
# VALIDATE EXISTING CHUNK
# ============================================================

def is_valid_chunk(path, expected_count):
    if not path.exists():
        return False

    try:
        payload = torch.load(
            path,
            map_location="cpu",
        )

        features = payload["features"]
        labels = payload["labels"]
        paths = payload["paths"]

        if features.ndim != 2:
            return False

        if features.shape[0] != expected_count:
            return False

        if features.shape[1] != EXPECTED_FEATURE_DIM:
            return False

        if features.dtype != torch.float16:
            return False

        if labels.shape[0] != expected_count:
            return False

        if len(paths) != expected_count:
            return False

        return True

    except Exception:
        return False


# ============================================================
# SOURCE/TARGET EXTRACTION
# ============================================================

@torch.no_grad()
def cache_split(
    split,
    root,
    cache_dir,
    model,
    device,
):
    print()
    print("=" * 70)
    print(f"CACHING {split.upper()}")
    print("=" * 70)

    transform = build_transform()

    dataset = IndexedImageFolder(
        root=root,
        transform=transform,
    )

    write_metadata(
        cache_dir,
        split,
        dataset,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    total_samples = len(dataset)

    num_chunks = (
        total_samples + CHUNK_SIZE - 1
    ) // CHUNK_SIZE

    print(
        f"{split} samples: "
        f"{total_samples}"
    )

    print(
        f"Chunks: "
        f"{num_chunks}"
    )

    split_start = time.perf_counter()

    completed_chunks = 0

    # Accumulate one chunk at a time.
    feature_buffer = []
    label_buffer = []
    path_buffer = []

    current_chunk_index = 0
    current_chunk_start = 0

    processed = 0

    for images, labels, indices in loader:
        images = images.to(device)

        features = model(images)

        # ResNet-50:
        # [B, 2048, 1, 1]
        features = features.flatten(1)

        if features.shape[1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                "Unexpected feature dimension: "
                f"{features.shape[1]}"
            )

        # Convert ONLY for storage.
        features = features.cpu().half()
        labels = labels.cpu()
        indices = indices.cpu()

        for i in range(features.shape[0]):
            feature_buffer.append(
                features[i]
            )

            label_buffer.append(
                labels[i]
            )

            dataset_index = int(
                indices[i].item()
            )

            original_path = Path(
                dataset.dataset.samples[
                    dataset_index
                ][0]
            )

            relative_path = original_path.relative_to(
                Path(dataset.dataset.root)
            )

            path_buffer.append(
                str(relative_path)
            )

            processed += 1

            # Complete current chunk.
            chunk_target_end = min(
                current_chunk_start + CHUNK_SIZE,
                total_samples,
            )

            if processed == chunk_target_end:
                chunk_file = chunk_path(
                    cache_dir,
                    current_chunk_index,
                )

                expected_count = (
                    chunk_target_end
                    - current_chunk_start
                )

                if is_valid_chunk(
                    chunk_file,
                    expected_count,
                ):
                    print(
                        f"{split} chunk "
                        f"{current_chunk_index:05d} "
                        f"already valid; skipping save."
                    )
                else:
                    chunk_features = torch.stack(
                        feature_buffer,
                        dim=0,
                    )

                    chunk_labels = torch.stack(
                        label_buffer,
                        dim=0,
                    ).long()

                    payload = {
                        "features": chunk_features,
                        "labels": chunk_labels,
                        "paths": list(
                            path_buffer
                        ),
                        "start_index":
                            current_chunk_start,
                        "end_index":
                            chunk_target_end,
                        "split":
                            split,
                        "feature_dim":
                            EXPECTED_FEATURE_DIM,
                    }

                    atomic_torch_save(
                        payload,
                        chunk_file,
                    )

                    print(
                        f"{split} chunk "
                        f"{current_chunk_index:05d} "
                        f"saved | "
                        f"{expected_count} samples | "
                        f"{chunk_file}"
                    )

                elapsed = (
                    time.perf_counter()
                    - split_start
                )

                rate = (
                    processed
                    / max(elapsed, 1e-8)
                )

                remaining = (
                    total_samples
                    - processed
                )

                eta = (
                    remaining
                    / max(rate, 1e-8)
                )

                print(
                    f"{split} | "
                    f"{processed}/{total_samples} "
                    f"| {rate:.2f} img/s "
                    f"| ETA {eta / 60:.1f} min"
                )

                completed_chunks += 1
                current_chunk_index += 1
                current_chunk_start = (
                    chunk_target_end
                )

                feature_buffer = []
                label_buffer = []
                path_buffer = []

    if feature_buffer:
        raise RuntimeError(
            "Internal chunking error: "
            "feature buffer remained after "
            "dataset iteration."
        )

    total_elapsed = (
        time.perf_counter()
        - split_start
    )

    print()
    print(
        f"{split} caching complete."
    )

    print(
        f"Samples: {total_samples}"
    )

    print(
        f"Chunks completed: "
        f"{completed_chunks}/{num_chunks}"
    )

    print(
        f"Elapsed: "
        f"{total_elapsed / 3600:.2f} hours"
    )

    print(
        f"Average throughput: "
        f"{total_samples / max(total_elapsed, 1e-8):.2f} img/s"
    )

    return total_samples


# ============================================================
# VERIFY CACHE
# ============================================================

def verify_cache(
    split,
    cache_dir,
    expected_total,
):
    print()
    print("=" * 70)
    print(f"VERIFYING {split.upper()} CACHE")
    print("=" * 70)

    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No cache chunks found for {split}."
        )

    total = 0
    expected_start = 0

    all_paths = set()

    for index, path in enumerate(files):
        payload = torch.load(
            path,
            map_location="cpu",
        )

        features = payload["features"]
        labels = payload["labels"]
        paths = payload["paths"]

        start_index = int(
            payload["start_index"]
        )

        end_index = int(
            payload["end_index"]
        )

        expected_count = (
            end_index
            - start_index
        )

        if start_index != expected_start:
            raise RuntimeError(
                f"Gap or overlap in {split} cache "
                f"at chunk {index}."
            )

        if features.dtype != torch.float16:
            raise RuntimeError(
                f"{path} is not float16."
            )

        if features.shape != (
            expected_count,
            EXPECTED_FEATURE_DIM,
        ):
            raise RuntimeError(
                f"Bad feature shape in {path}: "
                f"{tuple(features.shape)}"
            )

        if labels.shape[0] != expected_count:
            raise RuntimeError(
                f"Bad labels in {path}."
            )

        if len(paths) != expected_count:
            raise RuntimeError(
                f"Bad paths in {path}."
            )

        for item in paths:
            if item in all_paths:
                raise RuntimeError(
                    f"Duplicate path found: {item}"
                )

            all_paths.add(item)

        total += expected_count
        expected_start = end_index

    if total != expected_total:
        raise RuntimeError(
            f"{split} cache count mismatch: "
            f"{total} vs {expected_total}"
        )

    print(
        f"Chunks: {len(files)}"
    )

    print(
        f"Total cached samples: {total}"
    )

    print(
        f"Feature dimension: "
        f"{EXPECTED_FEATURE_DIM}"
    )

    print(
        "Feature dtype: float16"
    )

    print(
        f"Unique image paths: "
        f"{len(all_paths)}"
    )

    print(
        f"{split.upper()} CACHE PASS: True"
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
    print("VISDA-2017 FULL RESNET-50 FEATURE CACHE")
    print("=" * 70)

    print(
        f"device={device}"
    )

    print(
        f"batch_size={BATCH_SIZE}"
    )

    print(
        f"chunk_size={CHUNK_SIZE}"
    )

    print(
        "feature_dim=2048"
    )

    print(
        "storage_dtype=float16"
    )

    print(
        "preprocessing=Resize256 + CenterCrop224 + ImageNetNorm"
    )

    print()
    print(
        "Target labels will be stored ONLY for evaluation."
    )

    print(
        "They must NOT be used by adaptation training."
    )

    # --------------------------------------------------------
    # Dataset existence checks
    # --------------------------------------------------------

    if not SOURCE_ROOT.exists():
        raise RuntimeError(
            f"Missing source root: "
            f"{SOURCE_ROOT.resolve()}"
        )

    if not TARGET_ROOT.exists():
        raise RuntimeError(
            f"Missing target root: "
            f"{TARGET_ROOT.resolve()}"
        )

    # --------------------------------------------------------
    # Load frozen backbone
    # --------------------------------------------------------

    print()
    print(
        "Building frozen feature extractor..."
    )

    model = build_feature_extractor()
    model.to(device)

    # --------------------------------------------------------
    # Cache source
    # --------------------------------------------------------

    source_total = cache_split(
        split="source",
        root=SOURCE_ROOT,
        cache_dir=SOURCE_CACHE,
        model=model,
        device=device,
    )

    # --------------------------------------------------------
    # Cache target
    # --------------------------------------------------------

    target_total = cache_split(
        split="target",
        root=TARGET_ROOT,
        cache_dir=TARGET_CACHE,
        model=model,
        device=device,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_cache(
        split="source",
        cache_dir=SOURCE_CACHE,
        expected_total=source_total,
    )

    verify_cache(
        split="target",
        cache_dir=TARGET_CACHE,
        expected_total=target_total,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FULL FEATURE CACHE COMPLETE")
    print("=" * 70)

    print(
        f"Source samples: {source_total}"
    )

    print(
        f"Target samples: {target_total}"
    )

    print(
        f"Total samples: "
        f"{source_total + target_total}"
    )

    print(
        f"Source cache: "
        f"{SOURCE_CACHE.resolve()}"
    )

    print(
        f"Target cache: "
        f"{TARGET_CACHE.resolve()}"
    )

    print()
    print(
        "We are now ready for the fast CPU "
        "adaptation experiments."
    )


if __name__ == "__main__":
    main()