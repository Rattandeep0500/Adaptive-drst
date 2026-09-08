import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path("data/visda")
SOURCE_ROOT = ROOT / "train"
TARGET_ROOT = ROOT / "validation"

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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

SEED = 42
SAMPLES_PER_CLASS = 50
BATCH_SIZE = 16


def image_files(class_dir):
    return [
        path
        for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def count_images(root):
    counts = {}

    for class_name in CLASSES:
        class_dir = root / class_name

        if not class_dir.exists():
            counts[class_name] = 0
        else:
            counts[class_name] = len(
                image_files(class_dir)
            )

    return counts


def verify_images(root):
    random.seed(SEED)

    checked = 0
    failures = []
    modes = {}
    sizes = {}

    for class_name in CLASSES:
        class_dir = root / class_name

        files = image_files(class_dir)

        if len(files) == 0:
            failures.append(
                f"No images found for class: {class_name}"
            )
            continue

        sample_size = min(
            SAMPLES_PER_CLASS,
            len(files),
        )

        samples = random.sample(
            files,
            sample_size,
        )

        for path in samples:
            try:
                with Image.open(path) as image:
                    image.verify()

                with Image.open(path) as image:
                    image.load()

                    mode = image.mode
                    size = image.size

                modes[mode] = modes.get(mode, 0) + 1
                sizes[str(size)] = (
                    sizes.get(str(size), 0) + 1
                )

                checked += 1

            except Exception as exc:
                failures.append(
                    f"{path}: {exc}"
                )

    return {
        "checked": checked,
        "failures": failures,
        "modes": modes,
        "sizes": sizes,
    }


def print_counts(title, counts):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    total = 0

    for class_name in CLASSES:
        count = counts[class_name]
        total += count

        print(
            f"{class_name:12s}: {count:7d}"
        )

    print("-" * 70)
    print(
        f"{'TOTAL':12s}: {total:7d}"
    )

    return total


def build_loader(root):
    transform = transforms.Compose(
        [
            transforms.Resize(
                256
            ),
            transforms.CenterCrop(
                224
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    dataset = datasets.ImageFolder(
        root=str(root),
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(
        iter(loader)
    )

    return dataset, loader, images, labels


def main():
    print("=" * 70)
    print("VISDA-2017 PHASE 0 SANITY CHECK")
    print("=" * 70)

    print(
        f"Project root: {ROOT.resolve()}"
    )

    if not SOURCE_ROOT.exists():
        raise RuntimeError(
            f"Missing source directory: {SOURCE_ROOT.resolve()}"
        )

    if not TARGET_ROOT.exists():
        raise RuntimeError(
            f"Missing target directory: {TARGET_ROOT.resolve()}"
        )

    source_counts = count_images(
        SOURCE_ROOT
    )

    target_counts = count_images(
        TARGET_ROOT
    )

    source_total = print_counts(
        "SOURCE: SYNTHETIC TRAIN",
        source_counts,
    )

    target_total = print_counts(
        "TARGET: REAL VALIDATION",
        target_counts,
    )

    print()
    print("=" * 70)
    print("CLASS INTEGRITY")
    print("=" * 70)

    source_classes = sorted(
        [
            path.name
            for path in SOURCE_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    target_classes = sorted(
        [
            path.name
            for path in TARGET_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    expected_classes = sorted(
        CLASSES
    )

    source_class_ok = (
        source_classes
        == expected_classes
    )

    target_class_ok = (
        target_classes
        == expected_classes
    )

    print(
        f"Source classes correct: "
        f"{source_class_ok}"
    )

    print(
        f"Target classes correct: "
        f"{target_class_ok}"
    )

    print()
    print("=" * 70)
    print("IMAGE READABILITY")
    print("=" * 70)

    print("Checking source samples...")

    source_readability = verify_images(
        SOURCE_ROOT
    )

    print(
        f"Source images checked: "
        f"{source_readability['checked']}"
    )

    print(
        f"Source failures: "
        f"{len(source_readability['failures'])}"
    )

    print(
        f"Source modes: "
        f"{source_readability['modes']}"
    )

    print(
        f"Source size variants: "
        f"{len(source_readability['sizes'])}"
    )

    print("Checking target samples...")

    target_readability = verify_images(
        TARGET_ROOT
    )

    print(
        f"Target images checked: "
        f"{target_readability['checked']}"
    )

    print(
        f"Target failures: "
        f"{len(target_readability['failures'])}"
    )

    print(
        f"Target modes: "
        f"{target_readability['modes']}"
    )

    print(
        f"Target size variants: "
        f"{len(target_readability['sizes'])}"
    )

    print()
    print("=" * 70)
    print("PYTORCH LOADER TEST")
    print("=" * 70)

    source_dataset, source_loader, source_images, source_labels = (
        build_loader(
            SOURCE_ROOT
        )
    )

    target_dataset, target_loader, target_images, target_labels = (
        build_loader(
            TARGET_ROOT
        )
    )

    print(
        f"Source dataset size: "
        f"{len(source_dataset)}"
    )

    print(
        f"Target dataset size: "
        f"{len(target_dataset)}"
    )

    print(
        f"Source ImageFolder mapping: "
        f"{source_dataset.class_to_idx}"
    )

    print(
        f"Target ImageFolder mapping: "
        f"{target_dataset.class_to_idx}"
    )

    print(
        f"Source batch shape: "
        f"{tuple(source_images.shape)}"
    )

    print(
        f"Target batch shape: "
        f"{tuple(target_images.shape)}"
    )

    print(
        f"Source labels shape: "
        f"{tuple(source_labels.shape)}"
    )

    print(
        f"Target labels shape: "
        f"{tuple(target_labels.shape)}"
    )

    print(
        f"Source dtype: "
        f"{source_images.dtype}"
    )

    print(
        f"Target dtype: "
        f"{target_images.dtype}"
    )

    source_mapping_ok = (
        source_dataset.class_to_idx
        == {
            class_name: index
            for index, class_name in enumerate(
                CLASSES
            )
        }
    )

    target_mapping_ok = (
        target_dataset.class_to_idx
        == {
            class_name: index
            for index, class_name in enumerate(
                CLASSES
            )
        }
    )

    print()
    print(
        f"Source mapping matches expected: "
        f"{source_mapping_ok}"
    )

    print(
        f"Target mapping matches expected: "
        f"{target_mapping_ok}"
    )

    print()
    print("=" * 70)
    print("IMAGE LIST CHECK")
    print("=" * 70)

    image_list_results = {}

    for split in [
        "train",
        "validation",
    ]:
        image_list_path = (
            ROOT
            / split
            / "image_list.txt"
        )

        exists = image_list_path.exists()

        result = {
            "exists": exists,
            "entries": 0,
        }

        if exists:
            with open(
                image_list_path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as handle:
                entries = [
                    line.strip()
                    for line in handle
                    if line.strip()
                ]

            result["entries"] = len(
                entries
            )

        image_list_results[split] = result

        print(
            f"{split:12s}: "
            f"exists={result['exists']} "
            f"entries={result['entries']}"
        )

    print()
    print("=" * 70)
    print("PHASE 0 RESULT")
    print("=" * 70)

    failures = (
        source_readability["failures"]
        + target_readability["failures"]
    )

    phase0_pass = all(
        [
            source_total > 0,
            target_total > 0,
            source_class_ok,
            target_class_ok,
            source_readability["checked"]
            == SAMPLES_PER_CLASS
            * len(CLASSES),
            target_readability["checked"]
            == SAMPLES_PER_CLASS
            * len(CLASSES),
            len(failures) == 0,
            source_mapping_ok,
            target_mapping_ok,
            tuple(source_images.shape[1:])
            == (3, 224, 224),
            tuple(target_images.shape[1:])
            == (3, 224, 224),
            image_list_results["train"]["exists"],
            image_list_results["validation"]["exists"],
        ]
    )

    print(
        f"Source total: {source_total}"
    )

    print(
        f"Target total: {target_total}"
    )

    print(
        f"Images decoded without failure: "
        f"{len(failures) == 0}"
    )

    print(
        f"ImageNet preprocessing batch: "
        f"{tuple(source_images.shape[1:])}"
    )

    print(
        f"PHASE 0 PASS: {phase0_pass}"
    )

    report = {
        "source_total": source_total,
        "target_total": target_total,
        "source_counts": source_counts,
        "target_counts": target_counts,
        "source_classes": source_classes,
        "target_classes": target_classes,
        "source_readability": source_readability,
        "target_readability": target_readability,
        "source_mapping": source_dataset.class_to_idx,
        "target_mapping": target_dataset.class_to_idx,
        "source_batch_shape": list(
            source_images.shape
        ),
        "target_batch_shape": list(
            target_images.shape
        ),
        "image_lists": image_list_results,
        "phase0_pass": phase0_pass,
    }

    report_path = (
        Path("experiments")
        / "visda_phase0_data_card.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        f"Report written to: "
        f"{report_path}"
    )

    if not phase0_pass:
        raise RuntimeError(
            "VisDA Phase 0 FAILED. "
            "Fix the reported issue before training."
        )


if __name__ == "__main__":
    main()