import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# ============================================================
# DEFAULT CONFIG
# ============================================================

SEED = 42

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

NUM_CLASSES = len(CLASSES)

DEFAULT_SOURCE_SAMPLES = 10000
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 16

LR = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

RESIZE_SIZE = 256
CROP_SIZE = 224

OUTPUT_DIR = Path("checkpoints/visda_source_only_pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VisDA-2017 ResNet-101 source-only CPU pilot"
    )

    parser.add_argument(
        "--source-samples",
        type=int,
        default=DEFAULT_SOURCE_SAMPLES,
        help="Number of source images for the pilot.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of pilot epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    return parser.parse_args()


# ============================================================
# DATA
# ============================================================

def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, target_transform


def build_datasets(source_samples: int, seed: int):
    train_transform, target_transform = build_transforms()

    full_source = datasets.ImageFolder(
        root=str(SOURCE_ROOT),
        transform=train_transform,
    )

    target_dataset = datasets.ImageFolder(
        root=str(TARGET_ROOT),
        transform=target_transform,
    )

    expected_mapping = {
        name: idx
        for idx, name in enumerate(CLASSES)
    }

    if full_source.class_to_idx != expected_mapping:
        raise RuntimeError(
            "Source class mapping mismatch.\n"
            f"Expected: {expected_mapping}\n"
            f"Found:    {full_source.class_to_idx}"
        )

    if target_dataset.class_to_idx != expected_mapping:
        raise RuntimeError(
            "Target class mapping mismatch.\n"
            f"Expected: {expected_mapping}\n"
            f"Found:    {target_dataset.class_to_idx}"
        )

    if source_samples > len(full_source):
        raise ValueError(
            f"Requested {source_samples} source samples, "
            f"but only {len(full_source)} exist."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        len(full_source),
        generator=generator,
    )[:source_samples].tolist()

    source_dataset = Subset(
        full_source,
        indices,
    )

    return source_dataset, target_dataset


def build_loaders(
    source_dataset,
    target_dataset,
    batch_size: int,
):
    train_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, target_loader


# ============================================================
# MODEL
# ============================================================

def build_model():
    print("Loading ImageNet-pretrained ResNet-101...")

    weights = models.ResNet101_Weights.DEFAULT

    model = models.resnet101(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model


# ============================================================
# TARGET EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_target(
    model,
    loader,
    device,
):
    model.eval()

    correct_per_class = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    total_per_class = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += labels.size(0)

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                class_total = int(
                    mask.sum().item()
                )

                class_correct = int(
                    (
                        predictions[mask]
                        == labels[mask]
                    ).sum().item()
                )

                total_per_class[class_id] += class_total
                correct_per_class[class_id] += class_correct

    per_class_accuracy = []

    for class_id in range(NUM_CLASSES):
        total = int(
            total_per_class[class_id]
        )

        correct = int(
            correct_per_class[class_id]
        )

        if total == 0:
            accuracy = 0.0
        else:
            accuracy = (
                100.0
                * correct
                / total
            )

        per_class_accuracy.append(
            accuracy
        )

    mean_class_accuracy = float(
        np.mean(
            per_class_accuracy
        )
    )

    overall_accuracy = (
        100.0
        * total_correct
        / max(total_examples, 1)
    )

    return {
        "mean_class_accuracy":
            mean_class_accuracy,
        "overall_accuracy":
            overall_accuracy,
        "per_class_accuracy":
            per_class_accuracy,
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    num_batches = len(loader)

    epoch_start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

        # Print progress every 100 batches.
        if (
            batch_idx % 100 == 0
            or batch_idx == num_batches
        ):
            elapsed = (
                time.perf_counter()
                - epoch_start
            )

            batches_per_sec = (
                batch_idx
                / max(elapsed, 1e-8)
            )

            remaining = (
                num_batches
                - batch_idx
            )

            eta_seconds = (
                remaining
                / max(
                    batches_per_sec,
                    1e-8,
                )
            )

            print(
                f"  Epoch "
                f"{epoch:02d}/{total_epochs} "
                f"| Batch "
                f"{batch_idx:04d}/{num_batches:04d} "
                f"| Loss {loss.item():.4f} "
                f"| {batches_per_sec:.2f} batch/s "
                f"| ETA {eta_seconds:.1f}s"
            )

    avg_loss = (
        total_loss
        / max(total_examples, 1)
    )

    accuracy = (
        100.0
        * total_correct
        / max(total_examples, 1)
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "seconds": epoch_time,
        "batches": num_batches,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("VISDA-2017 RESNET-101 SOURCE-ONLY CPU PILOT")
    print("=" * 70)

    print(f"device={device}")
    print(f"seed={args.seed}")
    print(f"source_samples={args.source_samples}")
    print(f"batch_size={args.batch_size}")
    print(f"epochs={args.epochs}")
    print(f"lr={LR}")
    print(f"weight_decay={WEIGHT_DECAY}")
    print(f"momentum={MOMENTUM}")

    print()
    print("Building datasets...")

    source_dataset, target_dataset = (
        build_datasets(
            source_samples=args.source_samples,
            seed=args.seed,
        )
    )

    print(
        f"Pilot source samples: "
        f"{len(source_dataset)}"
    )

    print(
        f"Target evaluation samples: "
        f"{len(target_dataset)}"
    )

    train_loader, target_loader = (
        build_loaders(
            source_dataset,
            target_dataset,
            batch_size=args.batch_size,
        )
    )

    print(
        f"Training batches/epoch: "
        f"{len(train_loader)}"
    )

    print()
    print("Building model...")

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    history = []

    print()
    print("=" * 70)
    print("PILOT TRAINING")
    print("=" * 70)

    total_start = time.perf_counter()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        result = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        scheduler.step()

        print()
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"finished in "
            f"{result['seconds']:.1f}s"
        )

        print(
            f"Train Loss: "
            f"{result['loss']:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{result['accuracy']:.2f}%"
        )

        target_metrics = evaluate_target(
            model,
            target_loader,
            device,
        )

        print(
            f"Target Overall: "
            f"{target_metrics['overall_accuracy']:.2f}%"
        )

        print(
            f"Target Mean-Class: "
            f"{target_metrics['mean_class_accuracy']:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    result["loss"],
                "train_accuracy":
                    result["accuracy"],
                "epoch_seconds":
                    result["seconds"],
                "batches":
                    result["batches"],
                "target_overall_accuracy":
                    target_metrics[
                        "overall_accuracy"
                    ],
                "target_mean_class_accuracy":
                    target_metrics[
                        "mean_class_accuracy"
                    ],
                "target_per_class_accuracy":
                    target_metrics[
                        "per_class_accuracy"
                    ],
                "lr":
                    optimizer.param_groups[0]["lr"],
            }
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    final_metrics = evaluate_target(
        model,
        target_loader,
        device,
    )

    print()
    print("=" * 70)
    print("PILOT FINAL RESULT")
    print("=" * 70)

    print(
        f"Target overall accuracy: "
        f"{final_metrics['overall_accuracy']:.2f}%"
    )

    print(
        f"Target mean-class accuracy: "
        f"{final_metrics['mean_class_accuracy']:.2f}%"
    )

    print()
    print("Per-class accuracy:")

    for class_name, accuracy in zip(
        CLASSES,
        final_metrics[
            "per_class_accuracy"
        ],
    ):
        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}%"
        )

    print()
    print("=" * 70)
    print("THROUGHPUT")
    print("=" * 70)

    avg_epoch_seconds = (
        np.mean(
            [
                item["epoch_seconds"]
                for item in history
            ]
        )
        if history
        else 0.0
    )

    batches_per_epoch = len(
        train_loader
    )

    batches_per_second = (
        batches_per_epoch
        / max(avg_epoch_seconds, 1e-8)
    )

    print(
        f"Average epoch time: "
        f"{avg_epoch_seconds:.2f}s"
    )

    print(
        f"Batches per epoch: "
        f"{batches_per_epoch}"
    )

    print(
        f"Average batches/sec: "
        f"{batches_per_second:.3f}"
    )

    print(
        f"Total pilot time: "
        f"{total_seconds:.2f}s"
    )

    print()
    print(
        "Estimated full-dataset epoch time:"
    )

    full_source_size = 152397

    full_batches = (
        full_source_size
        // args.batch_size
    )

    estimated_full_epoch_seconds = (
        full_batches
        / max(
            batches_per_second,
            1e-8,
        )
    )

    print(
        f"{estimated_full_epoch_seconds / 60.0:.2f} minutes"
    )

    print()
    print(
        "Estimated 20-epoch full run time:"
    )

    estimated_full_run_seconds = (
        estimated_full_epoch_seconds
        * 20
    )

    print(
        f"{estimated_full_run_seconds / 3600.0:.2f} hours"
    )

    # Save pilot history.
    checkpoint_path = (
        OUTPUT_DIR
        / "resnet101_pilot_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "resnet101_pilot_seed42.json"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "seed":
                args.seed,
            "source_samples":
                args.source_samples,
            "epochs":
                args.epochs,
            "batch_size":
                args.batch_size,
            "lr":
                LR,
            "weight_decay":
                WEIGHT_DECAY,
            "momentum":
                MOMENTUM,
            "classes":
                CLASSES,
            "final_metrics":
                final_metrics,
            "history":
                history,
        },
        checkpoint_path,
    )

    report = {
        "experiment":
            "visda_resnet101_source_only_pilot",
        "seed":
            args.seed,
        "device":
            str(device),
        "source_samples":
            args.source_samples,
        "target_samples":
            len(target_dataset),
        "epochs":
            args.epochs,
        "batch_size":
            args.batch_size,
        "lr":
            LR,
        "weight_decay":
            WEIGHT_DECAY,
        "momentum":
            MOMENTUM,
        "backbone":
            "resnet101",
        "pretrained":
            True,
        "source_total":
            152397,
        "target_total":
            55388,
        "final_metrics":
            final_metrics,
        "history":
            history,
        "throughput":
            {
                "average_epoch_seconds":
                    avg_epoch_seconds,
                "batches_per_epoch":
                    batches_per_epoch,
                "batches_per_second":
                    batches_per_second,
                "estimated_full_epoch_seconds":
                    estimated_full_epoch_seconds,
                "estimated_20_epoch_seconds":
                    estimated_full_run_seconds,
            },
    }

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print()
    print("=" * 70)
    print("FILES")
    print("=" * 70)

    print(
        f"checkpoint={checkpoint_path}"
    )

    print(
        f"history={history_path}"
    )


if __name__ == "__main__":
    main()