import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# VISDA-2017
# FROZEN RESNET-50 FEATURE SPACE
# SOURCE-ONLY LINEAR PROBE
# ============================================================

SEED = 42

CACHE_ROOT = Path(
    "checkpoints/visda_feature_cache"
)

SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

OUTPUT_DIR = Path(
    "checkpoints/visda_cached_source_only"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 512
EPOCHS = 30

LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 12
FEATURE_DIM = 2048

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD CACHE
# ============================================================

def load_cache(cache_dir):
    chunk_files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not chunk_files:
        raise RuntimeError(
            f"No cache chunks found in {cache_dir}"
        )

    all_features = []
    all_labels = []
    all_paths = []

    for path in chunk_files:
        payload = torch.load(
            path,
            map_location="cpu",
        )

        features = payload["features"]
        labels = payload["labels"]
        paths = payload["paths"]

        if features.ndim != 2:
            raise RuntimeError(
                f"Invalid feature shape in {path}"
            )

        if features.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                f"Expected feature dimension "
                f"{FEATURE_DIM}, got "
                f"{features.shape[1]} in {path}"
            )

        all_features.append(
            features.float()
        )

        all_labels.append(
            labels.long()
        )

        all_paths.extend(paths)

    features = torch.cat(
        all_features,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    if len(features) != len(labels):
        raise RuntimeError(
            "Feature/label count mismatch."
        )

    if len(features) != len(all_paths):
        raise RuntimeError(
            "Feature/path count mismatch."
        )

    return features, labels, all_paths


# ============================================================
# DATASET
# ============================================================

class FeatureDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
    ):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return (
            self.features[index],
            self.labels[index],
        )


# ============================================================
# MODEL
# ============================================================

class LinearProbe(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_classes,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def evaluate(
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

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        predictions = logits.argmax(
            dim=1
        )

        total_correct += int(
            (predictions == labels)
            .sum()
            .item()
        )

        total_examples += labels.size(0)

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                labels == class_id
            )

            if mask.any():
                total_per_class[class_id] += int(
                    mask.sum().item()
                )

                correct_per_class[class_id] += int(
                    (
                        predictions[mask]
                        == labels[mask]
                    )
                    .sum()
                    .item()
                )

    per_class = []

    for class_id in range(
        NUM_CLASSES
    ):
        total = int(
            total_per_class[class_id]
        )

        correct = int(
            correct_per_class[class_id]
        )

        accuracy = (
            100.0 * correct / total
            if total > 0
            else 0.0
        )

        per_class.append(
            accuracy
        )

    mean_class_accuracy = float(
        np.mean(per_class)
    )

    overall_accuracy = (
        100.0
        * total_correct
        / max(total_examples, 1)
    )

    return {
        "overall_accuracy":
            overall_accuracy,
        "mean_class_accuracy":
            mean_class_accuracy,
        "per_class_accuracy":
            per_class,
        "correct_per_class":
            [
                int(x)
                for x in correct_per_class
            ],
        "total_per_class":
            [
                int(x)
                for x in total_per_class
            ],
    }


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(features)

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

        total_correct += int(
            (logits.argmax(dim=1) == labels)
            .sum()
            .item()
        )

        total_examples += batch_size

    return {
        "loss":
            total_loss
            / max(total_examples, 1),
        "accuracy":
            100.0
            * total_correct
            / max(total_examples, 1),
    }


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
    print("VISDA-2017 CACHED FEATURE SOURCE-ONLY")
    print("=" * 70)

    print(f"device={device}")
    print(f"seed={SEED}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"epochs={EPOCHS}")
    print(f"feature_dim={FEATURE_DIM}")
    print(f"lr={LR}")
    print(f"momentum={MOMENTUM}")
    print(f"weight_decay={WEIGHT_DECAY}")

    # --------------------------------------------------------
    # Load full cached features
    # --------------------------------------------------------

    print()
    print("Loading source feature cache...")

    source_features, source_labels, source_paths = (
        load_cache(SOURCE_CACHE)
    )

    print(
        f"Source features: "
        f"{tuple(source_features.shape)}"
    )

    print(
        f"Source labels: "
        f"{tuple(source_labels.shape)}"
    )

    print()
    print("Loading target feature cache...")

    target_features, target_labels, target_paths = (
        load_cache(TARGET_CACHE)
    )

    print(
        f"Target features: "
        f"{tuple(target_features.shape)}"
    )

    print(
        f"Target labels: "
        f"{tuple(target_labels.shape)}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    source_dataset = FeatureDataset(
        source_features,
        source_labels,
    )

    target_dataset = FeatureDataset(
        target_features,
        target_labels,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=False,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print()
    print(
        f"Source batches/epoch: "
        f"{len(source_loader)}"
    )

    print(
        f"Target evaluation batches: "
        f"{len(target_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LinearProbe(
        FEATURE_DIM,
        NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    history = []

    # --------------------------------------------------------
    # Initial target evaluation
    # --------------------------------------------------------

    initial_target = evaluate(
        model,
        target_loader,
        device,
    )

    print()
    print("=" * 70)
    print("INITIAL TARGET")
    print("=" * 70)

    print(
        f"Overall: "
        f"{initial_target['overall_accuracy']:.2f}%"
    )

    print(
        f"Mean-class: "
        f"{initial_target['mean_class_accuracy']:.2f}%"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SOURCE-ONLY TRAINING")
    print("=" * 70)

    training_start = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            source_loader,
            optimizer,
            criterion,
            device,
        )

        target_metrics = evaluate(
            model,
            target_loader,
            device,
        )

        scheduler.step()

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "train_loss":
                train_metrics["loss"],
            "train_accuracy":
                train_metrics["accuracy"],
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
            "epoch_seconds":
                epoch_seconds,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Source Loss "
            f"{train_metrics['loss']:.4f} | "
            f"Source Acc "
            f"{train_metrics['accuracy']:.2f}% | "
            f"Target Overall "
            f"{target_metrics['overall_accuracy']:.2f}% | "
            f"Target Mean-Class "
            f"{target_metrics['mean_class_accuracy']:.2f}% | "
            f"{epoch_seconds:.1f}s"
        )

    total_training_seconds = (
        time.perf_counter()
        - training_start
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_metrics = evaluate(
        model,
        target_loader,
        device,
    )

    print()
    print("=" * 70)
    print("FINAL SOURCE-ONLY RESULT")
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
    print(
        f"Total training time: "
        f"{total_training_seconds:.2f}s"
    )

    # --------------------------------------------------------
    # Best target diagnostic
    # --------------------------------------------------------

    best_epoch_record = max(
        history,
        key=lambda x:
            x["target_mean_class_accuracy"],
    )

    print()
    print("=" * 70)
    print("BEST TARGET DIAGNOSTIC")
    print("=" * 70)

    print(
        f"Best observed target epoch: "
        f"{best_epoch_record['epoch']}"
    )

    print(
        f"Best observed mean-class: "
        f"{best_epoch_record['target_mean_class_accuracy']:.2f}%"
    )

    print(
        "NOTE: Target labels are used here ONLY "
        "for diagnostic reporting."
    )

    print(
        "They are NOT used for training or "
        "checkpoint selection."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    checkpoint_path = (
        OUTPUT_DIR
        / "linear_probe_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "linear_probe_seed42.json"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "seed":
                SEED,
            "feature_dim":
                FEATURE_DIM,
            "num_classes":
                NUM_CLASSES,
            "classes":
                CLASSES,
            "batch_size":
                BATCH_SIZE,
            "epochs":
                EPOCHS,
            "lr":
                LR,
            "momentum":
                MOMENTUM,
            "weight_decay":
                WEIGHT_DECAY,
            "final_metrics":
                final_metrics,
            "history":
                history,
        },
        checkpoint_path,
    )

    report = {
        "experiment":
            "visda_cached_feature_source_only",
        "seed":
            SEED,
        "device":
            str(device),
        "backbone":
            "ImageNet-pretrained ResNet-50",
        "backbone_frozen":
            True,
        "feature_dim":
            FEATURE_DIM,
        "source_samples":
            len(source_dataset),
        "target_samples":
            len(target_dataset),
        "batch_size":
            BATCH_SIZE,
        "epochs":
            EPOCHS,
        "lr":
            LR,
        "momentum":
            MOMENTUM,
        "weight_decay":
            WEIGHT_DECAY,
        "metric":
            "mean_per_class_accuracy",
        "final_metrics":
            final_metrics,
        "best_observed_target_epoch":
            best_epoch_record["epoch"],
        "best_observed_target_mean_class_accuracy":
            best_epoch_record[
                "target_mean_class_accuracy"
            ],
        "history":
            history,
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