import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
TARGET_CACHE = CACHE_ROOT / "target"

OUTPUT_DIR = Path(
    "checkpoints/visda_target_capacity_probe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 12
INPUT_DIM = 2048

TRAIN_FRACTION = 0.80
BATCH_SIZE = 1024
EPOCHS = 30

LINEAR_LR = 0.05
LINEAR_WEIGHT_DECAY = 1e-4

MLP_LR = 0.01
MLP_WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 6

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_target_cache():
    files = sorted(
        TARGET_CACHE.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No target cache chunks found in {TARGET_CACHE}"
        )

    features = []
    labels = []

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu",
        )

        features.append(
            payload["features"].float()
        )

        labels.append(
            payload["labels"].long()
        )

    features = torch.cat(
        features,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2D features, got {features.shape}"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected {INPUT_DIM} features, got {features.shape[1]}"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Feature and label counts do not match."
        )

    return features, labels


def stratified_split(
    features,
    labels,
    train_fraction,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_indices = []
    validation_indices = []

    for class_id in range(NUM_CLASSES):
        indices = torch.nonzero(
            labels == class_id,
            as_tuple=False,
        ).flatten()

        permutation = torch.randperm(
            len(indices),
            generator=generator,
        )

        indices = indices[
            permutation
        ]

        train_count = int(
            len(indices)
            * train_fraction
        )

        train_count = min(
            max(train_count, 1),
            len(indices) - 1,
        )

        train_indices.extend(
            indices[:train_count].tolist()
        )

        validation_indices.extend(
            indices[train_count:].tolist()
        )

    train_indices = torch.tensor(
        train_indices,
        dtype=torch.long,
    )

    validation_indices = torch.tensor(
        validation_indices,
        dtype=torch.long,
    )

    train_shuffle = torch.randperm(
        len(train_indices),
        generator=generator,
    )

    validation_shuffle = torch.randperm(
        len(validation_indices),
        generator=generator,
    )

    train_indices = train_indices[
        train_shuffle
    ]

    validation_indices = validation_indices[
        validation_shuffle
    ]

    return (
        features[train_indices],
        labels[train_indices],
        features[validation_indices],
        labels[validation_indices],
    )


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(
            INPUT_DIM,
            NUM_CLASSES,
        )

    def forward(self, x):
        return self.classifier(x)


class MLP512(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                512,
            ),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(
                512,
                NUM_CLASSES,
            ),
        )

    def forward(self, x):
        return self.network(x)


class MLP512256(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                512,
            ),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(
                512,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(
                256,
                NUM_CLASSES,
            ),
        )

    def forward(self, x):
        return self.network(x)


def make_loaders(
    train_features,
    train_labels,
    validation_features,
    validation_labels,
):
    train_loader = DataLoader(
        TensorDataset(
            train_features,
            train_labels,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    validation_loader = DataLoader(
        TensorDataset(
            validation_features,
            validation_labels,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return (
        train_loader,
        validation_loader,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
):
    model.eval()

    total = 0
    correct = 0

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for x, y in loader:
        logits = model(x)

        predictions = logits.argmax(
            dim=1
        )

        total += y.size(0)

        correct += int(
            (
                predictions == y
            ).sum().item()
        )

        for class_id in range(NUM_CLASSES):
            mask = y == class_id

            if mask.any():
                class_total[
                    class_id
                ] += int(
                    mask.sum().item()
                )

                class_correct[
                    class_id
                ] += int(
                    (
                        predictions[mask]
                        == y[mask]
                    )
                    .sum()
                    .item()
                )

    per_class_accuracy = (
        100.0
        * class_correct.float()
        / class_total.clamp_min(1)
    )

    overall = (
        100.0
        * correct
        / max(total, 1)
    )

    mean_class = float(
        per_class_accuracy.mean().item()
    )

    return {
        "overall_accuracy":
            overall,
        "mean_class_accuracy":
            mean_class,
        "per_class_accuracy":
            per_class_accuracy.tolist(),
        "total":
            total,
    }


def train_model(
    model,
    train_loader,
    validation_loader,
    learning_rate,
    weight_decay,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )

    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_epoch = 0
    best_metric = -float("inf")
    patience_counter = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        loss_sum = 0.0
        sample_count = 0

        for x, y in train_loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()

            batch_size = y.size(0)

            loss_sum += (
                loss.item()
                * batch_size
            )

            sample_count += batch_size

        scheduler.step()

        validation = evaluate(
            model,
            validation_loader,
        )

        mean_loss = (
            loss_sum
            / max(
                sample_count,
                1,
            )
        )

        record = {
            "epoch":
                epoch,
            "train_loss":
                mean_loss,
            "validation_overall":
                validation[
                    "overall_accuracy"
                ],
            "validation_mean_class":
                validation[
                    "mean_class_accuracy"
                ],
        }

        history.append(
            record
        )

        current_metric = validation[
            "mean_class_accuracy"
        ]

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {mean_loss:.5f} | "
            f"Overall "
            f"{validation['overall_accuracy']:.2f}% | "
            f"Mean-class "
            f"{validation['mean_class_accuracy']:.2f}%"
        )

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "No best model state was recorded."
        )

    model.load_state_dict(
        best_state
    )

    final_metrics = evaluate(
        model,
        validation_loader,
    )

    return (
        final_metrics,
        best_epoch,
        history,
    )


def class_balance(labels):
    counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES,
    )

    return [
        {
            "class":
                CLASSES[i],
            "count":
                int(counts[i].item()),
            "fraction":
                float(
                    counts[i].item()
                    / len(labels)
                ),
        }
        for i in range(NUM_CLASSES)
    ]


def print_final_result(
    name,
    metrics,
):
    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    print(
        f"Overall accuracy: "
        f"{metrics['overall_accuracy']:.2f}%"
    )

    print(
        f"Mean-class accuracy: "
        f"{metrics['mean_class_accuracy']:.2f}%"
    )

    print()
    print("Per-class accuracy:")

    for class_name, accuracy in zip(
        CLASSES,
        metrics["per_class_accuracy"],
    ):
        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}%"
        )


def main():
    set_seed(SEED)

    print("=" * 90)
    print(
        "VISDA-2017 FROZEN-FEATURE TARGET CAPACITY PROBE"
    )
    print("=" * 90)

    print(
        "Target labels are used only for this offline capacity diagnostic."
    )

    print()
    print(
        "Loading target feature cache..."
    )

    features, labels = load_target_cache()

    print(
        f"Total target samples: "
        f"{len(features)}"
    )

    print(
        f"Feature dimension: "
        f"{features.shape[1]}"
    )

    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = stratified_split(
        features,
        labels,
        TRAIN_FRACTION,
        SEED,
    )

    print()
    print(
        f"Train samples: "
        f"{len(train_features)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_features)}"
    )

    print()
    print(
        "=" * 90
    )

    print(
        "TRAIN/VALIDATION CLASS BALANCE"
    )

    print(
        "=" * 90
    )

    print("Train:")

    for row in class_balance(
        train_labels
    ):
        print(
            f"{row['class']:12s}: "
            f"{row['count']:6d} "
            f"({100.0 * row['fraction']:6.2f}%)"
        )

    print()
    print("Validation:")

    for row in class_balance(
        validation_labels
    ):
        print(
            f"{row['class']:12s}: "
            f"{row['count']:6d} "
            f"({100.0 * row['fraction']:6.2f}%)"
        )

    (
        train_loader,
        validation_loader,
    ) = make_loaders(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )

    models = [
        (
            "LINEAR PROBE",
            LinearProbe(),
            LINEAR_LR,
            LINEAR_WEIGHT_DECAY,
        ),
        (
            "MLP 2048->512->12",
            MLP512(),
            MLP_LR,
            MLP_WEIGHT_DECAY,
        ),
        (
            "MLP 2048->512->256->12",
            MLP512256(),
            MLP_LR,
            MLP_WEIGHT_DECAY,
        ),
    ]

    results = {}

    start_time = time.perf_counter()

    for (
        name,
        model,
        learning_rate,
        weight_decay,
    ) in models:
        print()
        print("=" * 90)
        print(name)
        print("=" * 90)

        model_start = time.perf_counter()

        (
            metrics,
            best_epoch,
            history,
        ) = train_model(
            model,
            train_loader,
            validation_loader,
            learning_rate,
            weight_decay,
        )

        model_seconds = (
            time.perf_counter()
            - model_start
        )

        print_final_result(
            name,
            metrics,
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Training time: "
            f"{model_seconds:.2f}s"
        )

        results[name] = {
            "metrics":
                metrics,
            "best_epoch":
                best_epoch,
            "training_seconds":
                model_seconds,
            "history":
                history,
        }

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    ranking = sorted(
        [
            (
                name,
                value["metrics"][
                    "mean_class_accuracy"
                ],
                value["metrics"][
                    "overall_accuracy"
                ],
            )
            for name, value
            in results.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("=" * 90)
    print(
        "CAPACITY RANKING"
    )
    print("=" * 90)

    for rank, (
        name,
        mean_class,
        overall,
    ) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{name:30s} | "
            f"Mean-class "
            f"{mean_class:.2f}% | "
            f"Overall "
            f"{overall:.2f}%"
        )

    best_name = ranking[0][0]

    best_metrics = results[
        best_name
    ]["metrics"]

    print()
    print("=" * 90)
    print(
        "BEST CAPACITY ESTIMATE"
    )
    print("=" * 90)

    print(
        f"Model: {best_name}"
    )

    print(
        f"Mean-class accuracy: "
        f"{best_metrics['mean_class_accuracy']:.2f}%"
    )

    print(
        f"Overall accuracy: "
        f"{best_metrics['overall_accuracy']:.2f}%"
    )

    print()
    print(
        "Hard classes:"
    )

    hard_classes = [
        "truck",
        "skateboard",
        "knife",
        "person",
    ]

    accuracy_map = {
        class_name: accuracy
        for class_name, accuracy
        in zip(
            CLASSES,
            best_metrics[
                "per_class_accuracy"
            ],
        )
    }

    for class_name in hard_classes:
        print(
            f"{class_name:12s}: "
            f"{accuracy_map[class_name]:.2f}%"
        )

    report = {
        "experiment":
            "visda_target_frozen_feature_capacity_probe",
        "seed":
            SEED,
        "target_labels_used_only_for_offline_diagnostic":
            True,
        "total_target_samples":
            len(features),
        "feature_dimension":
            int(features.shape[1]),
        "train_fraction":
            TRAIN_FRACTION,
        "train_samples":
            len(train_features),
        "validation_samples":
            len(validation_features),
        "train_class_balance":
            class_balance(
                train_labels
            ),
        "validation_class_balance":
            class_balance(
                validation_labels
            ),
        "models":
            results,
        "ranking":
            [
                {
                    "rank":
                        index + 1,
                    "model":
                        name,
                    "mean_class_accuracy":
                        mean_class,
                    "overall_accuracy":
                        overall,
                }
                for index, (
                    name,
                    mean_class,
                    overall,
                ) in enumerate(
                    ranking
                )
            ],
        "best_model":
            best_name,
        "best_mean_class_accuracy":
            best_metrics[
                "mean_class_accuracy"
            ],
        "best_overall_accuracy":
            best_metrics[
                "overall_accuracy"
            ],
        "total_seconds":
            total_seconds,
    }

    report_path = (
        OUTPUT_DIR
        / "target_capacity_seed42.json"
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

    print()
    print("=" * 90)
    print(
        "CAPACITY PROBE COMPLETE"
    )
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )


if __name__ == "__main__":
    main()