import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 42

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

OUTPUT_DIR = Path("checkpoints/visda_cached_mcd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EPOCHS = 20
STEP_C_UPDATES = 1

LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DISCREPANCY_WEIGHT = 1.0

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


def load_cache(cache_dir):
    files = sorted(cache_dir.glob("chunk_*.pt"))

    if not files:
        raise RuntimeError(
            f"No cache chunks found in {cache_dir}"
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

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    if features.ndim != 2:
        raise RuntimeError(
            f"Invalid feature shape: {features.shape}"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected feature dimension {INPUT_DIM}, "
            f"got {features.shape[1]}"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Feature and label counts do not match."
        )

    return features, labels


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MCDModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.adapter = Adapter()

        self.classifier1 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

        self.classifier2 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

    def features(self, x):
        return self.adapter(x)

    def forward(self, x):
        z = self.adapter(x)

        return (
            self.classifier1(z),
            self.classifier2(z),
        )


def initialize_deterministic_heads(model):
    with torch.no_grad():
        model.classifier2.weight.copy_(
            model.classifier1.weight
        )

        model.classifier2.bias.copy_(
            model.classifier1.bias
        )

        pattern = torch.arange(
            model.classifier2.weight.numel(),
            dtype=model.classifier2.weight.dtype,
        ).reshape_as(
            model.classifier2.weight
        )

        signed = torch.where(
            pattern.remainder(2) == 0,
            torch.ones_like(pattern),
            -torch.ones_like(pattern),
        )

        model.classifier2.weight.add_(
            0.005 * signed
        )


def discrepancy(logits1, logits2):
    p1 = F.softmax(
        logits1,
        dim=1,
    )

    p2 = F.softmax(
        logits2,
        dim=1,
    )

    return torch.mean(
        torch.abs(p1 - p2)
    )


@torch.no_grad()
def evaluate(
    model,
    features,
    labels,
    batch_size,
):
    model.eval()

    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

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
    disagreement_sum = 0.0

    for x, y in loader:
        z = model.features(x)

        logits1 = model.classifier1(z)
        logits2 = model.classifier2(z)

        probabilities1 = F.softmax(
            logits1,
            dim=1,
        )

        probabilities2 = F.softmax(
            logits2,
            dim=1,
        )

        probabilities = (
            probabilities1
            + probabilities2
        ) / 2.0

        predictions = probabilities.argmax(
            dim=1
        )

        disagreement_sum += (
            torch.mean(
                torch.abs(
                    probabilities1
                    - probabilities2
                )
            ).item()
            * y.size(0)
        )

        total_correct += int(
            (predictions == y)
            .sum()
            .item()
        )

        total_examples += y.size(0)

        for class_id in range(NUM_CLASSES):
            mask = y == class_id

            if mask.any():
                total_per_class[class_id] += int(
                    mask.sum().item()
                )

                correct_per_class[class_id] += int(
                    (
                        predictions[mask]
                        == y[mask]
                    )
                    .sum()
                    .item()
                )

    per_class = []

    for class_id in range(NUM_CLASSES):
        total = int(
            total_per_class[class_id]
        )

        correct = int(
            correct_per_class[class_id]
        )

        value = (
            100.0 * correct / total
            if total > 0
            else 0.0
        )

        per_class.append(value)

    mean_class_accuracy = float(
        np.mean(per_class)
    )

    overall_accuracy = (
        100.0
        * total_correct
        / max(total_examples, 1)
    )

    mean_disagreement = (
        disagreement_sum
        / max(total_examples, 1)
    )

    return {
        "overall_accuracy": overall_accuracy,
        "mean_class_accuracy": mean_class_accuracy,
        "per_class_accuracy": per_class,
        "mean_disagreement": mean_disagreement,
    }


def train_epoch(
    model,
    source_loader,
    target_loader,
    optimizer_a,
    optimizer_c,
    device,
):
    model.train()

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    steps = min(
        len(source_loader),
        len(target_loader),
    )

    source_loss_total = 0.0
    source_correct_total = 0
    source_examples_total = 0
    discrepancy_total = 0.0

    for _ in range(steps):
        try:
            source_x, source_y = next(
                source_iter
            )
        except StopIteration:
            source_iter = iter(
                source_loader
            )
            source_x, source_y = next(
                source_iter
            )

        try:
            target_x = next(
                target_iter
            )[0]
        except StopIteration:
            target_iter = iter(
                target_loader
            )
            target_x = next(
                target_iter
            )[0]

        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        optimizer_a.zero_grad(
            set_to_none=True
        )

        source_logits1, source_logits2 = (
            model(source_x)
        )

        loss_a = (
            F.cross_entropy(
                source_logits1,
                source_y,
            )
            + F.cross_entropy(
                source_logits2,
                source_y,
            )
        )

        loss_a.backward()
        optimizer_a.step()

        for _ in range(STEP_C_UPDATES):
            optimizer_c.zero_grad(
                set_to_none=True
            )

            source_features = (
                model.features(
                    source_x
                ).detach()
            )

            target_features = (
                model.features(
                    target_x
                ).detach()
            )

            source_logits1 = (
                model.classifier1(
                    source_features
                )
            )

            source_logits2 = (
                model.classifier2(
                    source_features
                )
            )

            target_logits1 = (
                model.classifier1(
                    target_features
                )
            )

            target_logits2 = (
                model.classifier2(
                    target_features
                )
            )

            source_loss = (
                F.cross_entropy(
                    source_logits1,
                    source_y,
                )
                + F.cross_entropy(
                    source_logits2,
                    source_y,
                )
            )

            target_disc = discrepancy(
                target_logits1,
                target_logits2,
            )

            loss_b = (
                source_loss
                - DISCREPANCY_WEIGHT
                * target_disc
            )

            loss_b.backward()
            optimizer_c.step()

        for _ in range(STEP_C_UPDATES):
            optimizer_a.zero_grad(
                set_to_none=True
            )

            target_features = model.features(
                target_x
            )

            target_logits1 = (
                model.classifier1(
                    target_features.detach()
                )
            )

            target_logits2 = (
                model.classifier2(
                    target_features.detach()
                )
            )

            with torch.no_grad():
                target_features_for_adapter = (
                    model.features(
                        target_x
                    )
                )

            adapter_input = target_features_for_adapter

            adapter_output = model.adapter(
                target_x
            )

            logits1 = model.classifier1(
                adapter_output
            )

            logits2 = model.classifier2(
                adapter_output
            )

            loss_c = discrepancy(
                logits1,
                logits2,
            )

            loss_c.backward()
            optimizer_a.step()

        with torch.no_grad():
            source_predictions = (
                (
                    source_logits1
                    + source_logits2
                )
                / 2.0
            ).argmax(dim=1)

            source_correct_total += int(
                (
                    source_predictions
                    == source_y
                )
                .sum()
                .item()
            )

        source_examples_total += (
            source_y.size(0)
        )

        source_loss_total += (
            loss_a.item()
            * source_y.size(0)
        )

        discrepancy_total += (
            target_disc.item()
        )

    return {
        "source_loss":
            source_loss_total
            / max(
                source_examples_total,
                1,
            ),
        "source_accuracy":
            100.0
            * source_correct_total
            / max(
                source_examples_total,
                1,
            ),
        "mean_target_disagreement":
            discrepancy_total
            / max(steps, 1),
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("VISDA-2017 CACHED-FEATURE MCD")
    print("=" * 80)

    print(f"device={device}")
    print(f"seed={SEED}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"epochs={EPOCHS}")
    print(f"input_dim={INPUT_DIM}")
    print(f"hidden_dim={HIDDEN_DIM}")
    print(f"lr_adapter={LR_ADAPTER}")
    print(f"lr_classifier={LR_CLASSIFIER}")
    print(f"discrepancy_weight={DISCREPANCY_WEIGHT}")

    print()
    print("Loading cached source features...")

    source_features, source_labels = load_cache(
        SOURCE_CACHE
    )

    print(
        f"Source features: "
        f"{tuple(source_features.shape)}"
    )

    print()
    print("Loading cached target features...")

    target_features, target_labels = load_cache(
        TARGET_CACHE
    )

    print(
        f"Target features: "
        f"{tuple(target_features.shape)}"
    )

    source_dataset = TensorDataset(
        source_features,
        source_labels,
    )

    target_dataset = TensorDataset(
        target_features,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )

    model = MCDModel().to(device)

    initialize_deterministic_heads(
        model
    )

    adapter_parameters = list(
        model.adapter.parameters()
    )

    classifier_parameters = (
        list(model.classifier1.parameters())
        + list(model.classifier2.parameters())
    )

    optimizer_a = torch.optim.SGD(
        [
            {
                "params": adapter_parameters,
                "lr": LR_ADAPTER,
            },
            {
                "params": classifier_parameters,
                "lr": LR_CLASSIFIER,
            },
        ],
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_c = torch.optim.SGD(
        classifier_parameters,
        lr=LR_CLASSIFIER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a,
        T_max=EPOCHS,
    )

    scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_c,
        T_max=EPOCHS,
    )

    history = []

    print()
    print("=" * 80)
    print("INITIAL TARGET")
    print("=" * 80)

    initial_metrics = evaluate(
        model,
        target_features,
        target_labels,
        BATCH_SIZE * 4,
    )

    print(
        f"Target overall: "
        f"{initial_metrics['overall_accuracy']:.2f}%"
    )

    print(
        f"Target mean-class: "
        f"{initial_metrics['mean_class_accuracy']:.2f}%"
    )

    print(
        f"Target disagreement: "
        f"{initial_metrics['mean_disagreement']:.6f}"
    )

    training_start = time.perf_counter()

    print()
    print("=" * 80)
    print("MCD TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        train_metrics = train_epoch(
            model,
            source_loader,
            target_loader,
            optimizer_a,
            optimizer_c,
            device,
        )

        scheduler_a.step()
        scheduler_c.step()

        target_metrics = evaluate(
            model,
            target_features,
            target_labels,
            BATCH_SIZE * 4,
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "source_loss":
                train_metrics["source_loss"],
            "source_accuracy":
                train_metrics["source_accuracy"],
            "training_target_disagreement":
                train_metrics[
                    "mean_target_disagreement"
                ],
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
            "target_eval_disagreement":
                target_metrics[
                    "mean_disagreement"
                ],
            "epoch_seconds":
                epoch_seconds,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Source {train_metrics['source_accuracy']:.2f}% | "
            f"Target {target_metrics['mean_class_accuracy']:.2f}% | "
            f"Overall {target_metrics['overall_accuracy']:.2f}% | "
            f"Disc {target_metrics['mean_disagreement']:.5f} | "
            f"{epoch_seconds:.2f}s"
        )

    total_seconds = (
        time.perf_counter()
        - training_start
    )

    final_metrics = evaluate(
        model,
        target_features,
        target_labels,
        BATCH_SIZE * 4,
    )

    print()
    print("=" * 80)
    print("FINAL MCD RESULT")
    print("=" * 80)

    print(
        f"Target overall accuracy: "
        f"{final_metrics['overall_accuracy']:.2f}%"
    )

    print(
        f"Target mean-class accuracy: "
        f"{final_metrics['mean_class_accuracy']:.2f}%"
    )

    print(
        f"Target disagreement: "
        f"{final_metrics['mean_disagreement']:.6f}"
    )

    print()
    print("Per-class accuracy:")

    for name, accuracy in zip(
        CLASSES,
        final_metrics[
            "per_class_accuracy"
        ],
    ):
        print(
            f"{name:12s}: "
            f"{accuracy:.2f}%"
        )

    best_record = max(
        history,
        key=lambda x:
            x["target_mean_class_accuracy"],
    )

    print()
    print("=" * 80)
    print("BEST TARGET DIAGNOSTIC")
    print("=" * 80)

    print(
        f"Best observed epoch: "
        f"{best_record['epoch']}"
    )

    print(
        f"Best observed mean-class: "
        f"{best_record['target_mean_class_accuracy']:.2f}%"
    )

    print(
        "Target labels were used only for diagnostics."
    )

    print(
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "mcd_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "mcd_seed42.json"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "seed":
                SEED,
            "input_dim":
                INPUT_DIM,
            "hidden_dim":
                HIDDEN_DIM,
            "num_classes":
                NUM_CLASSES,
            "classes":
                CLASSES,
            "batch_size":
                BATCH_SIZE,
            "epochs":
                EPOCHS,
            "lr_adapter":
                LR_ADAPTER,
            "lr_classifier":
                LR_CLASSIFIER,
            "momentum":
                MOMENTUM,
            "weight_decay":
                WEIGHT_DECAY,
            "discrepancy_weight":
                DISCREPANCY_WEIGHT,
            "final_metrics":
                final_metrics,
            "history":
                history,
        },
        checkpoint_path,
    )

    report = {
        "experiment":
            "visda_cached_mcd",
        "seed":
            SEED,
        "backbone":
            "ImageNet-pretrained ResNet-50 frozen",
        "input_features":
            INPUT_DIM,
        "trainable_adapter":
            f"{INPUT_DIM}->{HIDDEN_DIM}",
        "num_classes":
            NUM_CLASSES,
        "source_samples":
            len(source_features),
        "target_samples":
            len(target_features),
        "batch_size":
            BATCH_SIZE,
        "epochs":
            EPOCHS,
        "lr_adapter":
            LR_ADAPTER,
        "lr_classifier":
            LR_CLASSIFIER,
        "momentum":
            MOMENTUM,
        "weight_decay":
            WEIGHT_DECAY,
        "discrepancy_weight":
            DISCREPANCY_WEIGHT,
        "target_labels_used_for_training":
            False,
        "final_metrics":
            final_metrics,
        "best_observed_epoch":
            best_record["epoch"],
        "best_observed_target_mean_class_accuracy":
            best_record[
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
    print("=" * 80)
    print("FILES")
    print("=" * 80)

    print(
        f"checkpoint={checkpoint_path}"
    )

    print(
        f"history={history_path}"
    )


if __name__ == "__main__":
    main()