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

OUTPUT_DIR = Path("checkpoints/visda_cached_mcd_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EPOCHS = 20

CLASSIFIER_STEPS = 1
GENERATOR_STEPS = 1

LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DISCREPANCY_WEIGHT = 1.0
HEAD_PERTURBATION = 0.005

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    def forward(self, x):
        z = self.adapter(x)

        return (
            self.classifier1(z),
            self.classifier2(z),
        )


def initialize_heads(model):
    with torch.no_grad():
        model.classifier2.weight.copy_(
            model.classifier1.weight
        )

        model.classifier2.bias.copy_(
            model.classifier1.bias
        )

        perturbation = torch.arange(
            model.classifier2.weight.numel(),
            dtype=model.classifier2.weight.dtype,
            device=model.classifier2.weight.device,
        ).reshape_as(
            model.classifier2.weight
        )

        perturbation = torch.where(
            perturbation.remainder(2) == 0,
            torch.ones_like(perturbation),
            -torch.ones_like(perturbation),
        )

        model.classifier2.weight.add_(
            HEAD_PERTURBATION * perturbation
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


def set_requires_grad(module, value):
    for parameter in module.parameters():
        parameter.requires_grad_(value)


@torch.no_grad()
def evaluate(
    model,
    features,
    labels,
    batch_size,
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features,
            labels,
        ),
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
    confidence_sum = 0.0

    for x, y in loader:
        z = model.adapter(x)

        logits1 = model.classifier1(z)
        logits2 = model.classifier2(z)

        p1 = F.softmax(
            logits1,
            dim=1,
        )

        p2 = F.softmax(
            logits2,
            dim=1,
        )

        ensemble = (
            p1 + p2
        ) / 2.0

        predictions = ensemble.argmax(
            dim=1
        )

        confidence = ensemble.max(
            dim=1
        ).values

        disagreement_sum += (
            torch.abs(p1 - p2)
            .mean()
            .item()
            * y.size(0)
        )

        confidence_sum += (
            confidence.mean().item()
            * y.size(0)
        )

        total_correct += int(
            (predictions == y)
            .sum()
            .item()
        )

        total_examples += y.size(0)

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                y == class_id
            )

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

    per_class_accuracy = []

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

        per_class_accuracy.append(
            accuracy
        )

    mean_class_accuracy = float(
        np.mean(per_class_accuracy)
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

    mean_confidence = (
        confidence_sum
        / max(total_examples, 1)
    )

    return {
        "overall_accuracy":
            overall_accuracy,
        "mean_class_accuracy":
            mean_class_accuracy,
        "per_class_accuracy":
            per_class_accuracy,
        "mean_disagreement":
            mean_disagreement,
        "mean_confidence":
            mean_confidence,
    }


def train_one_epoch(
    model,
    source_loader,
    target_loader,
    optimizer_adapter,
    optimizer_classifier,
    device,
):
    source_iter = iter(
        source_loader
    )

    target_iter = iter(
        target_loader
    )

    model.train()

    total_source_loss = 0.0
    total_source_correct = 0
    total_source_examples = 0
    total_classifier_discrepancy = 0.0
    total_generator_discrepancy = 0.0

    steps = len(source_loader)

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

        set_requires_grad(
            model.adapter,
            True,
        )

        set_requires_grad(
            model.classifier1,
            True,
        )

        set_requires_grad(
            model.classifier2,
            True,
        )

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        source_logits1, source_logits2 = (
            model(source_x)
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

        source_loss.backward()
        optimizer_adapter.step()

        total_source_loss += (
            source_loss.item()
            * source_y.size(0)
        )

        with torch.no_grad():
            source_predictions = (
                (
                    source_logits1
                    + source_logits2
                )
                / 2.0
            ).argmax(dim=1)

            total_source_correct += int(
                (
                    source_predictions
                    == source_y
                )
                .sum()
                .item()
            )

        total_source_examples += (
            source_y.size(0)
        )

        set_requires_grad(
            model.adapter,
            False,
        )

        set_requires_grad(
            model.classifier1,
            True,
        )

        set_requires_grad(
            model.classifier2,
            True,
        )

        for _ in range(
            CLASSIFIER_STEPS
        ):
            optimizer_classifier.zero_grad(
                set_to_none=True
            )

            with torch.no_grad():
                source_features = (
                    model.adapter(
                        source_x
                    )
                )

                target_features = (
                    model.adapter(
                        target_x
                    )
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

            source_classifier_loss = (
                F.cross_entropy(
                    source_logits1,
                    source_y,
                )
                + F.cross_entropy(
                    source_logits2,
                    source_y,
                )
            )

            target_discrepancy = discrepancy(
                target_logits1,
                target_logits2,
            )

            classifier_loss = (
                source_classifier_loss
                - DISCREPANCY_WEIGHT
                * target_discrepancy
            )

            classifier_loss.backward()
            optimizer_classifier.step()

            total_classifier_discrepancy += (
                target_discrepancy.item()
            )

        set_requires_grad(
            model.adapter,
            True,
        )

        set_requires_grad(
            model.classifier1,
            False,
        )

        set_requires_grad(
            model.classifier2,
            False,
        )

        for _ in range(
            GENERATOR_STEPS
        ):
            optimizer_adapter.zero_grad(
                set_to_none=True
            )

            target_features = (
                model.adapter(
                    target_x
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

            target_discrepancy = discrepancy(
                target_logits1,
                target_logits2,
            )

            target_discrepancy.backward()
            optimizer_adapter.step()

            total_generator_discrepancy += (
                target_discrepancy.item()
            )

        set_requires_grad(
            model.adapter,
            True,
        )

        set_requires_grad(
            model.classifier1,
            True,
        )

        set_requires_grad(
            model.classifier2,
            True,
        )

    return {
        "source_loss":
            total_source_loss
            / max(
                total_source_examples,
                1,
            ),
        "source_accuracy":
            100.0
            * total_source_correct
            / max(
                total_source_examples,
                1,
            ),
        "classifier_discrepancy":
            total_classifier_discrepancy
            / max(
                steps
                * CLASSIFIER_STEPS,
                1,
            ),
        "generator_discrepancy":
            total_generator_discrepancy
            / max(
                steps
                * GENERATOR_STEPS,
                1,
            ),
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("VISDA-2017 CORRECTED CACHED-FEATURE MCD")
    print("=" * 80)

    print(f"device={device}")
    print(f"seed={SEED}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"epochs={EPOCHS}")
    print(f"input_dim={INPUT_DIM}")
    print(f"hidden_dim={HIDDEN_DIM}")
    print(f"classifier_steps={CLASSIFIER_STEPS}")
    print(f"generator_steps={GENERATOR_STEPS}")
    print(f"lr_adapter={LR_ADAPTER}")
    print(f"lr_classifier={LR_CLASSIFIER}")
    print(
        f"discrepancy_weight={DISCREPANCY_WEIGHT}"
    )
    print(
        f"head_perturbation={HEAD_PERTURBATION}"
    )

    print()
    print("Loading source cache...")

    source_features, source_labels = load_cache(
        SOURCE_CACHE
    )

    print(
        f"Source features: "
        f"{tuple(source_features.shape)}"
    )

    print()
    print("Loading target cache...")

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
        drop_last=True,
        num_workers=0,
        generator=generator,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )

    model = MCDModel().to(device)

    initialize_heads(model)

    adapter_parameters = list(
        model.adapter.parameters()
    )

    classifier_parameters = (
        list(
            model.classifier1.parameters()
        )
        + list(
            model.classifier2.parameters()
        )
    )

    optimizer_adapter = torch.optim.SGD(
        adapter_parameters,
        lr=LR_ADAPTER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_classifier = torch.optim.SGD(
        classifier_parameters,
        lr=LR_CLASSIFIER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_adapter = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_adapter,
            T_max=EPOCHS,
        )
    )

    scheduler_classifier = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_classifier,
            T_max=EPOCHS,
        )
    )

    initial_metrics = evaluate(
        model,
        target_features,
        target_labels,
        2048,
    )

    print()
    print("=" * 80)
    print("INITIAL TARGET")
    print("=" * 80)

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
        f"{initial_metrics['mean_disagreement']:.8f}"
    )

    print(
        f"Target confidence: "
        f"{initial_metrics['mean_confidence']:.6f}"
    )

    history = []

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

        train_metrics = train_one_epoch(
            model,
            source_loader,
            target_loader,
            optimizer_adapter,
            optimizer_classifier,
            device,
        )

        scheduler_adapter.step()
        scheduler_classifier.step()

        target_metrics = evaluate(
            model,
            target_features,
            target_labels,
            2048,
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch":
                epoch,
            "source_loss":
                train_metrics["source_loss"],
            "source_accuracy":
                train_metrics["source_accuracy"],
            "classifier_discrepancy":
                train_metrics[
                    "classifier_discrepancy"
                ],
            "generator_discrepancy":
                train_metrics[
                    "generator_discrepancy"
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
            "target_disagreement":
                target_metrics[
                    "mean_disagreement"
                ],
            "target_confidence":
                target_metrics[
                    "mean_confidence"
                ],
            "epoch_seconds":
                epoch_seconds,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Source "
            f"{train_metrics['source_accuracy']:.2f}% | "
            f"Target "
            f"{target_metrics['mean_class_accuracy']:.2f}% | "
            f"Overall "
            f"{target_metrics['overall_accuracy']:.2f}% | "
            f"Disc "
            f"{target_metrics['mean_disagreement']:.6f} | "
            f"Conf "
            f"{target_metrics['mean_confidence']:.4f} | "
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
        2048,
    )

    best_record = max(
        history,
        key=lambda x:
            x["target_mean_class_accuracy"],
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
        f"{final_metrics['mean_disagreement']:.8f}"
    )

    print(
        f"Target confidence: "
        f"{final_metrics['mean_confidence']:.6f}"
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
        "Target labels were used only "
        "for diagnostics."
    )

    print(
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "mcd_corrected_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "mcd_corrected_seed42.json"
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
            "classifier_steps":
                CLASSIFIER_STEPS,
            "generator_steps":
                GENERATOR_STEPS,
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
            "head_perturbation":
                HEAD_PERTURBATION,
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
        },
        checkpoint_path,
    )

    report = {
        "experiment":
            "visda_cached_mcd_corrected",
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
        "classifier_steps":
            CLASSIFIER_STEPS,
        "generator_steps":
            GENERATOR_STEPS,
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
        "head_perturbation":
            HEAD_PERTURBATION,
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