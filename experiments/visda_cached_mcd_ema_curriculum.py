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

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_cached_mcd_ema_curriculum"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EPOCHS = 20
WARMUP_EPOCHS = 3

CLASSIFIER_STEPS = 1
GENERATOR_STEPS = 1

LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

EMA_DECAY = 0.97

PL_WEIGHT = 1.0

THRESHOLD_START = 0.65
THRESHOLD_END = 0.90

MAX_PL_FRACTION = 0.40

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
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

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
            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM,
            ),
            nn.BatchNorm1d(
                HIDDEN_DIM
            ),
            nn.ReLU(
                inplace=True
            ),
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


def clone_model(model):
    teacher = MCDModel()

    teacher.load_state_dict(
        model.state_dict()
    )

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    return teacher


@torch.no_grad()
def update_ema(
    teacher,
    student,
    decay,
):
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        teacher_parameter.mul_(decay)
        teacher_parameter.add_(
            student_parameter,
            alpha=1.0 - decay,
        )


def discrepancy(
    logits1,
    logits2,
):
    p1 = F.softmax(
        logits1,
        dim=1,
    )

    p2 = F.softmax(
        logits2,
        dim=1,
    )

    return torch.mean(
        torch.abs(
            p1 - p2
        )
    )


def threshold_for_epoch(epoch):
    if epoch <= WARMUP_EPOCHS:
        return float("inf")

    adaptation_epochs = (
        EPOCHS - WARMUP_EPOCHS
    )

    progress = (
        epoch - WARMUP_EPOCHS
    ) / max(
        adaptation_epochs,
        1,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return (
        THRESHOLD_START
        + (
            THRESHOLD_END
            - THRESHOLD_START
        )
        * progress
    )


def cap_selection(
    confidence,
    threshold,
    max_fraction,
):
    n = len(confidence)

    if n == 0:
        return torch.zeros(
            0,
            dtype=torch.bool,
        )

    threshold_mask = (
        confidence >= threshold
    )

    max_count = max(
        1,
        int(
            n
            * max_fraction
        ),
    )

    selected_indices = torch.nonzero(
        threshold_mask,
        as_tuple=False,
    ).flatten()

    if len(selected_indices) <= max_count:
        return threshold_mask

    selected_confidence = confidence[
        selected_indices
    ]

    top_indices = torch.topk(
        selected_confidence,
        k=max_count,
        largest=True,
    ).indices

    selected_indices = selected_indices[
        top_indices
    ]

    result = torch.zeros(
        n,
        dtype=torch.bool,
    )

    result[
        selected_indices
    ] = True

    return result


@torch.no_grad()
def teacher_predictions(
    teacher,
    x,
):
    logits1, logits2 = teacher(x)

    p1 = F.softmax(
        logits1,
        dim=1,
    )

    p2 = F.softmax(
        logits2,
        dim=1,
    )

    p = (
        p1
        + p2
    ) / 2.0

    confidence, pseudo_labels = p.max(
        dim=1
    )

    return (
        p,
        confidence,
        pseudo_labels,
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

    confidence_sum = 0.0
    disagreement_sum = 0.0

    for x, y in loader:
        logits1, logits2 = model(x)

        p1 = F.softmax(
            logits1,
            dim=1,
        )

        p2 = F.softmax(
            logits2,
            dim=1,
        )

        p = (
            p1
            + p2
        ) / 2.0

        confidence, predictions = p.max(
            dim=1
        )

        disagreement = (
            torch.abs(
                p1 - p2
            ).mean(dim=1)
        )

        total_correct += int(
            (
                predictions
                == y
            )
            .sum()
            .item()
        )

        total_examples += y.size(0)

        confidence_sum += (
            confidence.sum().item()
        )

        disagreement_sum += (
            disagreement.sum().item()
        )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = y == class_id

            if mask.any():
                total_per_class[
                    class_id
                ] += int(
                    mask.sum().item()
                )

                correct_per_class[
                    class_id
                ] += int(
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
            total_per_class[
                class_id
            ]
        )

        correct = int(
            correct_per_class[
                class_id
            ]
        )

        accuracy = (
            100.0
            * correct
            / total
            if total > 0
            else 0.0
        )

        per_class_accuracy.append(
            accuracy
        )

    return {
        "overall_accuracy":
            100.0
            * total_correct
            / max(
                total_examples,
                1,
            ),
        "mean_class_accuracy":
            float(
                np.mean(
                    per_class_accuracy
                )
            ),
        "per_class_accuracy":
            per_class_accuracy,
        "mean_confidence":
            confidence_sum
            / max(
                total_examples,
                1,
            ),
        "mean_disagreement":
            disagreement_sum
            / max(
                total_examples,
                1,
            ),
    }


def run_mcd_step(
    student,
    source_x,
    source_y,
    target_x,
    optimizer_adapter,
    optimizer_classifier,
    device,
):
    source_x = source_x.to(device)
    source_y = source_y.to(device)
    target_x = target_x.to(device)

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    source_logits1, source_logits2 = (
        student(source_x)
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

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():
        source_features = student.adapter(
            source_x
        )

        target_features = student.adapter(
            target_x
        )

    source_logits1 = student.classifier1(
        source_features
    )

    source_logits2 = student.classifier2(
        source_features
    )

    target_logits1 = student.classifier1(
        target_features
    )

    target_logits2 = student.classifier2(
        target_features
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
        - target_discrepancy
    )

    classifier_loss.backward()
    optimizer_classifier.step()

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    for parameter in student.classifier1.parameters():
        parameter.requires_grad_(False)

    for parameter in student.classifier2.parameters():
        parameter.requires_grad_(False)

    target_features = student.adapter(
        target_x
    )

    target_logits1 = student.classifier1(
        target_features
    )

    target_logits2 = student.classifier2(
        target_features
    )

    generator_discrepancy = discrepancy(
        target_logits1,
        target_logits2,
    )

    generator_discrepancy.backward()
    optimizer_adapter.step()

    for parameter in student.classifier1.parameters():
        parameter.requires_grad_(True)

    for parameter in student.classifier2.parameters():
        parameter.requires_grad_(True)

    return {
        "source_loss":
            source_loss.item(),
        "discrepancy":
            target_discrepancy.item(),
        "generator_discrepancy":
            generator_discrepancy.item(),
        "source_predictions":
            (
                (
                    source_logits1
                    + source_logits2
                )
                / 2.0
            )
            .argmax(
                dim=1
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
    print(
        "VISDA-2017 MCD + EMA + CONFIDENCE CURRICULUM"
    )
    print("=" * 80)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"batch_size={BATCH_SIZE}"
    )

    print(
        f"epochs={EPOCHS}"
    )

    print(
        f"warmup_epochs={WARMUP_EPOCHS}"
    )

    print(
        f"ema_decay={EMA_DECAY}"
    )

    print(
        f"threshold_start={THRESHOLD_START}"
    )

    print(
        f"threshold_end={THRESHOLD_END}"
    )

    print(
        f"max_pl_fraction={MAX_PL_FRACTION}"
    )

    print()
    print(
        "Loading source cache..."
    )

    source_features, source_labels = load_cache(
        SOURCE_CACHE
    )

    print(
        f"Source features: "
        f"{tuple(source_features.shape)}"
    )

    print()
    print(
        "Loading target cache..."
    )

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

    student = MCDModel().to(device)

    initialize_heads(
        student
    )

    teacher = clone_model(
        student
    ).to(device)

    optimizer_adapter = torch.optim.SGD(
        student.adapter.parameters(),
        lr=LR_ADAPTER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_classifier = torch.optim.SGD(
        list(
            student.classifier1.parameters()
        )
        + list(
            student.classifier2.parameters()
        ),
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

    initial_student = evaluate(
        student,
        target_features,
        target_labels,
        4096,
    )

    initial_teacher = evaluate(
        teacher,
        target_features,
        target_labels,
        4096,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "INITIAL TARGET"
    )

    print(
        f"Student mean-class: "
        f"{initial_student['mean_class_accuracy']:.2f}%"
    )

    print(
        f"Teacher mean-class: "
        f"{initial_teacher['mean_class_accuracy']:.2f}%"
    )

    history = []

    source_iterator = iter(
        source_loader
    )

    target_iterator = iter(
        target_loader
    )

    start_time = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        threshold = threshold_for_epoch(
            epoch
        )

        epoch_source_loss = 0.0
        epoch_discrepancy = 0.0
        epoch_generator_discrepancy = 0.0

        epoch_source_correct = 0
        epoch_source_examples = 0

        epoch_selected = 0
        epoch_target_examples = 0
        epoch_selected_confidence = 0.0

        steps = len(
            source_loader
        )

        for _ in range(steps):
            try:
                source_x, source_y = next(
                    source_iterator
                )
            except StopIteration:
                source_iterator = iter(
                    source_loader
                )
                source_x, source_y = next(
                    source_iterator
                )

            try:
                target_x = next(
                    target_iterator
                )[0]
            except StopIteration:
                target_iterator = iter(
                    target_loader
                )
                target_x = next(
                    target_iterator
                )[0]

            metrics = run_mcd_step(
                student,
                source_x,
                source_y,
                target_x,
                optimizer_adapter,
                optimizer_classifier,
                device,
            )

            epoch_source_loss += (
                metrics["source_loss"]
            )

            epoch_discrepancy += (
                metrics["discrepancy"]
            )

            epoch_generator_discrepancy += (
                metrics[
                    "generator_discrepancy"
                ]
            )

            epoch_source_correct += int(
                (
                    metrics[
                        "source_predictions"
                    ]
                    == source_y.to(device)
                )
                .sum()
                .item()
            )

            epoch_source_examples += (
                source_y.size(0)
            )

            update_ema(
                teacher,
                student,
                EMA_DECAY,
            )

            with torch.no_grad():
                target_x_device = target_x.to(
                    device
                )

                _, confidence, pseudo_labels = (
                    teacher_predictions(
                        teacher,
                        target_x_device,
                    )
                )

            if epoch <= WARMUP_EPOCHS:
                selected = torch.zeros_like(
                    confidence,
                    dtype=torch.bool,
                )
            else:
                selected = cap_selection(
                    confidence.cpu(),
                    threshold,
                    MAX_PL_FRACTION,
                ).to(device)

            selected_count = int(
                selected.sum().item()
            )

            epoch_selected += (
                selected_count
            )

            epoch_target_examples += (
                target_x.size(0)
            )

            if selected_count > 0:
                epoch_selected_confidence += (
                    confidence[
                        selected
                    ]
                    .sum()
                    .item()
                )

                optimizer_adapter.zero_grad(
                    set_to_none=True
                )

                selected_x = target_x_device[
                    selected
                ]

                selected_labels = pseudo_labels[
                    selected
                ]

                selected_confidence = confidence[
                    selected
                ].detach()

                student_features = student.adapter(
                    selected_x
                )

                logits1 = student.classifier1(
                    student_features
                )

                logits2 = student.classifier2(
                    student_features
                )

                loss1 = F.cross_entropy(
                    logits1,
                    selected_labels,
                    reduction="none",
                )

                loss2 = F.cross_entropy(
                    logits2,
                    selected_labels,
                    reduction="none",
                )

                pseudo_loss = (
                    (
                        loss1
                        + loss2
                    )
                    * selected_confidence
                ).mean()

                (
                    PL_WEIGHT
                    * pseudo_loss
                ).backward()

                optimizer_adapter.step()

                optimizer_classifier.zero_grad(
                    set_to_none=True
                )

                with torch.no_grad():
                    selected_features = (
                        student.adapter(
                            selected_x
                        )
                    )

                classifier_logits1 = (
                    student.classifier1(
                        selected_features
                    )
                )

                classifier_logits2 = (
                    student.classifier2(
                        selected_features
                    )
                )

                classifier_loss = (
                    F.cross_entropy(
                        classifier_logits1,
                        selected_labels,
                        reduction="none",
                    )
                    + F.cross_entropy(
                        classifier_logits2,
                        selected_labels,
                        reduction="none",
                    )
                )

                classifier_loss = (
                    classifier_loss
                    * selected_confidence
                ).mean()

                (
                    PL_WEIGHT
                    * classifier_loss
                ).backward()

                optimizer_classifier.step()

                update_ema(
                    teacher,
                    student,
                    EMA_DECAY,
                )

        scheduler_adapter.step()
        scheduler_classifier.step()

        student_metrics = evaluate(
            student,
            target_features,
            target_labels,
            4096,
        )

        teacher_metrics = evaluate(
            teacher,
            target_features,
            target_labels,
            4096,
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        pseudo_coverage = (
            100.0
            * epoch_selected
            / max(
                epoch_target_examples,
                1,
            )
        )

        mean_selected_confidence = (
            epoch_selected_confidence
            / max(
                epoch_selected,
                1,
            )
        )

        source_accuracy = (
            100.0
            * epoch_source_correct
            / max(
                epoch_source_examples,
                1,
            )
        )

        record = {
            "epoch":
                epoch,
            "threshold":
                threshold
                if np.isfinite(threshold)
                else None,
            "source_loss":
                epoch_source_loss
                / max(
                    steps,
                    1,
                ),
            "source_accuracy":
                source_accuracy,
            "mcd_discrepancy":
                epoch_discrepancy
                / max(
                    steps,
                    1,
                ),
            "generator_discrepancy":
                epoch_generator_discrepancy
                / max(
                    steps,
                    1,
                ),
            "pseudo_selected":
                epoch_selected,
            "pseudo_total":
                epoch_target_examples,
            "pseudo_coverage_percent":
                pseudo_coverage,
            "mean_selected_confidence":
                mean_selected_confidence,
            "student_target_overall":
                student_metrics[
                    "overall_accuracy"
                ],
            "student_target_mean_class":
                student_metrics[
                    "mean_class_accuracy"
                ],
            "student_target_confidence":
                student_metrics[
                    "mean_confidence"
                ],
            "teacher_target_overall":
                teacher_metrics[
                    "overall_accuracy"
                ],
            "teacher_target_mean_class":
                teacher_metrics[
                    "mean_class_accuracy"
                ],
            "teacher_target_confidence":
                teacher_metrics[
                    "mean_confidence"
                ],
            "epoch_seconds":
                epoch_seconds,
        }

        history.append(
            record
        )

        threshold_text = (
            "WARMUP"
            if not np.isfinite(threshold)
            else f"{threshold:.3f}"
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Thr {threshold_text:>7s} | "
            f"Source {source_accuracy:.2f}% | "
            f"Student {student_metrics['mean_class_accuracy']:.2f}% | "
            f"Teacher {teacher_metrics['mean_class_accuracy']:.2f}% | "
            f"PL {pseudo_coverage:.1f}% | "
            f"PLconf {mean_selected_confidence:.4f} | "
            f"{epoch_seconds:.2f}s"
        )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    final_student = evaluate(
        student,
        target_features,
        target_labels,
        4096,
    )

    final_teacher = evaluate(
        teacher,
        target_features,
        target_labels,
        4096,
    )

    best_student = max(
        history,
        key=lambda x:
            x[
                "student_target_mean_class"
            ],
    )

    best_teacher = max(
        history,
        key=lambda x:
            x[
                "teacher_target_mean_class"
            ],
    )

    print()
    print("=" * 80)
    print(
        "FINAL RESULT"
    )
    print("=" * 80)

    print(
        f"Student target overall: "
        f"{final_student['overall_accuracy']:.2f}%"
    )

    print(
        f"Student target mean-class: "
        f"{final_student['mean_class_accuracy']:.2f}%"
    )

    print(
        f"Teacher target overall: "
        f"{final_teacher['overall_accuracy']:.2f}%"
    )

    print(
        f"Teacher target mean-class: "
        f"{final_teacher['mean_class_accuracy']:.2f}%"
    )

    print()
    print(
        f"Best student epoch: "
        f"{best_student['epoch']}"
    )

    print(
        f"Best student mean-class: "
        f"{best_student['student_target_mean_class']:.2f}%"
    )

    print(
        f"Best teacher epoch: "
        f"{best_teacher['epoch']}"
    )

    print(
        f"Best teacher mean-class: "
        f"{best_teacher['teacher_target_mean_class']:.2f}%"
    )

    print()
    print(
        "Student per-class accuracy:"
    )

    for class_name, accuracy in zip(
        CLASSES,
        final_student[
            "per_class_accuracy"
        ],
    ):
        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}%"
        )

    print()
    print(
        "Teacher per-class accuracy:"
    )

    for class_name, accuracy in zip(
        CLASSES,
        final_teacher[
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
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "mcd_ema_curriculum_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "mcd_ema_curriculum_seed42.json"
    )

    torch.save(
        {
            "student_state_dict":
                student.state_dict(),
            "teacher_state_dict":
                teacher.state_dict(),
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
            "warmup_epochs":
                WARMUP_EPOCHS,
            "ema_decay":
                EMA_DECAY,
            "threshold_start":
                THRESHOLD_START,
            "threshold_end":
                THRESHOLD_END,
            "max_pl_fraction":
                MAX_PL_FRACTION,
            "pl_weight":
                PL_WEIGHT,
            "head_perturbation":
                HEAD_PERTURBATION,
            "final_student":
                final_student,
            "final_teacher":
                final_teacher,
            "history":
                history,
        },
        checkpoint_path,
    )

    report = {
        "experiment":
            "visda_cached_mcd_ema_confidence_curriculum",
        "seed":
            SEED,
        "backbone":
            "ImageNet-pretrained ResNet-50 frozen",
        "trainable_adapter":
            f"{INPUT_DIM}->{HIDDEN_DIM}",
        "target_labels_used_for_training":
            False,
        "batch_size":
            BATCH_SIZE,
        "epochs":
            EPOCHS,
        "warmup_epochs":
            WARMUP_EPOCHS,
        "ema_decay":
            EMA_DECAY,
        "threshold_start":
            THRESHOLD_START,
        "threshold_end":
            THRESHOLD_END,
        "max_pl_fraction":
            MAX_PL_FRACTION,
        "pl_weight":
            PL_WEIGHT,
        "final_student":
            final_student,
        "final_teacher":
            final_teacher,
        "best_student_epoch":
            best_student["epoch"],
        "best_student_mean_class":
            best_student[
                "student_target_mean_class"
            ],
        "best_teacher_epoch":
            best_teacher["epoch"],
        "best_teacher_mean_class":
            best_teacher[
                "teacher_target_mean_class"
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
    print(
        "FILES"
    )
    print("=" * 80)

    print(
        f"checkpoint={checkpoint_path}"
    )

    print(
        f"history={history_path}"
    )


if __name__ == "__main__":
    main()