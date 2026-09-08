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

OUTPUT_DIR = Path(
    "checkpoints/visda_mcd_class_budget_control"
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

LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

EMA_DECAY = 0.97

PL_WEIGHT = 1.0

THRESHOLD_START = 0.65
THRESHOLD_END = 0.90

GLOBAL_PL_FRACTION = 0.40

SOURCE_PRIOR_CAP_MULTIPLIER = 2.0
MAX_CLASS_CAP = 0.40
MIN_CLASS_CAP = 0.001

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
            nn.LayerNorm(
                HIDDEN_DIM,
            ),
            nn.ReLU(
                inplace=True,
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
            HEAD_PERTURBATION
            * perturbation
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
        p1 + p2
    ) / 2.0

    confidence, pseudo_labels = p.max(
        dim=1
    )

    return (
        p,
        confidence,
        pseudo_labels,
    )


def threshold_for_epoch(epoch):
    if epoch <= WARMUP_EPOCHS:
        return float("inf")

    span = max(
        EPOCHS - WARMUP_EPOCHS,
        1,
    )

    progress = (
        epoch - WARMUP_EPOCHS
    ) / span

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return (
        THRESHOLD_START
        + (
            THRESHOLD_END
            - THRESHOLD_START
        ) * progress
    )


def compute_source_prior(source_labels):
    counts = torch.bincount(
        source_labels,
        minlength=NUM_CLASSES,
    ).float()

    return counts / counts.sum().clamp_min(1.0)


def compute_class_caps(source_prior):
    caps = (
        SOURCE_PRIOR_CAP_MULTIPLIER
        * source_prior
    )

    return torch.clamp(
        caps,
        min=MIN_CLASS_CAP,
        max=MAX_CLASS_CAP,
    )


def select_with_source_caps(
    confidence,
    predictions,
    threshold,
    global_fraction,
    class_caps,
):
    n = len(confidence)

    eligible = (
        confidence >= threshold
    )

    selected = torch.zeros(
        n,
        dtype=torch.bool,
    )

    candidate_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    selected_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    global_limit = max(
        1,
        int(
            n
            * global_fraction
        ),
    )

    for class_id in range(NUM_CLASSES):
        mask = (
            eligible
            & (
                predictions
                == class_id
            )
        )

        indices = torch.nonzero(
            mask,
            as_tuple=False,
        ).flatten()

        candidate_counts[
            class_id
        ] = len(indices)

        if len(indices) == 0:
            continue

        order = torch.argsort(
            confidence[
                indices
            ],
            descending=True,
        )

        indices = indices[
            order
        ]

        class_limit = max(
            1,
            int(
                n
                * class_caps[
                    class_id
                ].item()
            ),
        )

        indices = indices[
            :class_limit
        ]

        selected[
            indices
        ] = True

        selected_counts[
            class_id
        ] = len(indices)

    selected_indices = torch.nonzero(
        selected,
        as_tuple=False,
    ).flatten()

    if len(selected_indices) > global_limit:
        order = torch.argsort(
            confidence[
                selected_indices
            ],
            descending=True,
        )

        keep = selected_indices[
            order[:global_limit]
        ]

        selected.zero_()

        selected[
            keep
        ] = True

        selected_counts.zero_()

        selected_counts.scatter_add_(
            0,
            predictions[
                keep
            ],
            torch.ones(
                len(keep),
                dtype=torch.long,
            ),
        )

    return (
        selected,
        candidate_counts,
        selected_counts,
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
        drop_last=False,
    )

    total_correct = 0
    total_examples = 0
    confidence_sum = 0.0

    per_class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    per_class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

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
            p1 + p2
        ) / 2.0

        confidence, predictions = p.max(
            dim=1
        )

        total_correct += int(
            (
                predictions == y
            ).sum().item()
        )

        total_examples += y.size(0)

        confidence_sum += (
            confidence.sum().item()
        )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = y == class_id

            if mask.any():
                per_class_total[
                    class_id
                ] += int(
                    mask.sum().item()
                )

                per_class_correct[
                    class_id
                ] += int(
                    (
                        predictions[mask]
                        == y[mask]
                    ).sum().item()
                )

    per_class_accuracy = (
        100.0
        * per_class_correct.float()
        / per_class_total.clamp_min(1)
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
                per_class_accuracy.mean().item()
            ),
        "per_class_accuracy":
            per_class_accuracy.tolist(),
        "mean_confidence":
            confidence_sum
            / max(
                total_examples,
                1,
            ),
    }


def train_one_epoch(
    student,
    teacher,
    source_loader,
    target_loader,
    optimizer_adapter,
    optimizer_classifier,
    device,
    class_caps,
    epoch,
):
    source_iter = iter(
        source_loader
    )

    target_iter = iter(
        target_loader
    )

    steps = len(
        source_loader
    )

    threshold = threshold_for_epoch(
        epoch
    )

    source_correct = 0
    source_total = 0

    source_loss_sum = 0.0
    discrepancy_sum = 0.0
    pseudo_loss_sum = 0.0

    pseudo_selected = 0
    pseudo_total = 0
    selected_confidence_sum = 0.0

    candidate_counts_epoch = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    selected_counts_epoch = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

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
            target_batch = next(
                target_iter
            )

            target_x = target_batch[0]

        except StopIteration:
            target_iter = iter(
                target_loader
            )

            target_batch = next(
                target_iter
            )

            target_x = target_batch[0]

        source_x = source_x.to(
            device
        )

        source_y = source_y.to(
            device
        )

        target_x = target_x.to(
            device
        )

        if source_x.size(0) < 2:
            continue

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

        source_predictions = (
            (
                source_logits1
                + source_logits2
            )
            / 2.0
        ).argmax(
            dim=1
        )

        source_correct += int(
            (
                source_predictions
                == source_y
            ).sum().item()
        )

        source_total += source_y.size(0)

        source_loss_sum += (
            source_loss.item()
        )

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

        (
            source_classifier_loss
            - target_discrepancy
        ).backward()

        optimizer_classifier.step()

        discrepancy_sum += (
            target_discrepancy.item()
        )

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

        update_ema(
            teacher,
            student,
            EMA_DECAY,
        )

        with torch.no_grad():
            (
                _,
                confidence,
                pseudo_labels,
            ) = teacher_predictions(
                teacher,
                target_x,
            )

        if epoch <= WARMUP_EPOCHS:
            selected = torch.zeros(
                len(target_x),
                dtype=torch.bool,
                device=device,
            )

            candidate_counts = torch.zeros(
                NUM_CLASSES,
                dtype=torch.long,
            )

            selected_counts = torch.zeros(
                NUM_CLASSES,
                dtype=torch.long,
            )

        else:
            (
                selected,
                candidate_counts,
                selected_counts,
            ) = select_with_source_caps(
                confidence.cpu(),
                pseudo_labels.cpu(),
                threshold,
                GLOBAL_PL_FRACTION,
                class_caps,
            )

            selected = selected.to(
                device
            )

            candidate_counts_epoch += (
                candidate_counts
            )

            selected_counts_epoch += (
                selected_counts
            )

        selected_count = int(
            selected.sum().item()
        )

        pseudo_selected += selected_count

        pseudo_total += target_x.size(0)

        if selected_count > 0:
            selected_confidence = (
                confidence[
                    selected
                ].detach()
            )

            selected_confidence_sum += (
                selected_confidence.sum().item()
            )

            optimizer_adapter.zero_grad(
                set_to_none=True
            )

            selected_x = target_x[
                selected
            ]

            selected_y = pseudo_labels[
                selected
            ]

            selected_features = student.adapter(
                selected_x
            )

            logits1 = student.classifier1(
                selected_features
            )

            logits2 = student.classifier2(
                selected_features
            )

            loss1 = F.cross_entropy(
                logits1,
                selected_y,
                reduction="none",
            )

            loss2 = F.cross_entropy(
                logits2,
                selected_y,
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

            classifier_pseudo_loss = (
                F.cross_entropy(
                    classifier_logits1,
                    selected_y,
                    reduction="none",
                )
                + F.cross_entropy(
                    classifier_logits2,
                    selected_y,
                    reduction="none",
                )
            )

            classifier_pseudo_loss = (
                classifier_pseudo_loss
                * selected_confidence
            ).mean()

            (
                PL_WEIGHT
                * classifier_pseudo_loss
            ).backward()

            optimizer_classifier.step()

            update_ema(
                teacher,
                student,
                EMA_DECAY,
            )

            pseudo_loss_sum += (
                pseudo_loss.item()
            )

    return {
        "threshold":
            threshold,
        "source_accuracy":
            100.0
            * source_correct
            / max(
                source_total,
                1,
            ),
        "source_loss":
            source_loss_sum
            / max(
                steps,
                1,
            ),
        "discrepancy":
            discrepancy_sum
            / max(
                steps,
                1,
            ),
        "pseudo_loss":
            pseudo_loss_sum
            / max(
                steps,
                1,
            ),
        "pseudo_selected":
            pseudo_selected,
        "pseudo_total":
            pseudo_total,
        "pseudo_coverage":
            100.0
            * pseudo_selected
            / max(
                pseudo_total,
                1,
            ),
        "pseudo_confidence":
            selected_confidence_sum
            / max(
                pseudo_selected,
                1,
            ),
        "candidate_counts":
            candidate_counts_epoch,
        "selected_counts":
            selected_counts_epoch,
    }


@torch.no_grad()
def get_prediction_counts(
    teacher,
    features,
):
    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            features
        ),
        batch_size=4096,
        shuffle=False,
        num_workers=0,
    )

    for (x,) in loader:
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
            p1 + p2
        ) / 2.0

        predictions = p.argmax(
            dim=1
        )

        counts += torch.bincount(
            predictions,
            minlength=NUM_CLASSES,
        )

    return counts


def main():
    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print(
        "VISDA-2017 MCD + EMA + CLASS-BUDGET CONTROL"
    )
    print("=" * 90)

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
        f"global_pl_fraction={GLOBAL_PL_FRACTION}"
    )

    print(
        f"source_prior_cap_multiplier="
        f"{SOURCE_PRIOR_CAP_MULTIPLIER}"
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

    source_prior = compute_source_prior(
        source_labels
    )

    class_caps = compute_class_caps(
        source_prior
    )

    print()
    print(
        "SOURCE PRIOR CAPS"
    )

    for class_name, prior, cap in zip(
        CLASSES,
        source_prior.tolist(),
        class_caps.tolist(),
    ):
        print(
            f"{class_name:12s} | "
            f"prior={prior:.5f} | "
            f"cap={cap:.5f}"
        )

    source_dataset = TensorDataset(
        source_features,
        source_labels,
    )

    target_dataset = TensorDataset(
        target_features
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

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

    student = MCDModel().to(
        device
    )

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
        "=" * 90
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

    start_time = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        metrics = train_one_epoch(
            student,
            teacher,
            source_loader,
            target_loader,
            optimizer_adapter,
            optimizer_classifier,
            device,
            class_caps,
            epoch,
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

        teacher_counts = get_prediction_counts(
            teacher,
            target_features,
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch":
                epoch,
            "threshold":
                metrics["threshold"]
                if np.isfinite(
                    metrics["threshold"]
                )
                else None,
            "source_accuracy":
                metrics["source_accuracy"],
            "source_loss":
                metrics["source_loss"],
            "mcd_discrepancy":
                metrics["discrepancy"],
            "pseudo_loss":
                metrics["pseudo_loss"],
            "pseudo_selected":
                metrics["pseudo_selected"],
            "pseudo_total":
                metrics["pseudo_total"],
            "pseudo_coverage":
                metrics["pseudo_coverage"],
            "pseudo_confidence":
                metrics["pseudo_confidence"],
            "student_target_overall":
                student_metrics[
                    "overall_accuracy"
                ],
            "student_target_mean_class":
                student_metrics[
                    "mean_class_accuracy"
                ],
            "teacher_target_overall":
                teacher_metrics[
                    "overall_accuracy"
                ],
            "teacher_target_mean_class":
                teacher_metrics[
                    "mean_class_accuracy"
                ],
            "teacher_prediction_counts":
                teacher_counts.tolist(),
            "candidate_counts":
                metrics[
                    "candidate_counts"
                ].tolist(),
            "selected_counts":
                metrics[
                    "selected_counts"
                ].tolist(),
            "class_caps":
                class_caps.tolist(),
            "epoch_seconds":
                epoch_seconds,
        }

        history.append(
            record
        )

        threshold_text = (
            "WARMUP"
            if not np.isfinite(
                metrics["threshold"]
            )
            else f"{metrics['threshold']:.3f}"
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Thr {threshold_text:>7s} | "
            f"Source "
            f"{metrics['source_accuracy']:.2f}% | "
            f"Student "
            f"{student_metrics['mean_class_accuracy']:.2f}% | "
            f"Teacher "
            f"{teacher_metrics['mean_class_accuracy']:.2f}% | "
            f"PL "
            f"{metrics['pseudo_coverage']:.2f}% | "
            f"PLconf "
            f"{metrics['pseudo_confidence']:.4f} | "
            f"{epoch_seconds:.2f}s"
        )

        if epoch > WARMUP_EPOCHS:
            print(
                "Selected pseudo-labels:"
            )

            for class_name, count in zip(
                CLASSES,
                metrics[
                    "selected_counts"
                ].tolist(),
            ):
                print(
                    f"  {class_name:12s}: "
                    f"{count}"
                )

            print(
                "Teacher prediction distribution:"
            )

            teacher_total = max(
                int(
                    teacher_counts.sum().item()
                ),
                1,
            )

            for class_name, count in zip(
                CLASSES,
                teacher_counts.tolist(),
            ):
                print(
                    f"  {class_name:12s}: "
                    f"{count:6d} "
                    f"({100.0 * count / teacher_total:6.2f}%)"
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
    print(
        "=" * 90
    )

    print(
        "FINAL RESULT"
    )

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
        "Final class caps:"
    )

    for class_name, cap in zip(
        CLASSES,
        class_caps.tolist(),
    ):
        print(
            f"{class_name:12s}: "
            f"{cap:.5f}"
        )

    print()
    print(
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "class_budget_control_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "class_budget_control_seed42.json"
    )

    torch.save(
        {
            "student_state_dict":
                student.state_dict(),
            "teacher_state_dict":
                teacher.state_dict(),
            "seed":
                SEED,
            "classes":
                CLASSES,
            "source_prior":
                source_prior.tolist(),
            "class_caps":
                class_caps.tolist(),
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
        },
        checkpoint_path,
    )

    report = {
        "experiment":
            "visda_mcd_ema_class_budget_control",
        "seed":
            SEED,
        "target_labels_used_for_training":
            False,
        "source_prior":
            source_prior.tolist(),
        "class_caps":
            class_caps.tolist(),
        "global_pl_fraction":
            GLOBAL_PL_FRACTION,
        "threshold_start":
            THRESHOLD_START,
        "threshold_end":
            THRESHOLD_END,
        "ema_decay":
            EMA_DECAY,
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
    print(
        "=" * 90
    )

    print(
        "FILES"
    )

    print(
        "=" * 90
    )

    print(
        f"checkpoint={checkpoint_path}"
    )

    print(
        f"history={history_path}"
    )


if __name__ == "__main__":
    main()