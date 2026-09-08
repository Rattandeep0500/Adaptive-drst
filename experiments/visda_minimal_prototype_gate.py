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
    "checkpoints/visda_minimal_prototype_gate"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
EPOCHS = 20
WARMUP_EPOCHS = 3

LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

EMA_DECAY = 0.97

HEAD_PERTURBATION = 0.005

PL_THRESHOLD_START = 0.65
PL_THRESHOLD_END = 0.90

GLOBAL_PL_FRACTION = 0.25

PROTOTYPE_THRESHOLD = 0.0

PL_WEIGHT = 0.20
PROTOTYPE_WEIGHT = 0.05

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

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
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
            f"Features must be 2D, got {tuple(features.shape)}"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected feature dimension {INPUT_DIM}, "
            f"got {features.shape[1]}"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Feature and label counts do not match"
        )

    if labels.min().item() < 0 or labels.max().item() >= NUM_CLASSES:
        raise RuntimeError(
            "Labels are outside the expected class range"
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
        if x.ndim != 2:
            raise RuntimeError(
                f"Adapter expected 2D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expected {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter produced {z.shape[1]}-D features"
            )

        return z


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

    def encode(self, x):
        return self.adapter(x)

    def forward(self, x):
        z = self.encode(x)

        logits1 = self.classifier1(z)
        logits2 = self.classifier2(z)

        if logits1.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                f"Classifier output dimension is {logits1.shape[1]}"
            )

        return logits1, logits2


def initialize_heads(model):
    with torch.no_grad():
        model.classifier2.weight.copy_(
            model.classifier1.weight
        )

        model.classifier2.bias.copy_(
            model.classifier1.bias
        )

        values = torch.arange(
            model.classifier2.weight.numel(),
            dtype=model.classifier2.weight.dtype,
        ).reshape_as(
            model.classifier2.weight
        )

        signs = torch.where(
            values.remainder(2) == 0,
            torch.ones_like(values),
            -torch.ones_like(values),
        )

        model.classifier2.weight.add_(
            HEAD_PERTURBATION * signs
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
    teacher_parameters = list(
        teacher.parameters()
    )

    student_parameters = list(
        student.parameters()
    )

    if len(teacher_parameters) != len(
        student_parameters
    ):
        raise RuntimeError(
            "Teacher and student parameter counts differ"
        )

    for teacher_parameter, student_parameter in zip(
        teacher_parameters,
        student_parameters,
    ):
        teacher_parameter.mul_(
            decay
        )

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
def teacher_predict(
    teacher,
    x,
):
    if x.ndim != 2:
        raise RuntimeError(
            f"Teacher input must be 2D, got {tuple(x.shape)}"
        )

    if x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Teacher expected {INPUT_DIM}-D input, "
            f"got {x.shape[1]}-D"
        )

    logits1, logits2 = teacher(x)

    p1 = F.softmax(
        logits1,
        dim=1,
    )

    p2 = F.softmax(
        logits2,
        dim=1,
    )

    probabilities = (
        p1 + p2
    ) / 2.0

    confidence, predictions = probabilities.max(
        dim=1
    )

    return (
        probabilities,
        confidence,
        predictions,
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
        PL_THRESHOLD_START
        + (
            PL_THRESHOLD_END
            - PL_THRESHOLD_START
        ) * progress
    )


@torch.no_grad()
def compute_source_prototypes(
    model,
    source_features,
    source_labels,
    device,
):
    model.eval()

    prototype_sums = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64,
    )

    prototype_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            source_features,
            source_labels,
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    for source_x, source_y in loader:
        source_x = source_x.to(device)

        source_z = model.encode(
            source_x
        )

        if source_z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Source encoded dimension is {source_z.shape[1]}"
            )

        source_z = F.normalize(
            source_z,
            dim=1,
        )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                source_y == class_id
            )

            if mask.any():
                mask_device = mask.to(device)

                class_features = source_z[
                    mask_device
                ]

                prototype_sums[
                    class_id
                ] += class_features.double().sum(
                    dim=0
                ).cpu()

                prototype_counts[
                    class_id
                ] += int(
                    mask.sum().item()
                )

    if (
        prototype_counts == 0
    ).any():
        missing = torch.nonzero(
            prototype_counts == 0,
            as_tuple=False,
        ).flatten().tolist()

        raise RuntimeError(
            f"Missing source prototypes for classes {missing}"
        )

    prototypes = (
        prototype_sums
        / prototype_counts.double().unsqueeze(1)
    )

    prototypes = prototypes.float()

    prototypes = F.normalize(
        prototypes,
        dim=1,
    )

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM,
    ):
        raise RuntimeError(
            f"Invalid prototype shape {tuple(prototypes.shape)}"
        )

    return prototypes


def cosine_to_predicted_prototype(
    features,
    predictions,
    prototypes,
):
    if features.ndim != 2:
        raise RuntimeError(
            f"Expected feature matrix, got {tuple(features.shape)}"
        )

    if features.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            f"Expected {HIDDEN_DIM}-D adapted features, "
            f"got {features.shape[1]}-D"
        )

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM,
    ):
        raise RuntimeError(
            f"Invalid prototype shape {tuple(prototypes.shape)}"
        )

    normalized_features = F.normalize(
        features,
        dim=1,
    )

    normalized_prototypes = F.normalize(
        prototypes,
        dim=1,
    )

    selected_prototypes = normalized_prototypes[
        predictions
    ]

    similarity = (
        normalized_features
        * selected_prototypes
    ).sum(
        dim=1
    )

    return similarity


def select_pseudo_labels(
    confidence,
    predictions,
    prototype_similarity,
    threshold,
):
    count = len(confidence)

    if not np.isfinite(
        threshold
    ):
        return torch.zeros(
            count,
            dtype=torch.bool,
        )

    confidence_mask = (
        confidence >= threshold
    )

    geometry_mask = (
        prototype_similarity
        >= PROTOTYPE_THRESHOLD
    )

    eligible = (
        confidence_mask
        & geometry_mask
    )

    eligible_indices = torch.nonzero(
        eligible,
        as_tuple=False,
    ).flatten()

    if len(eligible_indices) == 0:
        return torch.zeros(
            count,
            dtype=torch.bool,
        )

    global_limit = max(
        1,
        int(
            count
            * GLOBAL_PL_FRACTION
        ),
    )

    if len(eligible_indices) > global_limit:
        ordering = torch.argsort(
            confidence[
                eligible_indices
            ],
            descending=True,
        )

        eligible_indices = eligible_indices[
            ordering[
                :global_limit
            ]
        ]

    selected = torch.zeros(
        count,
        dtype=torch.bool,
    )

    selected[
        eligible_indices
    ] = True

    return selected


def evaluate(
    model,
    features,
    labels,
    device,
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features,
            labels,
        ),
        batch_size=EVAL_BATCH_SIZE,
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

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_device = y.to(device)

            logits1, logits2 = model(x)

            p1 = F.softmax(
                logits1,
                dim=1,
            )

            p2 = F.softmax(
                logits2,
                dim=1,
            )

            probabilities = (
                p1 + p2
            ) / 2.0

            confidence, predictions = probabilities.max(
                dim=1
            )

            total_correct += int(
                (
                    predictions
                    == y_device
                ).sum().item()
            )

            total_examples += (
                y.size(0)
            )

            confidence_sum += (
                confidence.sum().item()
            )

            for class_id in range(
                NUM_CLASSES
            ):
                mask = (
                    y
                    == class_id
                )

                if mask.any():
                    mask_device = mask.to(device)

                    per_class_total[
                        class_id
                    ] += int(
                        mask.sum().item()
                    )

                    per_class_correct[
                        class_id
                    ] += int(
                        (
                            predictions[
                                mask_device
                            ]
                            == y_device[
                                mask_device
                            ]
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
    prototypes,
    epoch,
    device,
):
    student.train()
    teacher.eval()

    source_iterator = iter(
        source_loader
    )

    target_iterator = iter(
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
    prototype_loss_sum = 0.0

    pseudo_selected = 0
    pseudo_total = 0
    pseudo_confidence_sum = 0.0
    prototype_similarity_sum = 0.0

    selected_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    predicted_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for _ in range(
        steps
    ):
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

        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        if source_x.size(0) < 2:
            continue

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        source_logits1, source_logits2 = student(
            source_x
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

        source_total += (
            source_y.size(0)
        )

        source_loss_sum += (
            source_loss.item()
        )

        optimizer_classifier.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            source_z = student.encode(
                source_x
            )

            target_z = student.encode(
                target_x
            )

        if source_z.shape != (
            source_x.size(0),
            HIDDEN_DIM,
        ):
            raise RuntimeError(
                f"Invalid source_z shape {tuple(source_z.shape)}"
            )

        if target_z.shape != (
            target_x.size(0),
            HIDDEN_DIM,
        ):
            raise RuntimeError(
                f"Invalid target_z shape {tuple(target_z.shape)}"
            )

        source_logits1 = student.classifier1(
            source_z
        )

        source_logits2 = student.classifier2(
            source_z
        )

        target_logits1 = student.classifier1(
            target_z
        )

        target_logits2 = student.classifier2(
            target_z
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

        discrepancy_sum += (
            target_discrepancy.item()
        )

        update_ema(
            teacher,
            student,
            EMA_DECAY,
        )

        with torch.no_grad():
            (
                target_probabilities,
                target_confidence,
                target_predictions,
            ) = teacher_predict(
                teacher,
                target_x,
            )

        predicted_counts += torch.bincount(
            target_predictions,
            minlength=NUM_CLASSES,
        )

        target_features = student.encode(
            target_x
        )

        prototype_similarity = (
            cosine_to_predicted_prototype(
                target_features,
                target_predictions,
                prototypes,
            )
        )

        selected = select_pseudo_labels(
            target_confidence.cpu(),
            target_predictions.cpu(),
            prototype_similarity.detach().cpu(),
            threshold,
        )

        selected = selected.to(
            device
        )

        selected_count = int(
            selected.sum().item()
        )

        pseudo_selected += (
            selected_count
        )

        pseudo_total += (
            target_x.size(0)
        )

        if selected_count > 0:
            selected_x = target_x[
                selected
            ]

            selected_y = target_predictions[
                selected
            ]

            selected_confidence = (
                target_confidence[
                    selected
                ].detach()
            )

            selected_similarity = (
                prototype_similarity[
                    selected
                ].detach()
            )

            selected_counts += torch.bincount(
                selected_y.detach().cpu(),
                minlength=NUM_CLASSES,
            )

            pseudo_confidence_sum += (
                selected_confidence.sum().item()
            )

            prototype_similarity_sum += (
                selected_similarity.sum().item()
            )

            optimizer_adapter.zero_grad(
                set_to_none=True
            )

            selected_z = student.encode(
                selected_x
            )

            pl_logits1 = student.classifier1(
                selected_z
            )

            pl_logits2 = student.classifier2(
                selected_z
            )

            loss1 = F.cross_entropy(
                pl_logits1,
                selected_y,
                reduction="none",
            )

            loss2 = F.cross_entropy(
                pl_logits2,
                selected_y,
                reduction="none",
            )

            confidence_weighted_loss = (
                (
                    loss1
                    + loss2
                )
                * selected_confidence
            ).mean()

            prototype_loss = prototype_anchor_loss(
                selected_z,
                selected_y,
                prototypes,
            )

            target_loss = (
                PL_WEIGHT
                * confidence_weighted_loss
                + PROTOTYPE_WEIGHT
                * prototype_loss
            )

            target_loss.backward()

            optimizer_adapter.step()

            optimizer_classifier.zero_grad(
                set_to_none=True
            )

            with torch.no_grad():
                selected_z_classifier = (
                    student.encode(
                        selected_x
                    )
                )

            classifier_logits1 = (
                student.classifier1(
                    selected_z_classifier
                )
            )

            classifier_logits2 = (
                student.classifier2(
                    selected_z_classifier
                )
            )

            classifier_pl_loss = (
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

            classifier_pl_loss = (
                classifier_pl_loss
                * selected_confidence
            ).mean()

            (
                PL_WEIGHT
                * classifier_pl_loss
            ).backward()

            optimizer_classifier.step()

            update_ema(
                teacher,
                student,
                EMA_DECAY,
            )

            pseudo_loss_sum += (
                confidence_weighted_loss.item()
            )

            prototype_loss_sum += (
                prototype_loss.item()
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
        "prototype_loss":
            prototype_loss_sum
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
            pseudo_confidence_sum
            / max(
                pseudo_selected,
                1,
            ),
        "prototype_similarity":
            prototype_similarity_sum
            / max(
                pseudo_selected,
                1,
            ),
        "selected_counts":
            selected_counts.tolist(),
        "predicted_counts":
            predicted_counts.tolist(),
    }


def prototype_anchor_loss(
    features,
    labels,
    prototypes,
):
    logits = (
        F.normalize(
            features,
            dim=1,
        )
        @ F.normalize(
            prototypes,
            dim=1,
        ).t()
    ) / PROTOTYPE_THRESHOLD if False else (
        F.normalize(
            features,
            dim=1,
        )
        @ F.normalize(
            prototypes,
            dim=1,
        ).t()
    ) / 0.10

    return F.cross_entropy(
        logits,
        labels,
    )


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
        "VISDA-2017 MINIMAL SOURCE-PROTOTYPE-GATED MCD"
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
        f"pl_threshold={PL_THRESHOLD_START}"
        f"->{PL_THRESHOLD_END}"
    )

    print(
        f"global_pl_fraction={GLOBAL_PL_FRACTION}"
    )

    print(
        f"prototype_threshold={PROTOTYPE_THRESHOLD}"
    )

    print(
        f"pl_weight={PL_WEIGHT}"
    )

    print(
        f"prototype_weight={PROTOTYPE_WEIGHT}"
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

    if source_features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Source feature dimension assertion failed"
        )

    if target_features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Target feature dimension assertion failed"
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

    initial_student = evaluate(
        student,
        target_features,
        target_labels,
        device,
    )

    initial_teacher = evaluate(
        teacher,
        target_features,
        target_labels,
        device,
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

    print()
    print(
        "Computing source prototypes..."
    )

    source_prototypes = (
        compute_source_prototypes(
            student,
            source_features,
            source_labels,
            device,
        )
    )

    print(
        f"Prototype shape: "
        f"{tuple(source_prototypes.shape)}"
    )

    assert source_prototypes.shape == (
        NUM_CLASSES,
        HIDDEN_DIM,
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
            source_prototypes,
            epoch,
            device,
        )

        scheduler_adapter.step()
        scheduler_classifier.step()

        student_metrics = evaluate(
            student,
            target_features,
            target_labels,
            device,
        )

        teacher_metrics = evaluate(
            teacher,
            target_features,
            target_labels,
            device,
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
            "prototype_loss":
                metrics["prototype_loss"],
            "pseudo_selected":
                metrics["pseudo_selected"],
            "pseudo_total":
                metrics["pseudo_total"],
            "pseudo_coverage":
                metrics["pseudo_coverage"],
            "pseudo_confidence":
                metrics["pseudo_confidence"],
            "prototype_similarity":
                metrics["prototype_similarity"],
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
            "student_target_confidence":
                student_metrics[
                    "mean_confidence"
                ],
            "teacher_target_confidence":
                teacher_metrics[
                    "mean_confidence"
                ],
            "selected_counts":
                metrics[
                    "selected_counts"
                ],
            "predicted_counts":
                metrics[
                    "predicted_counts"
                ],
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
            f"ProtoSim "
            f"{metrics['prototype_similarity']:.4f} | "
            f"{epoch_seconds:.2f}s"
        )

        if epoch > WARMUP_EPOCHS:
            print(
                "Selected pseudo-label counts:"
            )

            for class_name, count in zip(
                CLASSES,
                metrics[
                    "selected_counts"
                ],
            ):
                print(
                    f"  {class_name:12s}: "
                    f"{count}"
                )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    final_student = evaluate(
        student,
        target_features,
        target_labels,
        device,
    )

    final_teacher = evaluate(
        teacher,
        target_features,
        target_labels,
        device,
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
    print("=" * 90)
    print(
        "FINAL RESULT"
    )
    print("=" * 90)

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
        "Final target prediction distribution:"
    )

    final_pred_counts = torch.bincount(
        torch.tensor(
            [
                int(
                    np.argmax(
                        np.array(
                            [
                                row[
                                    "predicted_counts"
                                ][class_id]
                                for row in history
                            ]
                        )
                    )
                )
                for class_id in range(
                    NUM_CLASSES
                )
            ],
            dtype=torch.long,
        ),
        minlength=NUM_CLASSES,
    )

    latest_distribution = history[-1][
        "predicted_counts"
    ]

    total_predictions = max(
        sum(latest_distribution),
        1,
    )

    for class_name, count in zip(
        CLASSES,
        latest_distribution,
    ):
        print(
            f"{class_name:12s}: "
            f"{count:6d} "
            f"({100.0 * count / total_predictions:6.2f}%)"
        )

    print()
    print(
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "minimal_prototype_gate_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "minimal_prototype_gate_seed42.json"
    )

    torch.save(
        {
            "student_state_dict":
                student.state_dict(),
            "teacher_state_dict":
                teacher.state_dict(),
            "source_prototypes":
                source_prototypes.cpu(),
            "seed":
                SEED,
            "classes":
                CLASSES,
            "input_dim":
                INPUT_DIM,
            "hidden_dim":
                HIDDEN_DIM,
            "num_classes":
                NUM_CLASSES,
            "pl_threshold_start":
                PL_THRESHOLD_START,
            "pl_threshold_end":
                PL_THRESHOLD_END,
            "global_pl_fraction":
                GLOBAL_PL_FRACTION,
            "prototype_threshold":
                PROTOTYPE_THRESHOLD,
            "pl_weight":
                PL_WEIGHT,
            "prototype_weight":
                PROTOTYPE_WEIGHT,
            "final_student":
                final_student,
            "final_teacher":
                final_teacher,
            "best_student_epoch":
                best_student[
                    "epoch"
                ],
            "best_student_mean_class":
                best_student[
                    "student_target_mean_class"
                ],
            "best_teacher_epoch":
                best_teacher[
                    "epoch"
                ],
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
            "visda_minimal_source_prototype_gated_mcd",
        "seed":
            SEED,
        "backbone":
            "ImageNet-pretrained ResNet-50 frozen",
        "input_dim":
            INPUT_DIM,
        "hidden_dim":
            HIDDEN_DIM,
        "target_labels_used_for_training":
            False,
        "target_labels_used_for_evaluation":
            True,
        "epochs":
            EPOCHS,
        "warmup_epochs":
            WARMUP_EPOCHS,
        "ema_decay":
            EMA_DECAY,
        "pl_threshold_start":
            PL_THRESHOLD_START,
        "pl_threshold_end":
            PL_THRESHOLD_END,
        "global_pl_fraction":
            GLOBAL_PL_FRACTION,
        "prototype_threshold":
            PROTOTYPE_THRESHOLD,
        "pl_weight":
            PL_WEIGHT,
        "prototype_weight":
            PROTOTYPE_WEIGHT,
        "final_student":
            final_student,
        "final_teacher":
            final_teacher,
        "best_student_epoch":
            best_student[
                "epoch"
            ],
        "best_student_mean_class":
            best_student[
                "student_target_mean_class"
            ],
        "best_teacher_epoch":
            best_teacher[
                "epoch"
            ],
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
    print("=" * 90)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 90)

    print(
        f"checkpoint={checkpoint_path}"
    )

    print(
        f"history={history_path}"
    )


if __name__ == "__main__":
    main()