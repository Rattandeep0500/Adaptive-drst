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
    "checkpoints/visda_geometry_gated_mcd"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

GEOMETRY_SIMILARITY_THRESHOLD = 0.60
GEOMETRY_MARGIN_THRESHOLD = 0.00

PL_WEIGHT = 0.20

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
            map_location="cpu"
        )

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
            )

        batch_features = payload[
            "features"
        ].float()

        batch_labels = payload[
            "labels"
        ].long()

        if batch_features.ndim != 2:
            raise RuntimeError(
                f"Invalid feature shape in {path}"
            )

        if batch_features.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features in {path}, "
                f"got {batch_features.shape[1]}"
            )

        if len(batch_features) != len(batch_labels):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
            )

        features.append(
            batch_features
        )

        labels.append(
            batch_labels
        )

    features = torch.cat(
        features,
        dim=0
    )

    labels = torch.cat(
        labels,
        dim=0
    )

    if features.ndim != 2:
        raise RuntimeError(
            f"Invalid final feature tensor {tuple(features.shape)}"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Invalid final feature dimension {features.shape[1]}"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Final feature/label count mismatch"
        )

    return features, labels


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM
            ),
            nn.BatchNorm1d(
                HIDDEN_DIM
            ),
            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(
                f"Adapter input must be 2D, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter input must be {INPUT_DIM}-D, "
                f"got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter output must be {HIDDEN_DIM}-D, "
                f"got {z.shape[1]}-D"
            )

        return z


class MCDModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.adapter = Adapter()

        self.classifier1 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES
        )

        self.classifier2 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES
        )

    def encode(self, x):
        return self.adapter(x)

    def forward(self, x):
        z = self.encode(x)

        logits1 = self.classifier1(z)
        logits2 = self.classifier2(z)

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
            dtype=model.classifier2.weight.dtype
        ).reshape_as(
            model.classifier2.weight
        )

        signs = torch.where(
            values.remainder(2) == 0,
            torch.ones_like(values),
            -torch.ones_like(values)
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
    student
):
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(),
        student.parameters()
    ):
        teacher_parameter.mul_(
            EMA_DECAY
        )

        teacher_parameter.add_(
            student_parameter,
            alpha=1.0 - EMA_DECAY
        )


def discrepancy(
    logits1,
    logits2
):
    p1 = F.softmax(
        logits1,
        dim=1
    )

    p2 = F.softmax(
        logits2,
        dim=1
    )

    return (
        p1 - p2
    ).abs().mean()


def threshold_for_epoch(epoch):
    if epoch <= WARMUP_EPOCHS:
        return float("inf")

    span = max(
        EPOCHS - WARMUP_EPOCHS,
        1
    )

    progress = (
        epoch - WARMUP_EPOCHS
    ) / span

    progress = min(
        max(progress, 0.0),
        1.0
    )

    return (
        PL_THRESHOLD_START
        + (
            PL_THRESHOLD_END
            - PL_THRESHOLD_START
        )
        * progress
    )


@torch.no_grad()
def teacher_predict(
    teacher,
    x
):
    if x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Teacher expected {INPUT_DIM}-D input, "
            f"got {x.shape[1]}-D"
        )

    logits1, logits2 = teacher(x)

    p1 = F.softmax(
        logits1,
        dim=1
    )

    p2 = F.softmax(
        logits2,
        dim=1
    )

    probabilities = (
        p1 + p2
    ) / 2.0

    confidence, predictions = (
        probabilities.max(
            dim=1
        )
    )

    return (
        probabilities,
        confidence,
        predictions
    )


@torch.no_grad()
def compute_source_prototypes(
    model,
    source_features,
    source_labels,
    device
):
    model.eval()

    prototype_sums = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64
    )

    prototype_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(
            source_features,
            source_labels
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    for source_x, source_y in loader:
        source_x = source_x.to(device)

        source_z = model.encode(
            source_x
        )

        if source_z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Invalid source adapted dimension "
                f"{source_z.shape[1]}"
            )

        source_z = F.normalize(
            source_z,
            dim=1
        )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                source_y == class_id
            )

            if mask.any():
                mask_device = mask.to(
                    device
                )

                class_z = source_z[
                    mask_device
                ]

                prototype_sums[
                    class_id
                ] += class_z.double().sum(
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
            as_tuple=False
        ).flatten().tolist()

        raise RuntimeError(
            f"Missing prototypes for classes {missing}"
        )

    prototypes = (
        prototype_sums
        / prototype_counts.double().unsqueeze(1)
    )

    prototypes = prototypes.float()

    prototypes = F.normalize(
        prototypes,
        dim=1
    )

    return prototypes.to(
        device
    )


@torch.no_grad()
def collect_final_predictions(
    model,
    target_features,
    device
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            target_features
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    all_predictions = []
    all_confidence = []
    all_disagreement = []

    for (target_x,) in loader:
        target_x = target_x.to(
            device
        )

        logits1, logits2 = model(
            target_x
        )

        p1 = F.softmax(
            logits1,
            dim=1
        )

        p2 = F.softmax(
            logits2,
            dim=1
        )

        probabilities = (
            p1 + p2
        ) / 2.0

        confidence, predictions = (
            probabilities.max(
                dim=1
            )
        )

        disagreement = (
            p1 - p2
        ).abs().mean(
            dim=1
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_confidence.append(
            confidence.cpu()
        )

        all_disagreement.append(
            disagreement.cpu()
        )

    return (
        torch.cat(
            all_predictions,
            dim=0
        ),
        torch.cat(
            all_confidence,
            dim=0
        ),
        torch.cat(
            all_disagreement,
            dim=0
        )
    )


def geometry_scores(
    target_z,
    predictions,
    prototypes
):
    z = F.normalize(
        target_z,
        dim=1
    )

    p = F.normalize(
        prototypes,
        dim=1
    )

    similarities = (
        z @ p.t()
    )

    row = torch.arange(
        len(z),
        device=z.device
    )

    predicted_similarity = (
        similarities[
            row,
            predictions
        ]
    )

    sorted_values, sorted_indices = (
        torch.sort(
            similarities,
            dim=1,
            descending=True
        )
    )

    second_similarity = (
        sorted_values[:, 1]
    )

    second_class = (
        sorted_indices[:, 1]
    )

    geometry_margin = (
        predicted_similarity
        - second_similarity
    )

    return (
        predicted_similarity,
        second_similarity,
        second_class,
        geometry_margin
    )


@torch.no_grad()
def evaluate(
    model,
    features,
    labels,
    device
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features,
            labels
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    total_correct = 0
    total_examples = 0
    confidence_sum = 0.0

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for x, y in loader:
        x = x.to(device)
        y_device = y.to(device)

        logits1, logits2 = model(
            x
        )

        probabilities = (
            F.softmax(
                logits1,
                dim=1
            )
            + F.softmax(
                logits2,
                dim=1
            )
        ) / 2.0

        confidence, predictions = (
            probabilities.max(
                dim=1
            )
        )

        total_correct += int(
            (
                predictions
                == y_device
            ).sum().item()
        )

        total_examples += y.size(0)

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
                mask_device = mask.to(
                    device
                )

                class_total[
                    class_id
                ] += int(
                    mask.sum().item()
                )

                class_correct[
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

    per_class = (
        100.0
        * class_correct.float()
        / class_total.clamp_min(1)
    )

    return {
        "overall_accuracy":
            100.0
            * total_correct
            / max(
                total_examples,
                1
            ),
        "mean_class_accuracy":
            float(
                per_class.mean().item()
            ),
        "per_class_accuracy":
            per_class.tolist(),
        "mean_confidence":
            confidence_sum
            / max(
                total_examples,
                1
            )
    }


def train_warmup_epoch(
    student,
    source_loader,
    target_loader,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    student.train()

    source_iterator = iter(
        source_loader
    )

    target_iterator = iter(
        target_loader
    )

    steps = len(
        source_loader
    )

    source_correct = 0
    source_total = 0

    source_loss_sum = 0.0
    discrepancy_sum = 0.0

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

        source_x = source_x.to(
            device
        )

        source_y = source_y.to(
            device
        )

        target_x = target_x.to(
            device
        )

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        source_logits1, source_logits2 = (
            student(source_x)
        )

        source_loss = (
            F.cross_entropy(
                source_logits1,
                source_y
            )
            + F.cross_entropy(
                source_logits2,
                source_y
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

        source_z = student.encode(
            source_x
        )

        target_z = student.encode(
            target_x
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
                source_y
            )
            + F.cross_entropy(
                source_logits2,
                source_y
            )
        )

        target_discrepancy = discrepancy(
            target_logits1,
            target_logits2
        )

        (
            source_classifier_loss
            - target_discrepancy
        ).backward()

        optimizer_classifier.step()

        discrepancy_sum += (
            target_discrepancy.item()
        )

        for parameter in student.classifier1.parameters():
            parameter.requires_grad_(False)

        for parameter in student.classifier2.parameters():
            parameter.requires_grad_(False)

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        target_z = student.encode(
            target_x
        )

        target_logits1 = student.classifier1(
            target_z
        )

        target_logits2 = student.classifier2(
            target_z
        )

        generator_discrepancy = discrepancy(
            target_logits1,
            target_logits2
        )

        generator_discrepancy.backward()

        optimizer_adapter.step()

        for parameter in student.classifier1.parameters():
            parameter.requires_grad_(True)

        for parameter in student.classifier2.parameters():
            parameter.requires_grad_(True)

    return {
        "source_accuracy":
            100.0
            * source_correct
            / max(
                source_total,
                1
            ),
        "source_loss":
            source_loss_sum
            / max(
                steps,
                1
            ),
        "discrepancy":
            discrepancy_sum
            / max(
                steps,
                1
            )
    }


def train_adaptation_epoch(
    student,
    teacher,
    source_loader,
    target_loader,
    optimizer_adapter,
    optimizer_classifier,
    prototypes,
    epoch,
    device
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

    pseudo_selected = 0
    pseudo_total = 0

    pseudo_confidence_sum = 0.0
    pseudo_similarity_sum = 0.0
    pseudo_margin_sum = 0.0

    confidence_candidates = 0
    geometry_rejections = 0

    selected_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
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

        source_x = source_x.to(
            device
        )

        source_y = source_y.to(
            device
        )

        target_x = target_x.to(
            device
        )

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        source_logits1, source_logits2 = (
            student(source_x)
        )

        source_loss = (
            F.cross_entropy(
                source_logits1,
                source_y
            )
            + F.cross_entropy(
                source_logits2,
                source_y
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

        source_z = student.encode(
            source_x
        )

        target_z = student.encode(
            target_x
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
                source_y
            )
            + F.cross_entropy(
                source_logits2,
                source_y
            )
        )

        target_discrepancy = discrepancy(
            target_logits1,
            target_logits2
        )

        (
            source_classifier_loss
            - target_discrepancy
        ).backward()

        optimizer_classifier.step()

        discrepancy_sum += (
            target_discrepancy.item()
        )

        for parameter in student.classifier1.parameters():
            parameter.requires_grad_(False)

        for parameter in student.classifier2.parameters():
            parameter.requires_grad_(False)

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        target_z = student.encode(
            target_x
        )

        target_logits1 = student.classifier1(
            target_z
        )

        target_logits2 = student.classifier2(
            target_z
        )

        generator_discrepancy = discrepancy(
            target_logits1,
            target_logits2
        )

        generator_discrepancy.backward()

        optimizer_adapter.step()

        for parameter in student.classifier1.parameters():
            parameter.requires_grad_(True)

        for parameter in student.classifier2.parameters():
            parameter.requires_grad_(True)

        update_ema(
            teacher,
            student
        )

        with torch.no_grad():
            (
                _,
                teacher_confidence,
                teacher_predictions
            ) = teacher_predict(
                teacher,
                target_x
            )

        with torch.no_grad():
            geometry_z = student.encode(
                target_x
            )

            (
                predicted_similarity,
                _,
                _,
                geometry_margin
            ) = geometry_scores(
                geometry_z,
                teacher_predictions,
                prototypes
            )

        confidence_mask = (
            teacher_confidence
            >= threshold
        )

        confidence_candidates += int(
            confidence_mask.sum().item()
        )

        geometry_mask = (
            (
                predicted_similarity
                >= GEOMETRY_SIMILARITY_THRESHOLD
            )
            & (
                geometry_margin
                >= GEOMETRY_MARGIN_THRESHOLD
            )
        )

        geometry_rejections += int(
            (
                confidence_mask
                & ~geometry_mask
            ).sum().item()
        )

        eligible = (
            confidence_mask
            & geometry_mask
        )

        eligible_indices = torch.nonzero(
            eligible,
            as_tuple=False
        ).flatten()

        if len(eligible_indices) > 0:
            max_selected = max(
                1,
                int(
                    target_x.size(0)
                    * GLOBAL_PL_FRACTION
                )
            )

            if len(eligible_indices) > max_selected:
                ordering = torch.argsort(
                    teacher_confidence[
                        eligible_indices
                    ],
                    descending=True
                )

                eligible_indices = (
                    eligible_indices[
                        ordering[
                            :max_selected
                        ]
                    ]
                )

        selected = torch.zeros(
            target_x.size(0),
            dtype=torch.bool,
            device=device
        )

        if len(eligible_indices) > 0:
            selected[
                eligible_indices
            ] = True

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

            selected_y = teacher_predictions[
                selected
            ]

            selected_confidence = (
                teacher_confidence[
                    selected
                ].detach()
            )

            selected_similarity = (
                predicted_similarity[
                    selected
                ].detach()
            )

            selected_margin = (
                geometry_margin[
                    selected
                ].detach()
            )

            selected_counts += torch.bincount(
                selected_y.cpu(),
                minlength=NUM_CLASSES
            )

            pseudo_confidence_sum += (
                selected_confidence.sum().item()
            )

            pseudo_similarity_sum += (
                selected_similarity.sum().item()
            )

            pseudo_margin_sum += (
                selected_margin.sum().item()
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
                reduction="none"
            )

            loss2 = F.cross_entropy(
                pl_logits2,
                selected_y,
                reduction="none"
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

            classifier_pseudo_loss = (
                F.cross_entropy(
                    classifier_logits1,
                    selected_y,
                    reduction="none"
                )
                + F.cross_entropy(
                    classifier_logits2,
                    selected_y,
                    reduction="none"
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
                student
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
                1
            ),
        "source_loss":
            source_loss_sum
            / max(
                steps,
                1
            ),
        "discrepancy":
            discrepancy_sum
            / max(
                steps,
                1
            ),
        "pseudo_loss":
            pseudo_loss_sum
            / max(
                steps,
                1
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
                1
            ),
        "pseudo_confidence":
            pseudo_confidence_sum
            / max(
                pseudo_selected,
                1
            ),
        "pseudo_similarity":
            pseudo_similarity_sum
            / max(
                pseudo_selected,
                1
            ),
        "pseudo_margin":
            pseudo_margin_sum
            / max(
                pseudo_selected,
                1
            ),
        "confidence_candidates":
            confidence_candidates,
        "geometry_rejections":
            geometry_rejections,
        "selected_counts":
            selected_counts.tolist()
    }


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
        "VISDA-2017 WARMED-UP SOURCE-PROTOTYPE GATED MCD"
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
        f"geometry_similarity_threshold="
        f"{GEOMETRY_SIMILARITY_THRESHOLD}"
    )

    print(
        f"geometry_margin_threshold="
        f"{GEOMETRY_MARGIN_THRESHOLD}"
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
            "Source dimension assertion failed"
        )

    if target_features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Target dimension assertion failed"
        )

    source_loader = DataLoader(
        TensorDataset(
            source_features,
            source_labels
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    target_loader = DataLoader(
        TensorDataset(
            target_features
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    student = MCDModel().to(
        device
    )

    initialize_heads(
        student
    )

    optimizer_adapter = torch.optim.SGD(
        student.adapter.parameters(),
        lr=LR_ADAPTER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
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
        weight_decay=WEIGHT_DECAY
    )

    warmup_scheduler_adapter = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_adapter,
            T_max=EPOCHS
        )
    )

    warmup_scheduler_classifier = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_classifier,
            T_max=EPOCHS
        )
    )

    print()
    print(
        "INITIAL TARGET"
    )

    initial_metrics = evaluate(
        student,
        target_features,
        target_labels,
        device
    )

    print(
        f"Initial mean-class: "
        f"{initial_metrics['mean_class_accuracy']:.2f}%"
    )

    history = []

    start_time = time.perf_counter()

    print()
    print("=" * 90)
    print(
        "MCD WARM-UP"
    )
    print("=" * 90)

    for epoch in range(
        1,
        WARMUP_EPOCHS + 1
    ):
        epoch_start = time.perf_counter()

        metrics = train_warmup_epoch(
            student,
            source_loader,
            target_loader,
            optimizer_adapter,
            optimizer_classifier,
            device
        )

        warmup_scheduler_adapter.step()
        warmup_scheduler_classifier.step()

        warmup_eval = evaluate(
            student,
            target_features,
            target_labels,
            device
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        history.append(
            {
                "epoch":
                    epoch,
                "phase":
                    "warmup",
                "threshold":
                    None,
                "source_accuracy":
                    metrics[
                        "source_accuracy"
                    ],
                "source_loss":
                    metrics[
                        "source_loss"
                    ],
                "mcd_discrepancy":
                    metrics[
                        "discrepancy"
                    ],
                "pseudo_coverage":
                    0.0,
                "pseudo_confidence":
                    0.0,
                "pseudo_similarity":
                    0.0,
                "pseudo_margin":
                    0.0,
                "geometry_rejection_rate":
                    0.0,
                "student_target_overall":
                    warmup_eval[
                        "overall_accuracy"
                    ],
                "student_target_mean_class":
                    warmup_eval[
                        "mean_class_accuracy"
                    ],
                "teacher_target_overall":
                    None,
                "teacher_target_mean_class":
                    None,
                "selected_counts":
                    [0] * NUM_CLASSES,
                "epoch_seconds":
                    epoch_seconds
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Warmup | "
            f"Source "
            f"{metrics['source_accuracy']:.2f}% | "
            f"Student "
            f"{warmup_eval['mean_class_accuracy']:.2f}% | "
            f"Disc "
            f"{metrics['discrepancy']:.6f} | "
            f"{epoch_seconds:.2f}s"
        )

    print()
    print("=" * 90)
    print(
        "FREEZING SOURCE PROTOTYPES AFTER WARM-UP"
    )
    print("=" * 90)

    source_prototypes = (
        compute_source_prototypes(
            student,
            source_features,
            source_labels,
            device
        )
    )

    print(
        f"Prototype shape: "
        f"{tuple(source_prototypes.shape)}"
    )

    prototype_self_similarity = (
        source_prototypes
        @ source_prototypes.t()
    )

    diagonal = (
        prototype_self_similarity.diag()
    )

    print(
        f"Prototype self-similarity mean: "
        f"{diagonal.mean().item():.6f}"
    )

    teacher = clone_model(
        student
    ).to(device)

    adaptation_scheduler_adapter = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_adapter,
            T_max=EPOCHS - WARMUP_EPOCHS
        )
    )

    adaptation_scheduler_classifier = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_classifier,
            T_max=EPOCHS - WARMUP_EPOCHS
        )
    )

    print()
    print("=" * 90)
    print(
        "GEOMETRY-GATED ADAPTATION"
    )
    print("=" * 90)

    for epoch in range(
        WARMUP_EPOCHS + 1,
        EPOCHS + 1
    ):
        epoch_start = time.perf_counter()

        metrics = train_adaptation_epoch(
            student,
            teacher,
            source_loader,
            target_loader,
            optimizer_adapter,
            optimizer_classifier,
            source_prototypes,
            epoch,
            device
        )

        adaptation_scheduler_adapter.step()
        adaptation_scheduler_classifier.step()

        student_metrics = evaluate(
            student,
            target_features,
            target_labels,
            device
        )

        teacher_metrics = evaluate(
            teacher,
            target_features,
            target_labels,
            device
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        if (
            metrics["confidence_candidates"]
            > 0
        ):
            rejection_rate = (
                metrics[
                    "geometry_rejections"
                ]
                / metrics[
                    "confidence_candidates"
                ]
            )
        else:
            rejection_rate = 0.0

        record = {
            "epoch":
                epoch,
            "phase":
                "adaptation",
            "threshold":
                metrics[
                    "threshold"
                ],
            "source_accuracy":
                metrics[
                    "source_accuracy"
                ],
            "source_loss":
                metrics[
                    "source_loss"
                ],
            "mcd_discrepancy":
                metrics[
                    "discrepancy"
                ],
            "pseudo_loss":
                metrics[
                    "pseudo_loss"
                ],
            "pseudo_selected":
                metrics[
                    "pseudo_selected"
                ],
            "pseudo_total":
                metrics[
                    "pseudo_total"
                ],
            "pseudo_coverage":
                metrics[
                    "pseudo_coverage"
                ],
            "pseudo_confidence":
                metrics[
                    "pseudo_confidence"
                ],
            "pseudo_similarity":
                metrics[
                    "pseudo_similarity"
                ],
            "pseudo_margin":
                metrics[
                    "pseudo_margin"
                ],
            "confidence_candidates":
                metrics[
                    "confidence_candidates"
                ],
            "geometry_rejections":
                metrics[
                    "geometry_rejections"
                ],
            "geometry_rejection_rate":
                rejection_rate,
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
            "epoch_seconds":
                epoch_seconds
        }

        history.append(
            record
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Thr {metrics['threshold']:.3f} | "
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
            f"Sim "
            f"{metrics['pseudo_similarity']:.4f} | "
            f"Marg "
            f"{metrics['pseudo_margin']:.4f} | "
            f"Reject "
            f"{100.0 * rejection_rate:.2f}% | "
            f"{epoch_seconds:.2f}s"
        )

        print(
            "Selected pseudo-label counts:"
        )

        for class_name, count in zip(
            CLASSES,
            metrics[
                "selected_counts"
            ]
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
        device
    )

    final_teacher = evaluate(
        teacher,
        target_features,
        target_labels,
        device
    )

    best_student = max(
        history,
        key=lambda row:
            row[
                "student_target_mean_class"
            ]
    )

    best_teacher_records = [
        row
        for row in history
        if row[
            "teacher_target_mean_class"
        ] is not None
    ]

    best_teacher = max(
        best_teacher_records,
        key=lambda row:
            row[
                "teacher_target_mean_class"
            ]
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
        ]
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
        ]
    ):
        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}%"
        )

    print()
    print(
        "Final target prediction distribution:"
    )

    (
        final_predictions,
        final_confidence,
        final_disagreement
    ) = collect_final_predictions(
        student,
        target_features,
        device
    )

    final_prediction_counts = torch.bincount(
        final_predictions,
        minlength=NUM_CLASSES
    )

    for class_name, count in zip(
        CLASSES,
        final_prediction_counts.tolist()
    ):
        fraction = (
            count
            / max(
                len(final_predictions),
                1
            )
        )

        print(
            f"{class_name:12s}: "
            f"{count:6d} "
            f"({100.0 * fraction:6.2f}%)"
        )

    print()
    print(
        f"Final mean confidence: "
        f"{final_confidence.mean().item():.6f}"
    )

    print(
        f"Final mean disagreement: "
        f"{final_disagreement.mean().item():.6f}"
    )

    print()
    print(
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "geometry_gated_mcd_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "geometry_gated_mcd_seed42.json"
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
            "warmup_epochs":
                WARMUP_EPOCHS,
            "pl_threshold_start":
                PL_THRESHOLD_START,
            "pl_threshold_end":
                PL_THRESHOLD_END,
            "global_pl_fraction":
                GLOBAL_PL_FRACTION,
            "geometry_similarity_threshold":
                GEOMETRY_SIMILARITY_THRESHOLD,
            "geometry_margin_threshold":
                GEOMETRY_MARGIN_THRESHOLD,
            "pl_weight":
                PL_WEIGHT,
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
            "final_prediction_counts":
                final_prediction_counts.tolist(),
            "history":
                history
        },
        checkpoint_path
    )

    report = {
        "experiment":
            "visda_warmed_up_source_prototype_gated_mcd",
        "seed":
            SEED,
        "device":
            str(device),
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
        "geometry_similarity_threshold":
            GEOMETRY_SIMILARITY_THRESHOLD,
        "geometry_margin_threshold":
            GEOMETRY_MARGIN_THRESHOLD,
        "pl_weight":
            PL_WEIGHT,
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
        "final_prediction_counts":
            final_prediction_counts.tolist(),
        "final_mean_confidence":
            float(
                final_confidence.mean().item()
            ),
        "final_mean_disagreement":
            float(
                final_disagreement.mean().item()
            ),
        "history":
            history
    }

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2
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