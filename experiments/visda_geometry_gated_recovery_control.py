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

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/geometry_gated_mcd_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_geometry_gated_recovery_control"
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

EPOCHS = 10

LR_ADAPTER = 0.0005
LR_CLASSIFIER = 0.005

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

EMA_DECAY = 0.97

PL_THRESHOLD = 0.90
GLOBAL_PL_FRACTION = 0.25

GEOMETRY_SIMILARITY_THRESHOLD = 0.60
GEOMETRY_MARGIN_THRESHOLD = 0.00

PL_WEIGHT = 0.20

STARVATION_FRACTION = 0.01
STARVATION_PATIENCE = 2

RECOVERY_MAX_PER_CLASS_PER_EPOCH = 20
RECOVERY_CONFIDENCE_MIN = 0.50
RECOVERY_SIMILARITY_MIN = 0.40
RECOVERY_WEIGHT = 0.02

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

        x = payload["features"].float()
        y = payload["labels"].long()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature shape in {path}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, got {x.shape[1]}"
            )

        if len(x) != len(y):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
            )

        features.append(x)
        labels.append(y)

    features = torch.cat(
        features,
        dim=0
    )

    labels = torch.cat(
        labels,
        dim=0
    )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Final feature dimension mismatch: {features.shape[1]}"
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
                f"Adapter expected 2D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expected {INPUT_DIM}-D input, got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter output must be {HIDDEN_DIM}-D, got {z.shape[1]}-D"
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

        return (
            self.classifier1(z),
            self.classifier2(z)
        )


def normalize_state_dict(state_dict):
    result = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True

        while changed:
            changed = False

            for prefix in (
                "module.",
                "model.",
                "student.",
                "teacher."
            ):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                    break

        result[new_key] = value

    return result


def load_base_checkpoint(
    student,
    teacher
):
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {BASE_CHECKPOINT}"
        )

    payload = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
    )

    if "student_state_dict" not in payload:
        raise RuntimeError(
            "student_state_dict missing"
        )

    if "teacher_state_dict" not in payload:
        raise RuntimeError(
            "teacher_state_dict missing"
        )

    student.load_state_dict(
        normalize_state_dict(
            payload["student_state_dict"]
        ),
        strict=True
    )

    teacher.load_state_dict(
        normalize_state_dict(
            payload["teacher_state_dict"]
        ),
        strict=True
    )

    if "source_prototypes" not in payload:
        raise RuntimeError(
            "source_prototypes missing"
        )

    prototypes = payload[
        "source_prototypes"
    ].float()

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Invalid prototype shape {tuple(prototypes.shape)}"
        )

    return payload, prototypes


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


@torch.no_grad()
def teacher_predict(
    teacher,
    x
):
    teacher.eval()

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

    confidence, predictions = probabilities.max(
        dim=1
    )

    disagreement = (
        p1 - p2
    ).abs().mean(
        dim=1
    )

    return (
        probabilities,
        confidence,
        predictions,
        disagreement
    )


def geometry_scores(
    target_z,
    predictions,
    prototypes
):
    target_z = F.normalize(
        target_z,
        dim=1
    )

    prototypes = F.normalize(
        prototypes,
        dim=1
    )

    similarities = (
        target_z
        @ prototypes.t()
    )

    row = torch.arange(
        len(target_z),
        device=target_z.device
    )

    predicted_similarity = (
        similarities[
            row,
            predictions
        ]
    )

    sorted_values, sorted_indices = torch.sort(
        similarities,
        dim=1,
        descending=True
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
        similarities,
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

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    confidence_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y_device = y.to(device)

        logits1, logits2 = model(x)

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

        confidence, predictions = probabilities.max(
            dim=1
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


@torch.no_grad()
def collect_target_state(
    student,
    teacher,
    target_features,
    prototypes,
    device
):
    student.eval()
    teacher.eval()

    loader = DataLoader(
        TensorDataset(
            target_features
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    all_z = []
    all_probabilities = []
    all_confidence = []
    all_predictions = []
    all_disagreement = []

    for (x,) in loader:
        x = x.to(device)

        z = student.encode(
            x
        )

        (
            probabilities,
            confidence,
            predictions,
            disagreement
        ) = teacher_predict(
            teacher,
            x
        )

        all_z.append(
            z.cpu()
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        all_confidence.append(
            confidence.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_disagreement.append(
            disagreement.cpu()
        )

    z = torch.cat(
        all_z,
        dim=0
    )

    probabilities = torch.cat(
        all_probabilities,
        dim=0
    )

    confidence = torch.cat(
        all_confidence,
        dim=0
    )

    predictions = torch.cat(
        all_predictions,
        dim=0
    )

    disagreement = torch.cat(
        all_disagreement,
        dim=0
    )

    similarities = (
        F.normalize(
            z,
            dim=1
        )
        @ F.normalize(
            prototypes,
            dim=1
        ).t()
    )

    row = torch.arange(
        len(z)
    )

    predicted_similarity = (
        similarities[
            row,
            predictions
        ]
    )

    sorted_values, sorted_indices = torch.sort(
        similarities,
        dim=1,
        descending=True
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

    return {
        "z":
            z,
        "probabilities":
            probabilities,
        "confidence":
            confidence,
        "predictions":
            predictions,
        "disagreement":
            disagreement,
        "similarities":
            similarities,
        "predicted_similarity":
            predicted_similarity,
        "second_similarity":
            second_similarity,
        "second_class":
            second_class,
        "geometry_margin":
            geometry_margin
    }


def build_trusted_mask(
    state
):
    confidence_mask = (
        state["confidence"]
        >= PL_THRESHOLD
    )

    similarity_mask = (
        state["predicted_similarity"]
        >= GEOMETRY_SIMILARITY_THRESHOLD
    )

    margin_mask = (
        state["geometry_margin"]
        >= GEOMETRY_MARGIN_THRESHOLD
    )

    return (
        confidence_mask
        & similarity_mask
        & margin_mask
    )


def compute_starved_classes(
    trusted_predictions
):
    counts = torch.bincount(
        trusted_predictions,
        minlength=NUM_CLASSES
    )

    total = max(
        int(counts.sum().item()),
        1
    )

    fractions = (
        counts.float()
        / total
    )

    starved = []

    for class_id in range(
        NUM_CLASSES
    ):
        if (
            fractions[class_id].item()
            < STARVATION_FRACTION
        ):
            starved.append(
                class_id
            )

    return (
        counts,
        fractions,
        starved
    )


def select_trusted_batch(
    state
):
    trusted = build_trusted_mask(
        state
    )

    indices = torch.nonzero(
        trusted,
        as_tuple=False
    ).flatten()

    if len(indices) == 0:
        return trusted

    max_batch = max(
        1,
        int(
            len(indices)
            and len(indices)
            * 0.25
        )
    )

    if len(indices) > max_batch:
        order = torch.argsort(
            state["confidence"][
                indices
            ],
            descending=True
        )

        indices = indices[
            order[:max_batch]
        ]

    selected = torch.zeros_like(
        trusted
    )

    selected[
        indices
    ] = True

    return selected


def global_trusted_selection(
    state
):
    trusted = build_trusted_mask(
        state
    )

    indices = torch.nonzero(
        trusted,
        as_tuple=False
    ).flatten()

    if len(indices) == 0:
        return torch.zeros_like(
            trusted
        )

    limit = max(
        1,
        int(
            len(trusted)
            * GLOBAL_PL_FRACTION
        )
    )

    if len(indices) > limit:
        order = torch.argsort(
            state["confidence"][
                indices
            ],
            descending=True
        )

        indices = indices[
            order[:limit]
        ]

    selected = torch.zeros_like(
        trusted
    )

    selected[
        indices
    ] = True

    return selected


def select_global_recovery(
    state,
    starved_classes,
    recovery_used,
):
    selected = []

    if not starved_classes:
        return selected

    similarities = state[
        "similarities"
    ]

    predictions = state[
        "predictions"
    ]

    confidence = state[
        "confidence"
    ]

    trusted = build_trusted_mask(
        state
    )

    for class_id in starved_classes:
        remaining = (
            RECOVERY_MAX_PER_CLASS_PER_EPOCH
            - recovery_used[
                class_id
            ]
        )

        if remaining <= 0:
            continue

        class_similarity = (
            similarities[
                :,
                class_id
            ]
        )

        competitor_mask = torch.ones(
            NUM_CLASSES,
            dtype=torch.bool
        )

        competitor_mask[
            class_id
        ] = False

        competitor_similarity = (
            similarities[
                :,
                competitor_mask
            ].max(
                dim=1
            ).values
        )

        recovery_score = (
            class_similarity
            - competitor_similarity
        )

        candidate_mask = (
            ~trusted
        ) & (
            predictions
            != class_id
        ) & (
            confidence
            >= RECOVERY_CONFIDENCE_MIN
        ) & (
            class_similarity
            >= RECOVERY_SIMILARITY_MIN
        )

        indices = torch.nonzero(
            candidate_mask,
            as_tuple=False
        ).flatten()

        if len(indices) == 0:
            continue

        order = torch.argsort(
            recovery_score[
                indices
            ],
            descending=True
        )

        selected_indices = indices[
            order[
                :remaining
            ]
        ]

        for index in selected_indices.tolist():
            selected.append(
                {
                    "index":
                        index,
                    "target_class":
                        class_id,
                    "score":
                        float(
                            recovery_score[
                                index
                            ].item()
                        ),
                    "similarity":
                        float(
                            class_similarity[
                                index
                            ].item()
                        ),
                    "confidence":
                        float(
                            confidence[
                                index
                            ].item()
                        )
                }
            )

    return selected


def train_target_step(
    student,
    target_x,
    target_y,
    weight,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    if len(target_x) == 0:
        return 0.0

    target_x = target_x.to(device)
    target_y = target_y.to(device)

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    z = student.encode(
        target_x
    )

    logits1 = student.classifier1(
        z
    )

    logits2 = student.classifier2(
        z
    )

    loss1 = F.cross_entropy(
        logits1,
        target_y,
        reduction="none"
    )

    loss2 = F.cross_entropy(
        logits2,
        target_y,
        reduction="none"
    )

    loss = (
        (
            loss1
            + loss2
        )
        * weight
    ).mean()

    loss.backward()

    optimizer_adapter.step()

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():
        z_classifier = student.encode(
            target_x
        )

    logits1 = student.classifier1(
        z_classifier
    )

    logits2 = student.classifier2(
        z_classifier
    )

    classifier_loss = (
        F.cross_entropy(
            logits1,
            target_y,
            reduction="none"
        )
        + F.cross_entropy(
            logits2,
            target_y,
            reduction="none"
        )
    )

    classifier_loss = (
        classifier_loss
        * weight
    ).mean()

    classifier_loss.backward()

    optimizer_classifier.step()

    return float(
        loss.item()
    )


def train_epoch(
    student,
    teacher,
    source_loader,
    target_loader,
    optimizer_adapter,
    optimizer_classifier,
    prototypes,
    target_features_all,
    target_labels_all,
    recovery_used,
    starvation_state,
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

    source_correct = 0
    source_total = 0
    source_loss_sum = 0.0
    discrepancy_sum = 0.0
    pseudo_loss_sum = 0.0

    pseudo_selected = 0

    batch_recovery_counts = torch.zeros(
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

        source_logits1, source_logits2 = student(
            source_x
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
                confidence,
                predictions,
                _
            ) = teacher_predict(
                teacher,
                target_x
            )

            target_z_geometry = (
                student.encode(
                    target_x
                )
            )

            (
                _,
                predicted_similarity,
                _,
                _,
                geometry_margin
            ) = geometry_scores(
                target_z_geometry,
                predictions,
                prototypes
            )

            confidence_mask = (
                confidence
                >= PL_THRESHOLD
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

            trusted = (
                confidence_mask
                & geometry_mask
            )

        confidence_candidates = (
            confidence_mask
        )

        geometry_rejections = (
            confidence_candidates
            & ~geometry_mask
        )

        selected = trusted.clone()

        selected_indices = torch.nonzero(
            selected,
            as_tuple=False
        ).flatten()

        if len(selected_indices) > int(
            target_x.size(0)
            * GLOBAL_PL_FRACTION
        ):
            limit = max(
                1,
                int(
                    target_x.size(0)
                    * GLOBAL_PL_FRACTION
                )
            )

            order = torch.argsort(
                confidence[
                    selected_indices
                ],
                descending=True
            )

            selected_indices = selected_indices[
                order[
                    :limit
                ]
            ]

            selected = torch.zeros_like(
                trusted
            )

            selected[
                selected_indices
            ] = True

        pseudo_count = int(
            selected.sum().item()
        )

        pseudo_selected += (
            pseudo_count
        )

        if pseudo_count > 0:
            selected_x = target_x[
                selected
            ]

            selected_y = predictions[
                selected
            ]

            selected_weight = (
                confidence[
                    selected
                ].detach()
            )

            pseudo_loss_value = train_target_step(
                student,
                selected_x,
                selected_y,
                selected_weight,
                optimizer_adapter,
                optimizer_classifier,
                device
            )

            pseudo_loss_sum += (
                pseudo_loss_value
            )

            selected_predictions = (
                predictions[
                    selected
                ].cpu()
            )

            batch_recovery_counts += torch.bincount(
                selected_predictions,
                minlength=NUM_CLASSES
            )

    source_accuracy = (
        100.0
        * source_correct
        / max(
            source_total,
            1
        )
    )

    return {
        "source_accuracy":
            source_accuracy,
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
        "geometry_rejections":
            int(
                geometry_rejections.sum().item()
            ),
        "confidence_candidates":
            int(
                confidence_candidates.sum().item()
            )
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
        "VISDA-2017 GEOMETRY-GATED MCD + CONTROLLED CLASS RECOVERY"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"base_checkpoint={BASE_CHECKPOINT}"
    )

    print(
        f"epochs={EPOCHS}"
    )

    print(
        f"pl_threshold={PL_THRESHOLD}"
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

    print(
        f"recovery_max_per_class_per_epoch="
        f"{RECOVERY_MAX_PER_CLASS_PER_EPOCH}"
    )

    print(
        f"recovery_weight={RECOVERY_WEIGHT}"
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

    teacher = MCDModel().to(
        device
    )

    payload, source_prototypes = load_base_checkpoint(
        student,
        teacher
    )

    source_prototypes = (
        source_prototypes.to(
            device
        )
    )

    print()
    print(
        "Base checkpoint loaded."
    )

    base_metrics = evaluate(
        student,
        target_features,
        target_labels,
        device
    )

    print()
    print(
        "BASE CHECKPOINT"
    )

    print(
        f"Overall: "
        f"{base_metrics['overall_accuracy']:.2f}%"
    )

    print(
        f"Mean-class: "
        f"{base_metrics['mean_class_accuracy']:.2f}%"
    )

    print()
    print(
        "Building optimizers..."
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

    scheduler_adapter = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_adapter,
            T_max=EPOCHS
        )
    )

    scheduler_classifier = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_classifier,
            T_max=EPOCHS
        )
    )

    recovery_used = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    starvation_state = [
        0
        for _ in range(
            NUM_CLASSES
        )
    ]

    history = []

    best_student = (
        base_metrics[
            "mean_class_accuracy"
        ]
    )

    best_teacher = (
        base_metrics[
            "mean_class_accuracy"
        ]
    )

    best_student_epoch = 0
    best_teacher_epoch = 0

    best_student_state = {
        key:
            value.detach().cpu().clone()
        for key, value in student.state_dict().items()
    }

    best_teacher_state = {
        key:
            value.detach().cpu().clone()
        for key, value in teacher.state_dict().items()
    }

    start_time = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        epoch_start = time.perf_counter()

        if epoch > 1:
            recovery_used.zero_()

        state = collect_target_state(
            student,
            teacher,
            target_features,
            source_prototypes,
            device
        )

        initial_trusted = build_trusted_mask(
            state
        )

        trusted_predictions = state[
            "predictions"
        ][
            initial_trusted
        ]

        trusted_counts, trusted_fractions, candidate_starved = (
            compute_starved_classes(
                trusted_predictions
            )
        )

        for class_id in range(
            NUM_CLASSES
        ):
            if class_id in candidate_starved:
                starvation_state[
                    class_id
                ] += 1
            else:
                starvation_state[
                    class_id
                ] = 0

        active_starved_classes = [
            class_id
            for class_id in range(
                NUM_CLASSES
            )
            if starvation_state[
                class_id
            ] >= STARVATION_PATIENCE
        ]

        recovery_candidates = select_global_recovery(
            state,
            active_starved_classes,
            recovery_used
        )

        recovery_epoch_counts = torch.zeros(
            NUM_CLASSES,
            dtype=torch.long
        )

        recovery_precision_numerator = torch.zeros(
            NUM_CLASSES,
            dtype=torch.long
        )

        recovery_precision_denominator = torch.zeros(
            NUM_CLASSES,
            dtype=torch.long
        )

        if recovery_candidates:
            optimizer_adapter.zero_grad(
                set_to_none=True
            )

            recovery_indices = [
                item["index"]
                for item in recovery_candidates
            ]

            recovery_labels = torch.tensor(
                [
                    item["target_class"]
                    for item in recovery_candidates
                ],
                dtype=torch.long,
                device=device
            )

            recovery_x = target_features[
                recovery_indices
            ].to(device)

            recovery_z = student.encode(
                recovery_x
            )

            recovery_logits1 = student.classifier1(
                recovery_z
            )

            recovery_logits2 = student.classifier2(
                recovery_z
            )

            recovery_scores = torch.tensor(
                [
                    max(
                        item["score"],
                        0.05
                    )
                    for item in recovery_candidates
                ],
                dtype=torch.float32,
                device=device
            )

            recovery_weights = (
                0.5
                + recovery_scores
            )

            recovery_loss = (
                (
                    F.cross_entropy(
                        recovery_logits1,
                        recovery_labels,
                        reduction="none"
                    )
                    + F.cross_entropy(
                        recovery_logits2,
                        recovery_labels,
                        reduction="none"
                    )
                )
                * recovery_weights
            ).mean()

            (
                RECOVERY_WEIGHT
                * recovery_loss
            ).backward()

            optimizer_adapter.step()

            optimizer_classifier.zero_grad(
                set_to_none=True
            )

            with torch.no_grad():
                recovery_z_classifier = (
                    student.encode(
                        recovery_x
                    )
                )

            recovery_logits1 = student.classifier1(
                recovery_z_classifier
            )

            recovery_logits2 = student.classifier2(
                recovery_z_classifier
            )

            recovery_classifier_loss = (
                F.cross_entropy(
                    recovery_logits1,
                    recovery_labels,
                    reduction="none"
                )
                + F.cross_entropy(
                    recovery_logits2,
                    recovery_labels,
                    reduction="none"
                )
            )

            recovery_classifier_loss = (
                recovery_classifier_loss
                * recovery_weights
            ).mean()

            (
                RECOVERY_WEIGHT
                * recovery_classifier_loss
            ).backward()

            optimizer_classifier.step()

            update_ema(
                teacher,
                student
            )

            for candidate in recovery_candidates:
                class_id = candidate[
                    "target_class"
                ]

                recovery_used[
                    class_id
                ] += 1

                recovery_epoch_counts[
                    class_id
                ] += 1

                true_label = int(
                    target_labels[
                        candidate["index"]
                    ].item()
                )

                recovery_precision_denominator[
                    class_id
                ] += 1

                if true_label == class_id:
                    recovery_precision_numerator[
                        class_id
                    ] += 1

        metrics = train_epoch(
            student,
            teacher,
            source_loader,
            target_loader,
            optimizer_adapter,
            optimizer_classifier,
            source_prototypes,
            target_features,
            target_labels,
            recovery_used,
            starvation_state,
            epoch,
            device
        )

        scheduler_adapter.step()
        scheduler_classifier.step()

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

        if (
            student_metrics[
                "mean_class_accuracy"
            ]
            > best_student
        ):
            best_student = (
                student_metrics[
                    "mean_class_accuracy"
                ]
            )

            best_student_epoch = epoch

            best_student_state = {
                key:
                    value.detach().cpu().clone()
                for key, value in student.state_dict().items()
            }

        if (
            teacher_metrics[
                "mean_class_accuracy"
            ]
            > best_teacher
        ):
            best_teacher = (
                teacher_metrics[
                    "mean_class_accuracy"
                ]
            )

            best_teacher_epoch = epoch

            best_teacher_state = {
                key:
                    value.detach().cpu().clone()
                for key, value in teacher.state_dict().items()
            }

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        recovery_total = int(
            recovery_epoch_counts.sum().item()
        )

        recovery_precision = {}

        for class_id, class_name in enumerate(
            CLASSES
        ):
            denominator = int(
                recovery_precision_denominator[
                    class_id
                ].item()
            )

            numerator = int(
                recovery_precision_numerator[
                    class_id
                ].item()
            )

            if denominator > 0:
                recovery_precision[
                    class_name
                ] = (
                    numerator
                    / denominator
                )
            else:
                recovery_precision[
                    class_name
                ] = None

        record = {
            "epoch":
                epoch,
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
            "pseudo_coverage":
                100.0
                * metrics[
                    "pseudo_selected"
                ]
                / max(
                    len(target_features),
                    1
                ),
            "geometry_rejections":
                metrics[
                    "geometry_rejections"
                ],
            "confidence_candidates":
                metrics[
                    "confidence_candidates"
                ],
            "recovery_total":
                recovery_total,
            "recovery_counts":
                recovery_epoch_counts.tolist(),
            "recovery_precision":
                recovery_precision,
            "trusted_counts_before_epoch":
                trusted_counts.tolist(),
            "trusted_fractions_before_epoch":
                trusted_fractions.tolist(),
            "candidate_starved_classes":
                [
                    CLASSES[class_id]
                    for class_id in candidate_starved
                ],
            "active_starved_classes":
                [
                    CLASSES[class_id]
                    for class_id in active_starved_classes
                ],
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
            "epoch_seconds":
                epoch_seconds
        }

        history.append(
            record
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Source "
            f"{metrics['source_accuracy']:.2f}% | "
            f"Student "
            f"{student_metrics['mean_class_accuracy']:.2f}% | "
            f"Teacher "
            f"{teacher_metrics['mean_class_accuracy']:.2f}% | "
            f"PL "
            f"{record['pseudo_coverage']:.2f}% | "
            f"Recovery "
            f"{recovery_total} | "
            f"{epoch_seconds:.2f}s"
        )

        print(
            "Trusted mass:"
        )

        for class_name, count, fraction in zip(
            CLASSES,
            trusted_counts.tolist(),
            trusted_fractions.tolist()
        ):
            print(
                f"  {class_name:12s}: "
                f"{count:6d} "
                f"({100.0 * fraction:6.2f}%)"
            )

        print(
            "Recovery:"
        )

        for class_name, count in zip(
            CLASSES,
            recovery_epoch_counts.tolist()
        ):
            if count > 0:
                precision = recovery_precision[
                    class_name
                ]

                precision_text = (
                    "NA"
                    if precision is None
                    else f"{100.0 * precision:.2f}%"
                )

                print(
                    f"  {class_name:12s}: "
                    f"{count:3d} | "
                    f"precision={precision_text}"
                )

        if active_starved_classes:
            print(
                "Active starved classes:"
            )

            for class_id in active_starved_classes:
                print(
                    f"  {CLASSES[class_id]}"
                )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    student.load_state_dict(
        best_student_state
    )

    teacher.load_state_dict(
        best_teacher_state
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

    total_recovery = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for row in history:
        total_recovery += torch.tensor(
            row[
                "recovery_counts"
            ],
            dtype=torch.long
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
        f"{best_student_epoch}"
    )

    print(
        f"Best student mean-class: "
        f"{best_student:.2f}%"
    )

    print(
        f"Best teacher epoch: "
        f"{best_teacher_epoch}"
    )

    print(
        f"Best teacher mean-class: "
        f"{best_teacher:.2f}%"
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
        "TOTAL RECOVERY ASSIGNMENTS:"
    )

    for class_name, count in zip(
        CLASSES,
        total_recovery.tolist()
    ):
        print(
            f"{class_name:12s}: "
            f"{count}"
        )

    print()
    print(
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "controlled_recovery_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "controlled_recovery_seed42.json"
    )

    torch.save(
        {
            "student_state_dict":
                student.state_dict(),
            "teacher_state_dict":
                teacher.state_dict(),
            "source_prototypes":
                source_prototypes.cpu(),
            "base_checkpoint":
                str(BASE_CHECKPOINT),
            "seed":
                SEED,
            "classes":
                CLASSES,
            "input_dim":
                INPUT_DIM,
            "hidden_dim":
                HIDDEN_DIM,
            "epochs":
                EPOCHS,
            "pl_threshold":
                PL_THRESHOLD,
            "global_pl_fraction":
                GLOBAL_PL_FRACTION,
            "geometry_similarity_threshold":
                GEOMETRY_SIMILARITY_THRESHOLD,
            "geometry_margin_threshold":
                GEOMETRY_MARGIN_THRESHOLD,
            "recovery_max_per_class_per_epoch":
                RECOVERY_MAX_PER_CLASS_PER_EPOCH,
            "recovery_confidence_min":
                RECOVERY_CONFIDENCE_MIN,
            "recovery_similarity_min":
                RECOVERY_SIMILARITY_MIN,
            "recovery_weight":
                RECOVERY_WEIGHT,
            "best_student_epoch":
                best_student_epoch,
            "best_student_mean_class":
                best_student,
            "best_teacher_epoch":
                best_teacher_epoch,
            "best_teacher_mean_class":
                best_teacher,
            "final_student":
                final_student,
            "final_teacher":
                final_teacher,
            "total_recovery":
                total_recovery.tolist(),
            "history":
                history
        },
        checkpoint_path
    )

    report = {
        "experiment":
            "visda_geometry_gated_mcd_controlled_recovery",
        "seed":
            SEED,
        "base_checkpoint":
            str(BASE_CHECKPOINT),
        "input_dim":
            INPUT_DIM,
        "hidden_dim":
            HIDDEN_DIM,
        "epochs":
            EPOCHS,
        "pl_threshold":
            PL_THRESHOLD,
        "global_pl_fraction":
            GLOBAL_PL_FRACTION,
        "geometry_similarity_threshold":
            GEOMETRY_SIMILARITY_THRESHOLD,
        "geometry_margin_threshold":
            GEOMETRY_MARGIN_THRESHOLD,
        "starvation_fraction":
            STARVATION_FRACTION,
        "starvation_patience":
            STARVATION_PATIENCE,
        "recovery_max_per_class_per_epoch":
            RECOVERY_MAX_PER_CLASS_PER_EPOCH,
        "recovery_confidence_min":
            RECOVERY_CONFIDENCE_MIN,
        "recovery_similarity_min":
            RECOVERY_SIMILARITY_MIN,
        "recovery_weight":
            RECOVERY_WEIGHT,
        "base_metrics":
            base_metrics,
        "best_student_epoch":
            best_student_epoch,
        "best_student_mean_class":
            best_student,
        "best_teacher_epoch":
            best_teacher_epoch,
        "best_teacher_mean_class":
            best_teacher,
        "final_student":
            final_student,
        "final_teacher":
            final_teacher,
        "total_recovery":
            total_recovery.tolist(),
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