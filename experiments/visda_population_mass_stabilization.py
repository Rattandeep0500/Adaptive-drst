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

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096

EPOCHS = 10

EMA_DECAY = 0.97

LR_ADAPTER = 0.0005
LR_CLASSIFIER = 0.005

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

PL_THRESHOLD = 0.90
GLOBAL_PL_FRACTION = 0.25
PL_WEIGHT = 0.20

GEOMETRY_SIMILARITY_THRESHOLD = 0.60
GEOMETRY_MARGIN_THRESHOLD = 0.00

ANCHOR_CONFIDENCE = 0.90
MAX_ANCHORS_PER_CLASS = 2000

LOCAL_K = 50
LOCAL_SUPPORT_MIN = 100

STARVED_MASS_THRESHOLD = 0.005
STARVED_PATIENCE = 2

TARGET_MASS_FLOOR = 0.003
MASS_LOSS_WEIGHT = 0.01

KNN_GRAPH_PATH = Path(
    "checkpoints/visda_target_knn_geometry/"
    "target_knn_indices_seed42.pt"
)

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/"
    "geometry_gated_mcd_seed42.pt"
)

CACHE_ROOT = Path(
    "checkpoints/visda_feature_cache"
)

SOURCE_CACHE = (
    CACHE_ROOT / "source"
)

TARGET_CACHE = (
    CACHE_ROOT / "target"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_population_mass_stabilization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CHECKPOINT = (
    OUTPUT_DIR
    / "population_mass_stabilization_seed42.pt"
)

OUTPUT_HISTORY = (
    OUTPUT_DIR
    / "population_mass_stabilization_seed42.json"
)

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

TRUCK_ID = 11


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
            f"No cache files found: {cache_dir}"
        )

    feature_parts = []
    label_parts = []

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu"
        )

        if "features" not in payload:
            raise RuntimeError(
                f"'features' missing in {path}"
            )

        if "labels" not in payload:
            raise RuntimeError(
                f"'labels' missing in {path}"
            )

        features = payload[
            "features"
        ].float()

        labels = payload[
            "labels"
        ].long()

        if features.ndim != 2:
            raise RuntimeError(
                f"Invalid feature rank in {path}"
            )

        if features.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"got {features.shape[1]}"
            )

        if len(features) != len(labels):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
            )

        feature_parts.append(
            features
        )

        label_parts.append(
            labels
        )

    features = torch.cat(
        feature_parts,
        dim=0
    )

    labels = torch.cat(
        label_parts,
        dim=0
    )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Final feature dimension mismatch"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Final feature/label mismatch"
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
                f"Adapter expected 2-D input, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expected {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter expected output {HIDDEN_DIM}-D, "
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
        if x.ndim != 2:
            raise RuntimeError(
                f"Model expected 2-D input, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Model expected {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.adapter(x)

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
                "teacher.",
            ):
                if new_key.startswith(prefix):
                    new_key = new_key[
                        len(prefix):
                    ]
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
            f"Missing checkpoint:\n"
            f"{BASE_CHECKPOINT}"
        )

    payload = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
    )

    required = (
        "student_state_dict",
        "teacher_state_dict",
        "source_prototypes",
    )

    for key in required:
        if key not in payload:
            raise RuntimeError(
                f"Checkpoint missing {key}"
            )

    student.load_state_dict(
        normalize_state_dict(
            payload[
                "student_state_dict"
            ]
        ),
        strict=True
    )

    teacher.load_state_dict(
        normalize_state_dict(
            payload[
                "teacher_state_dict"
            ]
        ),
        strict=True
    )

    source_prototypes = payload[
        "source_prototypes"
    ].float()

    if source_prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Invalid source prototype shape: "
            f"{tuple(source_prototypes.shape)}"
        )

    return F.normalize(
        source_prototypes,
        dim=1
    )


def load_knn_graph(
    n_samples
):
    if not KNN_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing KNN graph:\n"
            f"{KNN_GRAPH_PATH}"
        )

    payload = torch.load(
        KNN_GRAPH_PATH,
        map_location="cpu"
    )

    if "indices" not in payload:
        raise RuntimeError(
            "KNN graph missing 'indices'"
        )

    indices = payload[
        "indices"
    ].long()

    if indices.ndim != 2:
        raise RuntimeError(
            f"Invalid KNN graph shape: "
            f"{tuple(indices.shape)}"
        )

    if indices.shape[0] != n_samples:
        raise RuntimeError(
            f"KNN rows={indices.shape[0]}, "
            f"expected {n_samples}"
        )

    if indices.shape[1] < LOCAL_K:
        raise RuntimeError(
            f"KNN graph provides only "
            f"{indices.shape[1]} neighbors"
        )

    if (indices < 0).any():
        raise RuntimeError(
            "Negative KNN index detected"
        )

    if (
        indices >= n_samples
    ).any():
        raise RuntimeError(
            "Out-of-range KNN index detected"
        )

    return indices[
        :,
        :LOCAL_K
    ]


@torch.no_grad()
def update_ema(
    teacher,
    student
):
    teacher_parameters = dict(
        teacher.named_parameters()
    )

    student_parameters = dict(
        student.named_parameters()
    )

    for name in teacher_parameters:
        teacher_parameters[
            name
        ].mul_(
            EMA_DECAY
        )

        teacher_parameters[
            name
        ].add_(
            student_parameters[
                name
            ],
            alpha=1.0 - EMA_DECAY
        )

    teacher_buffers = dict(
        teacher.named_buffers()
    )

    student_buffers = dict(
        student.named_buffers()
    )

    for name in teacher_buffers:
        teacher_buffers[
            name
        ].copy_(
            student_buffers[
                name
            ]
        )


@torch.no_grad()
def collect_target_state(
    student,
    teacher,
    target_features,
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

    z_parts = []
    p_parts = []
    confidence_parts = []
    prediction_parts = []
    disagreement_parts = []

    for (x,) in loader:
        x = x.to(
            device
        )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Target raw input dimension mismatch"
            )

        z = student.encode(
            x
        )

        logits1, logits2 = teacher(
            x
        )

        p1 = F.softmax(
            logits1,
            dim=1
        )

        p2 = F.softmax(
            logits2,
            dim=1
        )

        p = (
            p1 + p2
        ) / 2.0

        confidence, predictions = p.max(
            dim=1
        )

        disagreement = (
            p1 - p2
        ).abs().mean(
            dim=1
        )

        z_parts.append(
            F.normalize(
                z,
                dim=1
            ).cpu()
        )

        p_parts.append(
            p.cpu()
        )

        confidence_parts.append(
            confidence.cpu()
        )

        prediction_parts.append(
            predictions.cpu()
        )

        disagreement_parts.append(
            disagreement.cpu()
        )

    return {
        "z":
            torch.cat(
                z_parts,
                dim=0
            ),
        "probabilities":
            torch.cat(
                p_parts,
                dim=0
            ),
        "confidence":
            torch.cat(
                confidence_parts,
                dim=0
            ),
        "predictions":
            torch.cat(
                prediction_parts,
                dim=0
            ),
        "disagreement":
            torch.cat(
                disagreement_parts,
                dim=0
            )
    }


@torch.no_grad()
def build_local_prototypes(
    z,
    predictions,
    confidence,
    knn_indices
):
    prototypes = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM
    )

    support = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    anchor_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            (predictions == class_id)
            & (
                confidence
                >= ANCHOR_CONFIDENCE
            )
        )

        indices = torch.nonzero(
            mask,
            as_tuple=False
        ).flatten()

        if len(indices) == 0:
            continue

        if len(indices) > MAX_ANCHORS_PER_CLASS:
            order = torch.argsort(
                confidence[
                    indices
                ],
                descending=True
            )

            indices = indices[
                order[
                    :MAX_ANCHORS_PER_CLASS
                ]
            ]

        anchor_counts[
            class_id
        ] = len(indices)

        neighbor_indices = (
            knn_indices[
                indices
            ]
        ).reshape(
            -1
        )

        local_indices = torch.unique(
            neighbor_indices
        )

        if len(local_indices) == 0:
            continue

        anchor_z = z[
            indices
        ]

        local_z = z[
            local_indices
        ]

        anchor_weights = confidence[
            indices
        ]

        anchor_weights = (
            anchor_weights
            / anchor_weights.sum().clamp_min(
                1e-8
            )
        )

        anchor_center = (
            anchor_z
            * anchor_weights.unsqueeze(1)
        ).sum(
            dim=0
        )

        local_center = local_z.mean(
            dim=0
        )

        prototype = (
            0.5 * anchor_center
            + 0.5 * local_center
        )

        prototypes[
            class_id
        ] = F.normalize(
            prototype,
            dim=0
        )

        support[
            class_id
        ] = len(local_indices)

    return (
        prototypes,
        support,
        anchor_counts
    )


@torch.no_grad()
def target_probability_mass(
    probabilities
):
    return probabilities.mean(
        dim=0
    )


def make_geometry_candidates(
    z,
    predictions,
    confidence,
    prototypes
):
    prototypes = F.normalize(
        prototypes,
        dim=1
    )

    similarities = (
        z
        @ prototypes.t()
    )

    row_indices = torch.arange(
        len(z)
    )

    predicted_similarity = (
        similarities[
            row_indices,
            predictions
        ]
    )

    sorted_similarity = torch.sort(
        similarities,
        dim=1,
        descending=True
    ).values

    second_similarity = (
        sorted_similarity[
            :,
            1
        ]
    )

    margin = (
        predicted_similarity
        - second_similarity
    )

    candidate = (
        (confidence >= PL_THRESHOLD)
        & (
            predicted_similarity
            >= GEOMETRY_SIMILARITY_THRESHOLD
        )
        & (
            margin
            >= GEOMETRY_MARGIN_THRESHOLD
        )
    )

    return (
        candidate,
        similarities,
        margin
    )


def enforce_global_cap(
    candidate,
    confidence
):
    indices = torch.nonzero(
        candidate,
        as_tuple=False
    ).flatten()

    limit = int(
        len(candidate)
        * GLOBAL_PL_FRACTION
    )

    limit = max(
        1,
        limit
    )

    selected = torch.zeros_like(
        candidate
    )

    if len(indices) <= limit:
        selected[
            indices
        ] = True

        return selected

    order = torch.argsort(
        confidence[
            indices
        ],
        descending=True
    )

    chosen = indices[
        order[
            :limit
        ]
    ]

    selected[
        chosen
    ] = True

    return selected


def population_mass_penalty(
    probabilities,
    active_classes
):
    if not active_classes:
        return (
            probabilities.sum()
            * 0.0
        )

    mass = probabilities.mean(
        dim=0
    )

    terms = []

    for class_id in active_classes:
        deficit = F.relu(
            TARGET_MASS_FLOOR
            - mass[
                class_id
            ]
        )

        terms.append(
            deficit.pow(2)
        )

    return torch.stack(
        terms
    ).mean()


def mcd_train_step(
    student,
    source_x,
    source_y,
    target_x,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    if source_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Source batch is not raw 2048-D"
        )

    if target_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Target batch is not raw 2048-D"
        )

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
        student(
            source_x
        )
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

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    source_z = student.encode(
        source_x
    )

    target_z = student.encode(
        target_x
    )

    if source_z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            "Source adapted representation "
            "is not 512-D"
        )

    if target_z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            "Target adapted representation "
            "is not 512-D"
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

    source_predictions = (
        (
            source_logits1
            + source_logits2
        )
        / 2.0
    ).argmax(
        dim=1
    )

    return {
        "source_correct":
            int(
                (
                    source_predictions
                    == source_y
                ).sum().item()
            ),
        "source_total":
            len(source_y),
        "source_loss":
            float(
                source_loss.item()
            ),
        "discrepancy":
            float(
                target_discrepancy.item()
            )
    }


def train_pseudo_batch(
    student,
    raw_x,
    pseudo_y,
    confidence,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    if raw_x.ndim != 2:
        raise RuntimeError(
            "Pseudo-label input must be 2-D"
        )

    if raw_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Pseudo-label input must remain raw 2048-D"
        )

    raw_x = raw_x.to(
        device
    )

    pseudo_y = pseudo_y.to(
        device
    )

    confidence = confidence.to(
        device
    )

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    z = student.encode(
        raw_x
    )

    logits1 = student.classifier1(
        z
    )

    logits2 = student.classifier2(
        z
    )

    loss = (
        F.cross_entropy(
            logits1,
            pseudo_y,
            reduction="none"
        )
        + F.cross_entropy(
            logits2,
            pseudo_y,
            reduction="none"
        )
    )

    loss = (
        loss
        * confidence
    ).mean()

    (
        PL_WEIGHT
        * loss
    ).backward()

    optimizer_adapter.step()

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():
        z = student.encode(
            raw_x
        )

    logits1 = student.classifier1(
        z
    )

    logits2 = student.classifier2(
        z
    )

    classifier_loss = (
        F.cross_entropy(
            logits1,
            pseudo_y,
            reduction="none"
        )
        + F.cross_entropy(
            logits2,
            pseudo_y,
            reduction="none"
        )
    )

    classifier_loss = (
        classifier_loss
        * confidence
    ).mean()

    (
        PL_WEIGHT
        * classifier_loss
    ).backward()

    optimizer_classifier.step()

    return float(
        loss.item()
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

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    total_correct = 0
    total_count = 0

    confidence_sum = 0.0

    for x, y in loader:
        x = x.to(
            device
        )

        y = y.to(
            device
        )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Evaluation input dimension mismatch"
            )

        logits1, logits2 = model(
            x
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

        total_correct += int(
            (
                predictions
                == y
            ).sum().item()
        )

        total_count += len(y)

        confidence_sum += (
            confidence.sum().item()
        )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                y == class_id
            )

            if not mask.any():
                continue

            count = int(
                mask.sum().item()
            )

            class_total[
                class_id
            ] += count

            class_correct[
                class_id
            ] += int(
                (
                    predictions[mask]
                    == y[mask]
                ).sum().item()
            )

    per_class = (
        100.0
        * class_correct.float()
        / class_total.clamp_min(
            1
        )
    )

    return {
        "overall_accuracy":
            100.0
            * total_correct
            / max(
                total_count,
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
                total_count,
                1
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
        "VISDA-2017 POPULATION-MASS STABILIZATION"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
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
        f"anchor_confidence={ANCHOR_CONFIDENCE}"
    )

    print(
        f"mass_loss_weight={MASS_LOSS_WEIGHT}"
    )

    print(
        f"target_mass_floor={TARGET_MASS_FLOOR}"
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

    print()

    print(
        "Loading target KNN graph..."
    )

    knn_indices = load_knn_graph(
        len(target_features)
    )

    print(
        f"KNN graph: "
        f"{tuple(knn_indices.shape)}"
    )

    student = MCDModel().to(
        device
    )

    teacher = MCDModel().to(
        device
    )

    source_prototypes_cpu = (
        load_base_checkpoint(
            student,
            teacher
        )
    )

    source_prototypes = (
        source_prototypes_cpu.to(
            device
        )
    )

    print()

    print(
        "Base checkpoint loaded successfully."
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

    best_score = (
        base_metrics[
            "mean_class_accuracy"
        ]
    )

    best_epoch = 0

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

    history = []

    starvation_counter = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    total_start = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        epoch_start = time.perf_counter()

        print()
        print("=" * 90)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 90)

        state = collect_target_state(
            student,
            teacher,
            target_features,
            device
        )

        local_prototypes, local_support, anchor_counts = (
            build_local_prototypes(
                state["z"],
                state["predictions"],
                state["confidence"],
                knn_indices
            )
        )

        local_prototypes_device = (
            local_prototypes.to(
                device
            )
        )

        current_mass = target_probability_mass(
            state["probabilities"]
        )

        for class_id in range(
            NUM_CLASSES
        ):
            if (
                current_mass[
                    class_id
                ].item()
                < STARVED_MASS_THRESHOLD
            ):
                starvation_counter[
                    class_id
                ] += 1
            else:
                starvation_counter[
                    class_id
                ] = 0

        active_starved = []

        for class_id in range(
            NUM_CLASSES
        ):
            if (
                starvation_counter[
                    class_id
                ].item()
                >= STARVED_PATIENCE
                and local_support[
                    class_id
                ].item()
                >= LOCAL_SUPPORT_MIN
            ):
                active_starved.append(
                    class_id
                )

        candidate, similarities, margins = (
            make_geometry_candidates(
                state["z"],
                state["predictions"],
                state["confidence"],
                source_prototypes_cpu
            )
        )

        hybrid_candidate, hybrid_similarities, hybrid_margins = (
            make_geometry_candidates(
                state["z"],
                state["predictions"],
                state["confidence"],
                local_prototypes_device
            )
        )

        selected = enforce_global_cap(
            hybrid_candidate,
            state["confidence"]
        )

        selected_indices = torch.nonzero(
            selected,
            as_tuple=False
        ).flatten()

        selected_count = len(
            selected_indices
        )

        selected_fraction = (
            selected_count
            / len(target_features)
        )

        if selected_fraction > (
            GLOBAL_PL_FRACTION
            + 1e-9
        ):
            raise RuntimeError(
                "Global PL fraction exceeded"
            )

        selected_labels = (
            state["predictions"][
                selected_indices
            ]
        )

        selected_counts = torch.bincount(
            selected_labels,
            minlength=NUM_CLASSES
        )

        selected_fractions = (
            selected_counts.float()
            / max(
                selected_count,
                1
            )
        )

        print()
        print(
            "Target probability mass:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"  {CLASSES[class_id]:12s}: "
                f"{100.0 * current_mass[class_id].item():7.3f}%"
            )

        print()
        print(
            "Local target support:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"  {CLASSES[class_id]:12s}: "
                f"{int(local_support[class_id].item())}"
            )

        print()
        print(
            "Anchor counts:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"  {CLASSES[class_id]:12s}: "
                f"{int(anchor_counts[class_id].item())}"
            )

        if active_starved:
            print()
            print(
                "Active mass-stabilized classes:"
            )

            for class_id in active_starved:
                print(
                    f"  {CLASSES[class_id]}"
                )

        print()
        print(
            "Selected pseudo-label distribution:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"  {CLASSES[class_id]:12s}: "
                f"{int(selected_counts[class_id].item()):5d} "
                f"({100.0 * selected_fractions[class_id].item():6.2f}%)"
            )

        source_iterator = iter(
            source_loader
        )

        target_iterator = iter(
            target_loader
        )

        source_correct = 0
        source_total = 0

        source_loss_sum = 0.0
        discrepancy_sum = 0.0
        pseudo_loss_sum = 0.0
        mass_loss_sum = 0.0
        mass_steps = 0

        pseudo_pointer = 0

        steps = len(
            source_loader
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

            step = mcd_train_step(
                student,
                source_x,
                source_y,
                target_x,
                optimizer_adapter,
                optimizer_classifier,
                device
            )

            source_correct += (
                step["source_correct"]
            )

            source_total += (
                step["source_total"]
            )

            source_loss_sum += (
                step["source_loss"]
            )

            discrepancy_sum += (
                step["discrepancy"]
            )

            if (
                pseudo_pointer
                < selected_count
            ):
                end = min(
                    pseudo_pointer
                    + BATCH_SIZE,
                    selected_count
                )

                indices = (
                    selected_indices[
                        pseudo_pointer:end
                    ]
                )

                raw_x = (
                    target_features[
                        indices
                    ]
                )

                pseudo_y = (
                    state["predictions"][
                        indices
                    ]
                )

                confidence = (
                    state["confidence"][
                        indices
                    ]
                )

                pseudo_loss = train_pseudo_batch(
                    student,
                    raw_x,
                    pseudo_y,
                    confidence,
                    optimizer_adapter,
                    optimizer_classifier,
                    device
                )

                pseudo_loss_sum += (
                    pseudo_loss
                )

                pseudo_pointer = end

                update_ema(
                    teacher,
                    student
                )

            if active_starved:
                optimizer_classifier.zero_grad(
                    set_to_none=True
                )

                mass_x = target_features[
                    :
                    min(
                        BATCH_SIZE,
                        len(target_features)
                    )
                ].to(
                    device
                )

                logits1, logits2 = student(
                    mass_x
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

                mass_loss = population_mass_penalty(
                    probabilities,
                    active_starved
                )

                (
                    MASS_LOSS_WEIGHT
                    * mass_loss
                ).backward()

                optimizer_classifier.step()

                mass_loss_sum += (
                    mass_loss.item()
                )

                mass_steps += 1

        scheduler_adapter.step()
        scheduler_classifier.step()

        update_ema(
            teacher,
            student
        )

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
            > best_score
        ):
            best_score = (
                student_metrics[
                    "mean_class_accuracy"
                ]
            )

            best_epoch = epoch

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

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch":
                epoch,
            "target_probability_mass":
                current_mass.tolist(),
            "local_support":
                local_support.tolist(),
            "anchor_counts":
                anchor_counts.tolist(),
            "active_starved_classes":
                [
                    CLASSES[c]
                    for c in active_starved
                ],
            "selected_pl_count":
                selected_count,
            "selected_pl_fraction":
                selected_fraction,
            "selected_counts":
                selected_counts.tolist(),
            "selected_fractions":
                selected_fractions.tolist(),
            "source_geometry_coverage":
                float(
                    candidate.float().mean().item()
                ),
            "local_geometry_coverage":
                float(
                    hybrid_candidate.float().mean().item()
                ),
            "source_accuracy":
                100.0
                * source_correct
                / max(
                    source_total,
                    1
                ),
            "student_overall":
                student_metrics[
                    "overall_accuracy"
                ],
            "student_mean_class":
                student_metrics[
                    "mean_class_accuracy"
                ],
            "student_per_class":
                student_metrics[
                    "per_class_accuracy"
                ],
            "teacher_overall":
                teacher_metrics[
                    "overall_accuracy"
                ],
            "teacher_mean_class":
                teacher_metrics[
                    "mean_class_accuracy"
                ],
            "teacher_per_class":
                teacher_metrics[
                    "per_class_accuracy"
                ],
            "source_loss":
                source_loss_sum
                / max(
                    steps,
                    1
                ),
            "mcd_discrepancy":
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
            "mass_loss":
                mass_loss_sum
                / max(
                    mass_steps,
                    1
                ),
            "epoch_seconds":
                epoch_seconds
        }

        history.append(
            record
        )

        print()
        print(
            f"Source "
            f"{record['source_accuracy']:.2f}% | "
            f"Student "
            f"{record['student_mean_class']:.2f}% | "
            f"Teacher "
            f"{record['teacher_mean_class']:.2f}%"
        )

        print(
            f"PL: "
            f"{selected_count} "
            f"({100.0 * selected_fraction:.2f}%)"
        )

        print(
            f"Mass loss: "
            f"{record['mass_loss']:.8f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_seconds:.2f}s"
        )

    total_seconds = (
        time.perf_counter()
        - total_start
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

    torch.save(
        {
            "student_state_dict":
                student.state_dict(),
            "teacher_state_dict":
                teacher.state_dict(),
            "source_prototypes":
                source_prototypes_cpu,
            "best_epoch":
                best_epoch,
            "best_mean_class":
                best_score,
            "history":
                history
        },
        OUTPUT_CHECKPOINT
    )

    report = {
        "experiment":
            "visda_population_mass_stabilization",
        "seed":
            SEED,
        "base_checkpoint":
            str(BASE_CHECKPOINT),
        "knn_graph":
            str(KNN_GRAPH_PATH),
        "epochs":
            EPOCHS,
        "pl_threshold":
            PL_THRESHOLD,
        "global_pl_fraction":
            GLOBAL_PL_FRACTION,
        "anchor_confidence":
            ANCHOR_CONFIDENCE,
        "mass_loss_weight":
            MASS_LOSS_WEIGHT,
        "target_mass_floor":
            TARGET_MASS_FLOOR,
        "starved_mass_threshold":
            STARVED_MASS_THRESHOLD,
        "starved_patience":
            STARVED_PATIENCE,
        "local_k":
            LOCAL_K,
        "local_support_min":
            LOCAL_SUPPORT_MIN,
        "base_metrics":
            base_metrics,
        "best_epoch":
            best_epoch,
        "best_mean_class":
            best_score,
        "final_student":
            final_student,
        "final_teacher":
            final_teacher,
        "total_seconds":
            total_seconds,
        "history":
            history
    }

    with open(
        OUTPUT_HISTORY,
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
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best mean-class: "
        f"{best_score:.2f}%"
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
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    print()
    print(
        f"checkpoint={OUTPUT_CHECKPOINT}"
    )

    print(
        f"history={OUTPUT_HISTORY}"
    )


if __name__ == "__main__":
    main()