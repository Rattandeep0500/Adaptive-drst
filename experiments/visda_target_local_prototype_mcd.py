import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

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

# Pseudo-label configuration
PL_THRESHOLD = 0.90
GLOBAL_PL_FRACTION = 0.25
PL_WEIGHT = 0.20

# Source geometry
SOURCE_SIMILARITY_THRESHOLD = 0.60
SOURCE_MARGIN_THRESHOLD = 0.00

# Target-local geometry
LOCAL_K = 50
LOCAL_SIMILARITY_THRESHOLD = 0.60
LOCAL_MARGIN_THRESHOLD = 0.00

# Target-local prototype construction
ANCHOR_CONFIDENCE = 0.90
ANCHOR_TOP_FRACTION = 0.10
MAX_ANCHORS_PER_CLASS = 2000

# How strongly local geometry replaces source geometry.
# This is deliberately conservative.
LOCAL_MIX_START = 0.00
LOCAL_MIX_MAX = 0.75

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
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10
SKATEBOARD_ID = 9
PERSON_ID = 7
KNIFE_ID = 5


# ============================================================
# PATHS
# ============================================================

CACHE_ROOT = Path(
    "checkpoints/visda_feature_cache"
)

SOURCE_CACHE = (
    CACHE_ROOT / "source"
)

TARGET_CACHE = (
    CACHE_ROOT / "target"
)

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/"
    "geometry_gated_mcd_seed42.pt"
)

KNN_GRAPH = Path(
    "checkpoints/visda_target_knn_geometry/"
    "target_knn_indices_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_target_local_prototype_mcd"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CHECKPOINT = (
    OUTPUT_DIR
    / "target_local_prototype_mcd_seed42.pt"
)

OUTPUT_HISTORY = (
    OUTPUT_DIR
    / "target_local_prototype_mcd_seed42.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATA
# ============================================================

def load_cache(cache_dir):
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No cache chunks found in {cache_dir}"
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

        x = payload[
            "features"
        ].float()

        y = payload[
            "labels"
        ].long()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature rank in {path}: "
                f"{tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"{path}: expected "
                f"{INPUT_DIM}-D features, "
                f"got {x.shape[1]}"
            )

        if len(x) != len(y):
            raise RuntimeError(
                f"{path}: feature/label length mismatch"
            )

        feature_parts.append(x)
        label_parts.append(y)

    features = torch.cat(
        feature_parts,
        dim=0
    )

    labels = torch.cat(
        label_parts,
        dim=0
    )

    if features.ndim != 2:
        raise RuntimeError(
            "Final features are not 2-D"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Final feature dimension should be "
            f"{INPUT_DIM}, got {features.shape[1]}"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Final features/labels mismatch"
        )

    return features, labels


# ============================================================
# MODEL
# ============================================================

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
            ),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(
                f"Adapter expects 2-D input, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expects {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter must output {HIDDEN_DIM}-D, "
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
                f"Model expects 2-D input, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Model expects {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.adapter(x)

        logits1 = self.classifier1(
            z
        )

        logits2 = self.classifier2(
            z
        )

        return (
            logits1,
            logits2
        )


# ============================================================
# CHECKPOINT
# ============================================================

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
            f"Base checkpoint not found:\n"
            f"{BASE_CHECKPOINT}"
        )

    payload = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
    )

    if (
        "student_state_dict"
        not in payload
    ):
        raise RuntimeError(
            "student_state_dict missing"
        )

    if (
        "teacher_state_dict"
        not in payload
    ):
        raise RuntimeError(
            "teacher_state_dict missing"
        )

    if (
        "source_prototypes"
        not in payload
    ):
        raise RuntimeError(
            "source_prototypes missing"
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
            "Invalid source prototype shape: "
            f"{tuple(source_prototypes.shape)}"
        )

    source_prototypes = F.normalize(
        source_prototypes,
        dim=1
    )

    return source_prototypes


# ============================================================
# EMA
# ============================================================

@torch.no_grad()
def update_ema(
    teacher,
    student
):
    student_parameters = dict(
        student.named_parameters()
    )

    teacher_parameters = dict(
        teacher.named_parameters()
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

    student_buffers = dict(
        student.named_buffers()
    )

    teacher_buffers = dict(
        teacher.named_buffers()
    )

    for name in teacher_buffers:
        teacher_buffers[
            name
        ].copy_(
            student_buffers[
                name
            ]
        )


# ============================================================
# MCD
# ============================================================

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


# ============================================================
# TARGET STATE
# ============================================================

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

    adapted_parts = []
    probability_parts = []
    confidence_parts = []
    prediction_parts = []
    disagreement_parts = []

    for (x_raw,) in loader:
        x_raw = x_raw.to(
            device
        )

        if x_raw.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Target batch lost input dimension"
            )

        z = student.encode(
            x_raw
        )

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                "Adapted target batch has "
                f"{z.shape[1]} dimensions"
            )

        logits1, logits2 = teacher(
            x_raw
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

        adapted_parts.append(
            F.normalize(
                z,
                dim=1
            ).cpu()
        )

        probability_parts.append(
            probabilities.cpu()
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
                adapted_parts,
                dim=0
            ),
        "probabilities":
            torch.cat(
                probability_parts,
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


# ============================================================
# KNN GRAPH
# ============================================================

def load_knn_graph(
    expected_samples
):
    if not KNN_GRAPH.exists():
        raise FileNotFoundError(
            f"KNN graph not found:\n"
            f"{KNN_GRAPH}"
        )

    payload = torch.load(
        KNN_GRAPH,
        map_location="cpu"
    )

    if "indices" not in payload:
        raise RuntimeError(
            "KNN graph does not contain 'indices'"
        )

    indices = payload[
        "indices"
    ].long()

    if indices.ndim != 2:
        raise RuntimeError(
            f"Invalid KNN shape: "
            f"{tuple(indices.shape)}"
        )

    if indices.shape[0] != expected_samples:
        raise RuntimeError(
            f"KNN rows={indices.shape[0]}, "
            f"expected {expected_samples}"
        )

    if indices.shape[1] < LOCAL_K:
        raise RuntimeError(
            f"KNN contains only "
            f"{indices.shape[1]} neighbors; "
            f"LOCAL_K={LOCAL_K}"
        )

    if (indices < 0).any():
        raise RuntimeError(
            "KNN contains negative indices"
        )

    if (
        indices >= expected_samples
    ).any():
        raise RuntimeError(
            "KNN contains out-of-range indices"
        )

    return indices


# ============================================================
# TARGET-LOCAL PROTOTYPES
# ============================================================

@torch.no_grad()
def build_target_local_prototypes(
    z,
    confidence,
    predictions,
    knn_indices
):
    if z.ndim != 2:
        raise RuntimeError(
            "z must be 2-D"
        )

    if z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            f"z must be {HIDDEN_DIM}-D"
        )

    anchors = torch.zeros(
        len(z),
        dtype=torch.bool
    )

    anchor_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    prototypes = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float32
    )

    support_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for class_id in range(
        NUM_CLASSES
    ):
        class_mask = (
            (predictions == class_id)
            & (
                confidence
                >= ANCHOR_CONFIDENCE
            )
        )

        class_indices = torch.nonzero(
            class_mask,
            as_tuple=False
        ).flatten()

        if len(class_indices) == 0:
            continue

        desired = int(
            round(
                len(z)
                * ANCHOR_TOP_FRACTION
            )
        )

        desired = max(
            1,
            desired
        )

        desired = min(
            desired,
            MAX_ANCHORS_PER_CLASS,
            len(class_indices)
        )

        if len(class_indices) > desired:
            order = torch.argsort(
                confidence[
                    class_indices
                ],
                descending=True
            )

            class_indices = (
                class_indices[
                    order[:desired]
                ]
            )

        anchors[
            class_indices
        ] = True

        anchor_counts[
            class_id
        ] = len(class_indices)

        neighbor_indices = (
            knn_indices[
                class_indices,
                :LOCAL_K
            ]
        )

        local_indices = torch.unique(
            neighbor_indices.reshape(-1)
        )

        if len(local_indices) == 0:
            continue

        anchor_features = z[
            class_indices
        ]

        anchor_confidence = confidence[
            class_indices
        ]

        anchor_weights = (
            anchor_confidence
            / anchor_confidence.sum().clamp_min(
                1e-8
            )
        )

        anchor_center = (
            anchor_features
            * anchor_weights.unsqueeze(1)
        ).sum(
            dim=0
        )

        local_center = z[
            local_indices
        ].mean(
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

        support_counts[
            class_id
        ] = len(local_indices)

    return (
        prototypes,
        anchors,
        anchor_counts,
        support_counts
    )


def build_hybrid_prototypes(
    source_prototypes_cpu,
    local_prototypes,
    local_support,
    epoch
):
    hybrid = source_prototypes_cpu.clone()

    progress = (
        epoch
        / max(
            EPOCHS,
            1
        )
    )

    global_mix = (
        LOCAL_MIX_START
        + (
            LOCAL_MIX_MAX
            - LOCAL_MIX_START
        )
        * progress
    )

    for class_id in range(
        NUM_CLASSES
    ):
        if (
            local_support[
                class_id
            ].item()
            <= 0
        ):
            continue

        mix = global_mix

        # Truck receives more local influence because
        # source-prototype diagnostics showed severe mismatch.
        if class_id == TRUCK_ID:
            mix = min(
                LOCAL_MIX_MAX,
                global_mix * 1.25
            )

        combined = (
            (
                1.0
                - mix
            )
            * source_prototypes_cpu[
                class_id
            ]
            + mix
            * local_prototypes[
                class_id
            ]
        )

        hybrid[
            class_id
        ] = F.normalize(
            combined,
            dim=0
        )

    return hybrid


# ============================================================
# GEOMETRY GATE
# ============================================================

@torch.no_grad()
def geometry_gate(
    state,
    prototypes
):
    z = state[
        "z"
    ]

    predictions = state[
        "predictions"
    ]

    confidence = state[
        "confidence"
    ]

    prototypes = F.normalize(
        prototypes,
        dim=1
    )

    similarities = (
        z
        @ prototypes.t()
    )

    rows = torch.arange(
        len(z)
    )

    predicted_similarity = (
        similarities[
            rows,
            predictions
        ]
    )

    sorted_values = torch.sort(
        similarities,
        dim=1,
        descending=True
    ).values

    second_similarity = (
        sorted_values[:, 1]
    )

    margin = (
        predicted_similarity
        - second_similarity
    )

    gate = (
        (confidence >= PL_THRESHOLD)
        & (
            predicted_similarity
            >= LOCAL_SIMILARITY_THRESHOLD
        )
        & (
            margin
            >= LOCAL_MARGIN_THRESHOLD
        )
    )

    return {
        "gate":
            gate,
        "similarities":
            similarities,
        "predicted_similarity":
            predicted_similarity,
        "margin":
            margin
    }


def enforce_global_pl_cap(
    candidate_mask,
    confidence
):
    candidate_indices = torch.nonzero(
        candidate_mask,
        as_tuple=False
    ).flatten()

    max_selected = int(
        len(candidate_mask)
        * GLOBAL_PL_FRACTION
    )

    max_selected = max(
        1,
        max_selected
    )

    if len(candidate_indices) <= max_selected:
        selected = torch.zeros_like(
            candidate_mask
        )

        selected[
            candidate_indices
        ] = True

        return selected

    ranking = torch.argsort(
        confidence[
            candidate_indices
        ],
        descending=True
    )

    selected_indices = (
        candidate_indices[
            ranking[:max_selected]
        ]
    )

    selected = torch.zeros_like(
        candidate_mask
    )

    selected[
        selected_indices
    ] = True

    return selected


# ============================================================
# MCD TRAINING STEP
# ============================================================

def mcd_step(
    student,
    source_x,
    source_y,
    target_x,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    # --------------------------------------------------------
    # Explicit raw-feature assertions
    # --------------------------------------------------------

    if source_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Source input expected {INPUT_DIM}-D, "
            f"got {source_x.shape[1]}"
        )

    if target_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Target input expected {INPUT_DIM}-D, "
            f"got {target_x.shape[1]}"
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

    # --------------------------------------------------------
    # Step A: source supervised adapter update
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Step B: classifier maximize discrepancy
    # --------------------------------------------------------

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
            "Source adapted representation dimension error"
        )

    if target_z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            "Target adapted representation dimension error"
        )

    source_logits1 = (
        student.classifier1(
            source_z
        )
    )

    source_logits2 = (
        student.classifier2(
            source_z
        )
    )

    target_logits1 = (
        student.classifier1(
            target_z
        )
    )

    target_logits2 = (
        student.classifier2(
            target_z
        )
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

    classifier_objective = (
        source_classifier_loss
        - target_discrepancy
    )

    classifier_objective.backward()

    optimizer_classifier.step()

    # --------------------------------------------------------
    # Step C: adapter minimize discrepancy
    # --------------------------------------------------------

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

    target_logits1 = (
        student.classifier1(
            target_z
        )
    )

    target_logits2 = (
        student.classifier2(
            target_z
        )
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

    source_correct = int(
        (
            source_predictions
            == source_y
        ).sum().item()
    )

    source_total = (
        source_y.shape[0]
    )

    return {
        "source_loss":
            float(
                source_loss.item()
            ),
        "discrepancy":
            float(
                target_discrepancy.item()
            ),
        "source_correct":
            source_correct,
        "source_total":
            source_total
    }


# ============================================================
# PSEUDO-LABEL TRAINING
# ============================================================

def train_pseudo_batch(
    student,
    raw_x,
    pseudo_y,
    confidence,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    # --------------------------------------------------------
    # CRITICAL:
    # raw_x MUST be 2048-D.
    # Never pass adapted 512-D features here.
    # --------------------------------------------------------

    if raw_x.ndim != 2:
        raise RuntimeError(
            f"Pseudo-label batch must be 2-D, "
            f"got {tuple(raw_x.shape)}"
        )

    if raw_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Pseudo-label training expects "
            f"{INPUT_DIM}-D raw features, "
            f"got {raw_x.shape[1]}"
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

    # --------------------------------------------------------
    # Adapter update
    # --------------------------------------------------------

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    z = student.encode(
        raw_x
    )

    if z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            f"Pseudo batch adapter output "
            f"must be {HIDDEN_DIM}-D, "
            f"got {z.shape[1]}"
        )

    logits1 = student.classifier1(
        z
    )

    logits2 = student.classifier2(
        z
    )

    loss1 = F.cross_entropy(
        logits1,
        pseudo_y,
        reduction="none"
    )

    loss2 = F.cross_entropy(
        logits2,
        pseudo_y,
        reduction="none"
    )

    weighted_loss = (
        loss1
        + loss2
    ) * confidence

    loss = weighted_loss.mean()

    (
        PL_WEIGHT
        * loss
    ).backward()

    optimizer_adapter.step()

    # --------------------------------------------------------
    # Classifier update
    # --------------------------------------------------------

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():
        z = student.encode(
            raw_x
        )

    if z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            "Pseudo classifier path received "
            "invalid adapted dimension"
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


# ============================================================
# EVALUATION
# ============================================================

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
    total_count = 0

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
        x = x.to(
            device
        )

        y = y.to(
            device
        )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Evaluation received invalid raw dimension"
            )

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
                == y
            ).sum().item()
        )

        total_count += (
            len(y)
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

            if not mask.any():
                continue

            class_count = int(
                mask.sum().item()
            )

            class_total[
                class_id
            ] += class_count

            class_correct[
                class_id
            ] += int(
                (
                    predictions[
                        mask
                    ]
                    == y[
                        mask
                    ]
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


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

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
        "VISDA-2017 MCD + TARGET-LOCAL PROTOTYPE ADAPTATION"
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
        f"local_k={LOCAL_K}"
    )

    print(
        f"global_pl_fraction={GLOBAL_PL_FRACTION}"
    )

    print(
        f"anchor_confidence={ANCHOR_CONFIDENCE}"
    )

    print(
        f"local_mix_max={LOCAL_MIX_MAX}"
    )

    print()

    # --------------------------------------------------------
    # Load caches
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load KNN graph once
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    student = MCDModel().to(
        device
    )

    teacher = MCDModel().to(
        device
    )

    source_prototypes_cpu = load_base_checkpoint(
        student,
        teacher
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

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Best checkpoints
    # --------------------------------------------------------

    best_student_score = (
        base_metrics[
            "mean_class_accuracy"
        ]
    )

    best_teacher_score = (
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

    history = []

    total_start = time.perf_counter()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

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

        # ----------------------------------------------------
        # Build target-local prototypes using CURRENT state
        # ----------------------------------------------------

        target_state = collect_target_state(
            student,
            teacher,
            target_features,
            device
        )

        # Keep RAW target features separately.
        #
        # target_state["z"] is 512-D.
        # target_features is 2048-D.
        #
        # This separation prevents the exact bug from the
        # previous implementation.

        local_prototypes, anchor_mask, anchor_counts, local_support = (
            build_target_local_prototypes(
                target_state["z"],
                target_state["confidence"],
                target_state["predictions"],
                knn_indices
            )
        )

        hybrid_prototypes_cpu = (
            build_hybrid_prototypes(
                source_prototypes_cpu,
                local_prototypes,
                local_support,
                epoch
            )
        )

        hybrid_prototypes = (
            hybrid_prototypes_cpu.to(
                device
            )
        )

        # ----------------------------------------------------
        # Source geometry gate
        # ----------------------------------------------------

        source_geometry = geometry_gate(
            target_state,
            source_prototypes
        )

        # ----------------------------------------------------
        # Hybrid geometry gate
        # ----------------------------------------------------

        hybrid_geometry = geometry_gate(
            target_state,
            hybrid_prototypes
        )

        # ----------------------------------------------------
        # Global 25% cap
        # ----------------------------------------------------

        selected = enforce_global_pl_cap(
            hybrid_geometry["gate"],
            target_state["confidence"]
        )

        selected_indices = torch.nonzero(
            selected,
            as_tuple=False
        ).flatten()

        if len(selected_indices) > 0:
            selected_order = torch.argsort(
                target_state[
                    "confidence"
                ][
                    selected_indices
                ],
                descending=True
            )

            selected_indices = (
                selected_indices[
                    selected_order
                ]
            )

        selected_labels = (
            target_state[
                "predictions"
            ][
                selected_indices
            ]
        )

        selected_confidence = (
            target_state[
                "confidence"
            ][
                selected_indices
            ]
        )

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
                "GLOBAL PL CAP VIOLATION: "
                f"{selected_fraction:.6f}"
            )

        # ----------------------------------------------------
        # Selected class distribution
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Diagnostics for truck
        # ----------------------------------------------------

        truck_source_similarity = float(
            source_geometry[
                "similarities"
            ][
                :,
                TRUCK_ID
            ].mean().item()
        )

        truck_hybrid_similarity = float(
            hybrid_geometry[
                "similarities"
            ][
                :,
                TRUCK_ID
            ].mean().item()
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        student.train()
        teacher.eval()

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

        pseudo_pointer = 0

        steps = len(
            source_loader
        )

        for _ in range(
            steps
        ):
            # -----------------------------------------------
            # Raw source batch
            # -----------------------------------------------

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

            # -----------------------------------------------
            # Raw target batch
            # -----------------------------------------------

            try:
                target_batch = next(
                    target_iterator
                )[0]
            except StopIteration:
                target_iterator = iter(
                    target_loader
                )

                target_batch = next(
                    target_iterator
                )[0]

            # ------------------------------------------------
            # CRITICAL RAW DIMENSION ASSERTIONS
            # ------------------------------------------------

            if source_x.shape[1] != INPUT_DIM:
                raise RuntimeError(
                    f"Source raw batch must be "
                    f"{INPUT_DIM}-D, got {source_x.shape[1]}"
                )

            if target_batch.shape[1] != INPUT_DIM:
                raise RuntimeError(
                    f"Target raw batch must be "
                    f"{INPUT_DIM}-D, got {target_batch.shape[1]}"
                )

            # ------------------------------------------------
            # MCD
            # ------------------------------------------------

            step_metrics = mcd_step(
                student,
                source_x,
                source_y,
                target_batch,
                optimizer_adapter,
                optimizer_classifier,
                device
            )

            source_correct += (
                step_metrics[
                    "source_correct"
                ]
            )

            source_total += (
                step_metrics[
                    "source_total"
                ]
            )

            source_loss_sum += (
                step_metrics[
                    "source_loss"
                ]
            )

            discrepancy_sum += (
                step_metrics[
                    "discrepancy"
                ]
            )

            update_ema(
                teacher,
                student
            )

            # ------------------------------------------------
            # PSEUDO-LABEL TRAINING
            #
            # selected_indices points into target_features,
            # which are 2048-D RAW features.
            # ------------------------------------------------

            if (
                pseudo_pointer
                < selected_count
            ):
                end = min(
                    pseudo_pointer
                    + BATCH_SIZE,
                    selected_count
                )

                batch_indices = (
                    selected_indices[
                        pseudo_pointer:end
                    ]
                )

                pseudo_x_raw = (
                    target_features[
                        batch_indices
                    ]
                )

                pseudo_y = (
                    target_state[
                        "predictions"
                    ][
                        batch_indices
                    ]
                )

                pseudo_confidence = (
                    target_state[
                        "confidence"
                    ][
                        batch_indices
                    ]
                )

                # Explicitly verify this is still RAW 2048-D.
                if pseudo_x_raw.shape[1] != INPUT_DIM:
                    raise RuntimeError(
                        "Pseudo-label training received "
                        "non-raw target features"
                    )

                pseudo_loss = train_pseudo_batch(
                    student,
                    pseudo_x_raw,
                    pseudo_y,
                    pseudo_confidence,
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

        scheduler_adapter.step()
        scheduler_classifier.step()

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Best models
        # ----------------------------------------------------

        if (
            student_metrics[
                "mean_class_accuracy"
            ]
            > best_student_score
        ):
            best_student_score = (
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
            > best_teacher_score
        ):
            best_teacher_score = (
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

        record = {
            "epoch":
                epoch,
            "selected_pl_count":
                selected_count,
            "selected_pl_fraction":
                selected_fraction,
            "selected_counts":
                selected_counts.tolist(),
            "selected_fractions":
                selected_fractions.tolist(),
            "anchor_counts":
                anchor_counts.tolist(),
            "local_support":
                local_support.tolist(),
            "source_geometry_coverage":
                float(
                    source_geometry[
                        "gate"
                    ].float().mean().item()
                ),
            "hybrid_geometry_coverage":
                float(
                    hybrid_geometry[
                        "gate"
                    ].float().mean().item()
                ),
            "truck_source_similarity":
                truck_source_similarity,
            "truck_hybrid_similarity":
                truck_hybrid_similarity,
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
            "epoch_seconds":
                epoch_seconds
        }

        history.append(
            record
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

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
            f"Source geometry coverage: "
            f"{100.0 * record['source_geometry_coverage']:.2f}%"
        )

        print(
            f"Hybrid geometry coverage: "
            f"{100.0 * record['hybrid_geometry_coverage']:.2f}%"
        )

        print(
            f"Selected PL: "
            f"{selected_count} "
            f"({100.0 * selected_fraction:.2f}%)"
        )

        print(
            f"Truck source similarity: "
            f"{truck_source_similarity:.6f}"
        )

        print(
            f"Truck hybrid similarity: "
            f"{truck_hybrid_similarity:.6f}"
        )

        print()
        print(
            "Selected PL distribution:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"  {CLASSES[class_id]:12s}: "
                f"{int(selected_counts[class_id].item()):5d} "
                f"({100.0 * selected_fractions[class_id].item():6.2f}%)"
            )

        print()
        print(
            "Target-local prototype support:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"  {CLASSES[class_id]:12s}: "
                f"{int(local_support[class_id].item())}"
            )

        print(
            f"Epoch time: "
            f"{epoch_seconds:.2f}s"
        )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

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

    # ========================================================
    # SAVE
    # ========================================================

    torch.save(
        {
            "student_state_dict":
                student.state_dict(),
            "teacher_state_dict":
                teacher.state_dict(),
            "source_prototypes":
                source_prototypes_cpu,
            "base_checkpoint":
                str(BASE_CHECKPOINT),
            "knn_graph":
                str(KNN_GRAPH),
            "seed":
                SEED,
            "local_k":
                LOCAL_K,
            "global_pl_fraction":
                GLOBAL_PL_FRACTION,
            "pl_threshold":
                PL_THRESHOLD,
            "anchor_confidence":
                ANCHOR_CONFIDENCE,
            "anchor_top_fraction":
                ANCHOR_TOP_FRACTION,
            "max_anchors_per_class":
                MAX_ANCHORS_PER_CLASS,
            "local_mix_max":
                LOCAL_MIX_MAX,
            "best_student_epoch":
                best_student_epoch,
            "best_student_mean_class":
                best_student_score,
            "best_teacher_epoch":
                best_teacher_epoch,
            "best_teacher_mean_class":
                best_teacher_score,
            "final_student":
                final_student,
            "final_teacher":
                final_teacher,
            "history":
                history
        },
        OUTPUT_CHECKPOINT
    )

    report = {
        "experiment":
            "visda_mcd_target_local_prototype_adaptation",
        "seed":
            SEED,
        "base_checkpoint":
            str(BASE_CHECKPOINT),
        "knn_graph":
            str(KNN_GRAPH),
        "local_k":
            LOCAL_K,
        "pl_threshold":
            PL_THRESHOLD,
        "global_pl_fraction":
            GLOBAL_PL_FRACTION,
        "anchor_confidence":
            ANCHOR_CONFIDENCE,
        "anchor_top_fraction":
            ANCHOR_TOP_FRACTION,
        "max_anchors_per_class":
            MAX_ANCHORS_PER_CLASS,
        "local_mix_start":
            LOCAL_MIX_START,
        "local_mix_max":
            LOCAL_MIX_MAX,
        "base_metrics":
            base_metrics,
        "best_student_epoch":
            best_student_epoch,
        "best_student_mean_class":
            best_student_score,
        "best_teacher_epoch":
            best_teacher_epoch,
        "best_teacher_mean_class":
            best_teacher_score,
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

    # ========================================================
    # FINAL LOG
    # ========================================================

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
        f"{best_student_score:.2f}%"
    )

    print(
        f"Best teacher epoch: "
        f"{best_teacher_epoch}"
    )

    print(
        f"Best teacher mean-class: "
        f"{best_teacher_score:.2f}%"
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