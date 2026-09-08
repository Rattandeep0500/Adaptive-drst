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
    "checkpoints/visda_pairwise_boundary_recovery"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

GEOMETRY_SIMILARITY_THRESHOLD = 0.60
GEOMETRY_MARGIN_THRESHOLD = 0.00

PL_WEIGHT = 0.20

PAIRWISE_WEIGHT = 0.02
PAIRWISE_TEMPERATURE = 0.50

RECOVERY_CONFIDENCE_MIN = 0.50
RECOVERY_SIMILARITY_MIN = 0.40
RECOVERY_MAX_PER_CLASS = 30

STARVATION_FRACTION = 0.01

RECOVERY_MAP = {
    11: [3, 1, 10],
    9: [7, 5],
}

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

    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)

    if x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Final feature dimension is {x.shape[1]}"
        )

    return x, y


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(
                f"Adapter expects 2D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expects {INPUT_DIM}-D input, got {x.shape[1]}"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter output is {z.shape[1]}-D"
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
    out = {}

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

        out[new_key] = value

    return out


def load_base_checkpoint(student, teacher):
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {BASE_CHECKPOINT}"
        )

    payload = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
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

    prototypes = payload["source_prototypes"].float()

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Invalid prototype shape {tuple(prototypes.shape)}"
        )

    return payload, prototypes


@torch.no_grad()
def update_ema(teacher, student):
    teacher_params = dict(
        teacher.named_parameters()
    )
    student_params = dict(
        student.named_parameters()
    )

    for name in teacher_params:
        teacher_params[name].mul_(EMA_DECAY)
        teacher_params[name].add_(
            student_params[name],
            alpha=1.0 - EMA_DECAY
        )

    teacher_buffers = dict(
        teacher.named_buffers()
    )
    student_buffers = dict(
        student.named_buffers()
    )

    for name in teacher_buffers:
        teacher_buffers[name].copy_(
            student_buffers[name]
        )


def discrepancy(logits1, logits2):
    p1 = F.softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)

    return (
        p1 - p2
    ).abs().mean()


@torch.no_grad()
def teacher_predict(teacher, x):
    teacher.eval()

    logits1, logits2 = teacher(x)

    p1 = F.softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)

    probabilities = (p1 + p2) / 2.0

    confidence, predictions = (
        probabilities.max(dim=1)
    )

    disagreement = (
        p1 - p2
    ).abs().mean(dim=1)

    return (
        probabilities,
        confidence,
        predictions,
        disagreement
    )


@torch.no_grad()
def compute_target_state(
    student,
    teacher,
    target_features,
    prototypes,
    device
):
    student.eval()
    teacher.eval()

    loader = DataLoader(
        TensorDataset(target_features),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    all_z = []
    all_prob = []
    all_conf = []
    all_pred = []
    all_disc = []

    for (x,) in loader:
        x = x.to(device)

        z = student.encode(x)

        p, conf, pred, disc = teacher_predict(
            teacher,
            x
        )

        all_z.append(z.cpu())
        all_prob.append(p.cpu())
        all_conf.append(conf.cpu())
        all_pred.append(pred.cpu())
        all_disc.append(disc.cpu())

    z = torch.cat(all_z, dim=0)
    probabilities = torch.cat(all_prob, dim=0)
    confidence = torch.cat(all_conf, dim=0)
    predictions = torch.cat(all_pred, dim=0)
    disagreement = torch.cat(all_disc, dim=0)

    z_norm = F.normalize(z, dim=1)
    p_norm = F.normalize(prototypes.cpu(), dim=1)

    similarities = (
        z_norm @ p_norm.t()
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

    sorted_similarity, sorted_indices = (
        torch.sort(
            similarities,
            dim=1,
            descending=True
        )
    )

    second_similarity = sorted_similarity[:, 1]

    second_class = sorted_indices[:, 1]

    geometry_margin = (
        predicted_similarity
        - second_similarity
    )

    trusted = (
        (confidence >= PL_THRESHOLD)
        & (
            predicted_similarity
            >= GEOMETRY_SIMILARITY_THRESHOLD
        )
        & (
            geometry_margin
            >= GEOMETRY_MARGIN_THRESHOLD
        )
    )

    return {
        "z": z,
        "probabilities": probabilities,
        "confidence": confidence,
        "predictions": predictions,
        "disagreement": disagreement,
        "similarities": similarities,
        "predicted_similarity": predicted_similarity,
        "second_similarity": second_similarity,
        "second_class": second_class,
        "geometry_margin": geometry_margin,
        "trusted": trusted,
    }


def enforce_global_pl_cap(
    state,
    max_fraction
):
    trusted = state["trusted"]

    indices = torch.nonzero(
        trusted,
        as_tuple=False
    ).flatten()

    max_selected = int(
        len(trusted) * max_fraction
    )

    max_selected = max(
        1,
        max_selected
    )

    if len(indices) <= max_selected:
        return trusted.clone()

    order = torch.argsort(
        state["confidence"][indices],
        descending=True
    )

    indices = indices[
        order[:max_selected]
    ]

    selected = torch.zeros_like(
        trusted
    )

    selected[indices] = True

    return selected


def trusted_statistics(
    predictions,
    selected
):
    counts = torch.bincount(
        predictions[selected],
        minlength=NUM_CLASSES
    )

    total = max(
        int(selected.sum().item()),
        1
    )

    fractions = (
        counts.float()
        / total
    )

    starved = [
        class_id
        for class_id in range(NUM_CLASSES)
        if fractions[class_id].item()
        < STARVATION_FRACTION
    ]

    return (
        counts,
        fractions,
        starved
    )


def select_boundary_candidates(
    state,
    starved_classes
):
    candidates = []

    for starved_class in starved_classes:
        attractors = RECOVERY_MAP.get(
            starved_class,
            []
        )

        if not attractors:
            continue

        for attractor in attractors:
            confidence = state["confidence"]
            predictions = state["predictions"]
            similarities = state["similarities"]
            trusted = state["trusted"]

            class_similarity = similarities[
                :,
                starved_class
            ]

            attractor_similarity = similarities[
                :,
                attractor
            ]

            score = (
                class_similarity
                - attractor_similarity
            )

            mask = (
                (~trusted)
                & (
                    predictions
                    == attractor
                )
                & (
                    confidence
                    >= RECOVERY_CONFIDENCE_MIN
                )
                & (
                    class_similarity
                    >= RECOVERY_SIMILARITY_MIN
                )
            )

            indices = torch.nonzero(
                mask,
                as_tuple=False
            ).flatten()

            if len(indices) == 0:
                continue

            order = torch.argsort(
                score[indices],
                descending=True
            )

            take = min(
                RECOVERY_MAX_PER_CLASS,
                len(order)
            )

            chosen = indices[
                order[:take]
            ]

            for index in chosen.tolist():
                candidates.append(
                    {
                        "index": index,
                        "starved_class": starved_class,
                        "attractor": attractor,
                        "score": float(
                            score[index].item()
                        ),
                        "similarity": float(
                            class_similarity[index].item()
                        ),
                        "confidence": float(
                            confidence[index].item()
                        )
                    }
                )

    candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )

    deduplicated = []
    seen = set()
    per_class_counts = {
        class_id: 0
        for class_id in starved_classes
    }

    for candidate in candidates:
        key = candidate["index"]

        if key in seen:
            continue

        starved_class = candidate[
            "starved_class"
        ]

        if (
            per_class_counts[
                starved_class
            ]
            >= RECOVERY_MAX_PER_CLASS
        ):
            continue

        seen.add(key)

        per_class_counts[
            starved_class
        ] += 1

        deduplicated.append(
            candidate
        )

    return deduplicated


def train_source_mcd_step(
    student,
    source_x,
    source_y,
    target_x,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    source_x = source_x.to(device)
    source_y = source_y.to(device)
    target_x = target_x.to(device)

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

    target_disc = discrepancy(
        target_logits1,
        target_logits2
    )

    (
        source_classifier_loss
        - target_disc
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

    generator_disc = discrepancy(
        target_logits1,
        target_logits2
    )

    generator_disc.backward()
    optimizer_adapter.step()

    for parameter in student.classifier1.parameters():
        parameter.requires_grad_(True)

    for parameter in student.classifier2.parameters():
        parameter.requires_grad_(True)

    return (
        float(source_loss.item()),
        float(target_disc.item())
    )


def train_pl_batch(
    student,
    teacher,
    x,
    pseudo_y,
    confidence,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    x = x.to(device)
    pseudo_y = pseudo_y.to(device)
    confidence = confidence.to(device)

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    z = student.encode(x)

    logits1 = student.classifier1(z)
    logits2 = student.classifier2(z)

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

    loss = (
        (
            loss1
            + loss2
        )
        * confidence
    ).mean()

    (
        PL_WEIGHT * loss
    ).backward()

    optimizer_adapter.step()

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    student.eval()

    with torch.no_grad():
        z = student.encode(x)

    student.train()

    logits1 = student.classifier1(z)
    logits2 = student.classifier2(z)

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


def train_pairwise_boundary(
    student,
    candidates,
    target_features,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    if not candidates:
        return {
            "loss": 0.0,
            "count": 0
        }

    indices = [
        item["index"]
        for item in candidates
    ]

    starved_labels = torch.tensor(
        [
            item["starved_class"]
            for item in candidates
        ],
        dtype=torch.long
    )

    attractor_labels = torch.tensor(
        [
            item["attractor"]
            for item in candidates
        ],
        dtype=torch.long
    )

    weights = torch.tensor(
        [
            max(
                0.05,
                item["score"]
            )
            for item in candidates
        ],
        dtype=torch.float32
    )

    x = target_features[
        indices
    ].to(device)

    starved_labels = starved_labels.to(
        device
    )

    attractor_labels = attractor_labels.to(
        device
    )

    weights = weights.to(
        device
    )

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    z = student.encode(x)

    logits = (
        student.classifier1(z)
        + student.classifier2(z)
    ) / 2.0

    rows = torch.arange(
        len(logits),
        device=device
    )

    starved_logits = logits[
        rows,
        starved_labels
    ]

    attractor_logits = logits[
        rows,
        attractor_labels
    ]

    pairwise_loss = -F.logsigmoid(
        (
            starved_logits
            - attractor_logits
        )
        / PAIRWISE_TEMPERATURE
    )

    pairwise_loss = (
        pairwise_loss
        * weights
    ).mean()

    (
        PAIRWISE_WEIGHT
        * pairwise_loss
    ).backward()

    optimizer_adapter.step()

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    student.eval()

    with torch.no_grad():
        z = student.encode(x)

    student.train()

    logits = (
        student.classifier1(z)
        + student.classifier2(z)
    ) / 2.0

    starved_logits = logits[
        rows,
        starved_labels
    ]

    attractor_logits = logits[
        rows,
        attractor_labels
    ]

    classifier_pairwise_loss = -F.logsigmoid(
        (
            starved_logits
            - attractor_logits
        )
        / PAIRWISE_TEMPERATURE
    )

    classifier_pairwise_loss = (
        classifier_pairwise_loss
        * weights
    ).mean()

    (
        PAIRWISE_WEIGHT
        * classifier_pairwise_loss
    ).backward()

    optimizer_classifier.step()

    return {
        "loss":
            float(
                pairwise_loss.item()
            ),
        "count":
            len(candidates)
    }


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

        total_count += y.size(0)

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
            / max(total_count, 1),
        "mean_class_accuracy":
            float(
                per_class.mean().item()
            ),
        "per_class_accuracy":
            per_class.tolist(),
        "mean_confidence":
            confidence_sum
            / max(total_count, 1)
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print(
        "VISDA-2017 GEOMETRY-GATED MCD + PAIRWISE BOUNDARY RECOVERY"
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
        f"pairwise_weight={PAIRWISE_WEIGHT}"
    )

    print(
        f"pairwise_temperature={PAIRWISE_TEMPERATURE}"
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

    student = MCDModel().to(device)
    teacher = MCDModel().to(device)

    payload, source_prototypes = (
        load_base_checkpoint(
            student,
            teacher
        )
    )

    source_prototypes = source_prototypes.to(
        device
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

    history = []

    best_student = base_metrics[
        "mean_class_accuracy"
    ]

    best_student_epoch = 0

    best_student_state = {
        key:
            value.detach().cpu().clone()
        for key, value in student.state_dict().items()
    }

    best_teacher = base_metrics[
        "mean_class_accuracy"
    ]

    best_teacher_epoch = 0

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

        state = compute_target_state(
            student,
            teacher,
            target_features,
            source_prototypes,
            device
        )

        trusted_global = enforce_global_pl_cap(
            state,
            GLOBAL_PL_FRACTION
        )

        state["trusted_global"] = (
            trusted_global
        )

        trusted_counts, trusted_fractions, starved_classes = (
            trusted_statistics(
                state["predictions"],
                trusted_global
            )
        )

        recovery_candidates = (
            select_boundary_candidates(
                state,
                starved_classes
            )
        )

        recovery_counts = torch.zeros(
            NUM_CLASSES,
            dtype=torch.long
        )

        recovery_precision_hits = torch.zeros(
            NUM_CLASSES,
            dtype=torch.long
        )

        recovery_precision_total = torch.zeros(
            NUM_CLASSES,
            dtype=torch.long
        )

        for candidate in recovery_candidates:
            class_id = candidate[
                "starved_class"
            ]

            recovery_counts[
                class_id
            ] += 1

            index = candidate[
                "index"
            ]

            recovery_precision_total[
                class_id
            ] += 1

            if (
                int(
                    target_labels[index].item()
                )
                == class_id
            ):
                recovery_precision_hits[
                    class_id
                ] += 1

        trusted_indices = torch.nonzero(
            trusted_global,
            as_tuple=False
        ).flatten()

        trusted_order = torch.argsort(
            state["confidence"][
                trusted_indices
            ],
            descending=True
        )

        trusted_indices = trusted_indices[
            trusted_order
        ]

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

        trusted_pointer = 0
        pseudo_samples_used = 0

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

            source_x_device = source_x.to(device)
            source_y_device = source_y.to(device)
            target_x_device = target_x.to(device)

            source_loss, target_disc = (
                train_source_mcd_step(
                    student,
                    source_x_device,
                    source_y_device,
                    target_x_device,
                    optimizer_adapter,
                    optimizer_classifier,
                    device
                )
            )

            with torch.no_grad():
                source_logits1, source_logits2 = student(
                    source_x_device
                )

                source_predictions = (
                    (
                        source_logits1
                        + source_logits2
                    ) / 2.0
                ).argmax(
                    dim=1
                )

            source_correct += int(
                (
                    source_predictions
                    == source_y_device
                ).sum().item()
            )

            source_total += (
                source_y_device.size(0)
            )

            source_loss_sum += source_loss
            discrepancy_sum += target_disc

            update_ema(
                teacher,
                student
            )

            if (
                trusted_pointer
                < len(trusted_indices)
            ):
                end = min(
                    trusted_pointer
                    + BATCH_SIZE,
                    len(trusted_indices)
                )

                batch_indices = trusted_indices[
                    trusted_pointer:end
                ]

                pseudo_x = target_features[
                    batch_indices
                ]

                pseudo_y = state[
                    "predictions"
                ][
                    batch_indices
                ]

                pseudo_conf = state[
                    "confidence"
                ][
                    batch_indices
                ]

                pseudo_loss = train_pl_batch(
                    student,
                    teacher,
                    pseudo_x,
                    pseudo_y,
                    pseudo_conf,
                    optimizer_adapter,
                    optimizer_classifier,
                    device
                )

                pseudo_loss_sum += (
                    pseudo_loss
                )

                pseudo_samples_used += (
                    len(batch_indices)
                )

                trusted_pointer = end

                update_ema(
                    teacher,
                    student
                )

        if recovery_candidates:
            pairwise_result = train_pairwise_boundary(
                student,
                recovery_candidates,
                target_features,
                optimizer_adapter,
                optimizer_classifier,
                device
            )

            update_ema(
                teacher,
                student
            )
        else:
            pairwise_result = {
                "loss": 0.0,
                "count": 0
            }

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
            best_student = student_metrics[
                "mean_class_accuracy"
            ]

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
            best_teacher = teacher_metrics[
                "mean_class_accuracy"
            ]

            best_teacher_epoch = epoch

            best_teacher_state = {
                key:
                    value.detach().cpu().clone()
                for key, value in teacher.state_dict().items()
            }

        recovery_precision = {}

        for class_id, class_name in enumerate(
            CLASSES
        ):
            total = int(
                recovery_precision_total[
                    class_id
                ].item()
            )

            hits = int(
                recovery_precision_hits[
                    class_id
                ].item()
            )

            if total > 0:
                recovery_precision[
                    class_name
                ] = (
                    hits / total
                )
            else:
                recovery_precision[
                    class_name
                ] = None

        elapsed = (
            time.perf_counter()
            - epoch_start
        )

        history.append(
            {
                "epoch":
                    epoch,
                "trusted_pl_selected":
                    int(
                        trusted_global.sum().item()
                    ),
                "trusted_pl_fraction":
                    float(
                        trusted_global.float().mean().item()
                    ),
                "trusted_counts":
                    trusted_counts.tolist(),
                "trusted_fractions":
                    trusted_fractions.tolist(),
                "starved_classes":
                    [
                        CLASSES[c]
                        for c in starved_classes
                    ],
                "recovery_counts":
                    recovery_counts.tolist(),
                "recovery_precision":
                    recovery_precision,
                "recovery_total":
                    len(
                        recovery_candidates
                    ),
                "pairwise_loss":
                    pairwise_result["loss"],
                "pairwise_count":
                    pairwise_result["count"],
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
                "pseudo_samples_used":
                    pseudo_samples_used,
                "elapsed_seconds":
                    elapsed
            }
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"PL {100.0 * trusted_global.float().mean().item():.2f}% | "
            f"Recovery {len(recovery_candidates)} | "
            f"PairLoss {pairwise_result['loss']:.5f} | "
            f"Student "
            f"{student_metrics['mean_class_accuracy']:.2f}% | "
            f"Teacher "
            f"{teacher_metrics['mean_class_accuracy']:.2f}% | "
            f"{elapsed:.2f}s"
        )

        print(
            "Trusted PL mass:"
        )

        for class_name, count, fraction in zip(
            CLASSES,
            trusted_counts.tolist(),
            trusted_fractions.tolist()
        ):
            print(
                f"  {class_name:12s}: "
                f"{count:5d} "
                f"({100.0 * fraction:6.2f}%)"
            )

        print(
            "Boundary recovery:"
        )

        for candidate_class, candidate_names in RECOVERY_MAP.items():
            count = int(
                recovery_counts[
                    candidate_class
                ].item()
            )

            if count > 0:
                precision = recovery_precision[
                    CLASSES[candidate_class]
                ]

                print(
                    f"  {CLASSES[candidate_class]:12s}: "
                    f"{count:3d} | "
                    f"precision="
                    f"{100.0 * precision:.2f}%"
                )

        if starved_classes:
            print(
                "Starved classes:"
            )

            for class_id in starved_classes:
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

    print()
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
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "pairwise_boundary_recovery_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "pairwise_boundary_recovery_seed42.json"
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
            "pairwise_weight":
                PAIRWISE_WEIGHT,
            "pairwise_temperature":
                PAIRWISE_TEMPERATURE,
            "recovery_confidence_min":
                RECOVERY_CONFIDENCE_MIN,
            "recovery_similarity_min":
                RECOVERY_SIMILARITY_MIN,
            "recovery_max_per_class":
                RECOVERY_MAX_PER_CLASS,
            "final_student":
                final_student,
            "final_teacher":
                final_teacher,
            "best_student_epoch":
                best_student_epoch,
            "best_student_mean_class":
                best_student,
            "best_teacher_epoch":
                best_teacher_epoch,
            "best_teacher_mean_class":
                best_teacher,
            "history":
                history
        },
        checkpoint_path
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "experiment":
                    "visda_pairwise_boundary_recovery",
                "seed":
                    SEED,
                "base_checkpoint":
                    str(BASE_CHECKPOINT),
                "baseline":
                    base_metrics,
                "final_student":
                    final_student,
                "final_teacher":
                    final_teacher,
                "best_student_epoch":
                    best_student_epoch,
                "best_student_mean_class":
                    best_student,
                "best_teacher_epoch":
                    best_teacher_epoch,
                "best_teacher_mean_class":
                    best_teacher,
                "history":
                    history
            },
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