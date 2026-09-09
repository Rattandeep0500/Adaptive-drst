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

EPOCHS = 2

LR_ADAPTER = 0.0005
LR_CLASSIFIER = 0.005

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/"
    "geometry_gated_mcd_seed42.pt"
)

CACHE_ROOT = Path(
    "checkpoints/visda_feature_cache"
)

SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

OUTPUT_DIR = Path(
    "checkpoints/visda_recipient_promotion_capping"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CHECKPOINT = (
    OUTPUT_DIR
    / "recipient_promotion_capping_seed42.pt"
)

OUTPUT_HISTORY = (
    OUTPUT_DIR
    / "recipient_promotion_capping_seed42.json"
)

PL_THRESHOLD = 0.90
GLOBAL_PL_FRACTION = 0.25
PL_WEIGHT = 0.20

GEOMETRY_SIMILARITY_THRESHOLD = 0.60
GEOMETRY_MARGIN_THRESHOLD = 0.00

RPC_KAPPA = 2.0
RPC_EPS = 1e-12

MATCH_NORM_GLOBAL_ATTENUATION = True

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10

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

    feature_parts = []
    label_parts = []

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

        x = payload["features"].float()
        y = payload["labels"].long()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature rank in {path}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features in {path}, "
                f"got {x.shape[1]}"
            )

        if len(x) != len(y):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
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
                f"Adapter expects 2-D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expects {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter output must be {HIDDEN_DIM}-D, "
                f"got {z.shape[1]}"
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
                f"Model expects 2-D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Model expects {INPUT_DIM}-D input, "
                f"got {x.shape[1]}"
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


def load_checkpoint(
    student,
    teacher
):
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{BASE_CHECKPOINT}"
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

    if "source_prototypes" not in payload:
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
            f"Invalid source prototype shape: "
            f"{tuple(source_prototypes.shape)}"
        )

    return F.normalize(
        source_prototypes,
        dim=1
    )


@torch.no_grad()
def update_ema(
    teacher,
    student,
    decay=0.97
):
    teacher_params = dict(
        teacher.named_parameters()
    )

    student_params = dict(
        student.named_parameters()
    )

    for name in teacher_params:
        teacher_params[name].mul_(
            decay
        )

        teacher_params[name].add_(
            student_params[name],
            alpha=1.0 - decay
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


@torch.no_grad()
def collect_target_state(
    teacher,
    student,
    target_features,
    device
):
    teacher.eval()
    student.eval()

    loader = DataLoader(
        target_features,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    adapted_parts = []
    probability_parts = []
    prediction_parts = []
    confidence_parts = []

    for x in loader:
        x = x.to(
            device
        )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Target input is not 2048-D"
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

        confidence, prediction = (
            p.max(
                dim=1
            )
        )

        adapted_parts.append(
            F.normalize(
                z,
                dim=1
            ).cpu()
        )

        probability_parts.append(
            p.cpu()
        )

        prediction_parts.append(
            prediction.cpu()
        )

        confidence_parts.append(
            confidence.cpu()
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
        "predictions":
            torch.cat(
                prediction_parts,
                dim=0
            ),
        "confidence":
            torch.cat(
                confidence_parts,
                dim=0
            )
    }


@torch.no_grad()
def build_local_prototypes(
    z,
    predictions,
    confidence
):
    prototypes = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM
    )

    support = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            (predictions == class_id)
            & (
                confidence >= PL_THRESHOLD
            )
        )

        indices = torch.nonzero(
            mask,
            as_tuple=False
        ).flatten()

        if len(indices) == 0:
            continue

        if len(indices) > 2000:
            order = torch.argsort(
                confidence[
                    indices
                ],
                descending=True
            )

            indices = indices[
                order[:2000]
            ]

        class_z = z[
            indices
        ]

        prototype = class_z.mean(
            dim=0
        )

        prototypes[
            class_id
        ] = F.normalize(
            prototype,
            dim=0
        )

        support[
            class_id
        ] = len(indices)

    return (
        prototypes,
        support
    )


@torch.no_grad()
def geometry_select(
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

    rows = torch.arange(
        len(z)
    )

    predicted_similarity = (
        similarities[
            rows,
            predictions
        ]
    )

    sorted_similarity = torch.sort(
        similarities,
        dim=1,
        descending=True
    ).values

    second_similarity = (
        sorted_similarity[:, 1]
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

    indices = torch.nonzero(
        candidate,
        as_tuple=False
    ).flatten()

    max_selected = int(
        len(z)
        * GLOBAL_PL_FRACTION
    )

    if len(indices) > max_selected:
        order = torch.argsort(
            confidence[
                indices
            ],
            descending=True
        )

        indices = indices[
            order[
                :max_selected
            ]
        ]

    selected = torch.zeros(
        len(z),
        dtype=torch.bool
    )

    selected[
        indices
    ] = True

    return {
        "mask":
            selected,
        "indices":
            indices,
        "similarities":
            similarities,
        "margin":
            margin
    }


def audit_target_promotion_mass(
    model,
    target_features,
    target_predictions,
    device
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            target_features,
            target_predictions
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    promotion_mass = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    suppression_mass = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    total_samples = 0

    for x, _ in loader:
        x = x.to(
            device
        )

        gradients, _ = direct_target_logit_gradient(
            model,
            x
        )

        promotion = F.relu(
            -gradients
        ).sum(
            dim=0
        )

        suppression = F.relu(
            gradients
        ).sum(
            dim=0
        )

        promotion_mass += (
            promotion.cpu()
            .double()
        )

        suppression_mass += (
            suppression.cpu()
            .double()
        )

        total_samples += (
            len(x)
        )

    return (
        promotion_mass,
        suppression_mass,
        total_samples
    )


def direct_target_logit_gradient(
    model,
    x
):
    model.zero_grad(
        set_to_none=True
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

    loss = (
        p1 - p2
    ).abs().mean()

    g1, g2 = torch.autograd.grad(
        loss,
        (
            logits1,
            logits2
        ),
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )

    gradient = (
        g1 + g2
    ) / 2.0

    return (
        gradient.detach(),
        loss.detach()
    )


def build_rpc_alpha(
    promotion_mass
):
    median_mass = torch.median(
        promotion_mass
    )

    target = (
        RPC_KAPPA
        * median_mass
    )

    alpha = torch.minimum(
        torch.ones_like(
            promotion_mass
        ),
        target
        / (
            promotion_mass
            + RPC_EPS
        )
    )

    return (
        alpha,
        median_mass,
        target
    )


def print_rpc_coefficients(
    promotion_mass,
    alpha
):
    print()
    print(
        "FROZEN RPC COEFFICIENTS"
    )

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"P={promotion_mass[class_id].item():.10f} | "
            f"alpha={alpha[class_id].item():.6f}"
        )


def split_target_indices(
    total
):
    indices = torch.arange(
        total
    )

    return indices


def apply_frozen_rpc_gradient(
    model,
    x,
    alpha,
    mode
):
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

    loss = (
        p1 - p2
    ).abs().mean()
    
    parameters = [
        p
        for p in model.adapter.parameters()
        if p.requires_grad
    ]

    gradients_adapter = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=False
    )

    if mode == "baseline":
        return loss

    grad_logits1, grad_logits2 = torch.autograd.grad(
        loss,
        (
            logits1,
            logits2
        ),
        retain_graph=True,
        create_graph=False,
        allow_unused=False
    )

    if mode == "rpc":
        p1_pos = F.relu(
            -grad_logits1
        )

        p2_pos = F.relu(
            -grad_logits2
        )

        s1 = F.relu(
            grad_logits1
        )

        s2 = F.relu(
            grad_logits2
        )

        alpha_device = alpha.to(
            x.device
        )

        clipped_p1 = (
            p1_pos
            * alpha_device.view(
                1,
                -1
            )
        )

        clipped_p2 = (
            p2_pos
            * alpha_device.view(
                1,
                -1
            )
        )

        beta1 = (
            clipped_p1.sum(
                dim=1,
                keepdim=True
            )
            / s1.sum(
                dim=1,
                keepdim=True
            ).clamp_min(
                RPC_EPS
            )
        )

        beta2 = (
            clipped_p2.sum(
                dim=1,
                keepdim=True
            )
            / s2.sum(
                dim=1,
                keepdim=True
            ).clamp_min(
                RPC_EPS
            )
        )

        beta1 = beta1.clamp(
            max=1.0
        )

        beta2 = beta2.clamp(
            max=1.0
        )

        new_g1 = (
            s1 * beta1
            - clipped_p1
        )

        new_g2 = (
            s2 * beta2
            - clipped_p2
        )

        rpc_signal = (
            (
                new_g1
                * logits1
            ).sum()
            + (
                new_g2
                * logits2
            ).sum()
        )

        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()

        pseudo_loss = rpc_signal

        return pseudo_loss

    if mode == "global":
        original_norm_sq = torch.tensor(
            0.0,
            device=x.device
        )

        for gradient in gradients_adapter:
            original_norm_sq += gradient.pow(2).sum()

        original_norm = torch.sqrt(
            original_norm_sq
            + RPC_EPS
        )

        grad_logits1, grad_logits2 = (
            torch.autograd.grad(
                loss,
                (
                    logits1,
                    logits2
                ),
                retain_graph=True,
                create_graph=False,
                allow_unused=False
            )
        )

        p1_pos = F.relu(
            -grad_logits1
        )

        p2_pos = F.relu(
            -grad_logits2
        )

        s1 = F.relu(
            grad_logits1
        )

        s2 = F.relu(
            grad_logits2
        )

        alpha_device = alpha.to(
            x.device
        )

        clipped_p1 = (
            p1_pos
            * alpha_device.view(
                1,
                -1
            )
        )

        clipped_p2 = (
            p2_pos
            * alpha_device.view(
                1,
                -1
            )
        )

        beta1 = (
            clipped_p1.sum(
                dim=1,
                keepdim=True
            )
            / s1.sum(
                dim=1,
                keepdim=True
            ).clamp_min(
                RPC_EPS
            )
        )

        beta2 = (
            clipped_p2.sum(
                dim=1,
                keepdim=True
            )
            / s2.sum(
                dim=1,
                keepdim=True
            ).clamp_min(
                RPC_EPS
            )
        )

        beta1 = beta1.clamp(
            max=1.0
        )

        beta2 = beta2.clamp(
            max=1.0
        )

        new_g1 = (
            s1 * beta1
            - clipped_p1
        )

        new_g2 = (
            s2 * beta2
            - clipped_p2
        )

        signal = (
            (
                new_g1
                * logits1
            ).sum()
            + (
                new_g2
                * logits2
            ).sum()
        )

        modified_gradients = torch.autograd.grad(
            signal,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )

        modified_norm_sq = torch.tensor(
            0.0,
            device=x.device
        )

        for gradient in modified_gradients:
            modified_norm_sq += gradient.pow(2).sum()

        modified_norm = torch.sqrt(
            modified_norm_sq
            + RPC_EPS
        )

        scale = (
            original_norm
            / modified_norm.clamp_min(
                RPC_EPS
            )
        )

        scaled_signal = (
            signal
            * scale.detach()
        )

        return scaled_signal

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def train_source_step(
    student,
    source_x,
    source_y,
    optimizer_adapter,
    optimizer_classifier,
    device
):
    source_x = source_x.to(
        device
    )

    source_y = source_y.to(
        device
    )

    if source_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Source batch is not 2048-D"
        )

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    logits1, logits2 = student(
        source_x
    )

    loss = (
        F.cross_entropy(
            logits1,
            source_y
        )
        + F.cross_entropy(
            logits2,
            source_y
        )
    )

    loss.backward()

    optimizer_adapter.step()
    optimizer_classifier.step()

    predictions = (
        (
            logits1
            + logits2
        )
        / 2.0
    ).argmax(
        dim=1
    )

    correct = int(
        (
            predictions
            == source_y
        ).sum().item()
    )

    return {
        "loss":
            float(
                loss.item()
            ),
        "correct":
            correct,
        "total":
            len(source_y)
    }


def train_target_step(
    student,
    target_x,
    optimizer_adapter,
    mode,
    alpha
):
    target_x = target_x.to(
        next(
            student.parameters()
        ).device
    )

    if target_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Target batch is not 2048-D"
        )

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    loss = apply_frozen_rpc_gradient(
        student,
        target_x,
        alpha,
        mode
    )

    loss.backward(
        retain_graph=False
    )

    optimizer_adapter.step()

    return float(
        loss.item()
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

        p = (
            p1 + p2
        ) / 2.0

        confidence, predictions = (
            p.max(
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

            if mask.any():
                class_total[
                    class_id
                ] += int(
                    mask.sum().item()
                )

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
        "overall":
            100.0
            * total_correct
            / max(
                total_count,
                1
            ),
        "mean_class":
            float(
                per_class.mean().item()
            ),
        "per_class":
            per_class.tolist(),
        "mean_confidence":
            confidence_sum
            / max(
                total_count,
                1
            )
    }


def run_mode(
    mode,
    base_student,
    base_teacher,
    source_features,
    source_labels,
    target_features,
    target_labels,
    alpha,
    device
):
    student = MCDModel().to(
        device
    )

    teacher = MCDModel().to(
        device
    )

    student.load_state_dict(
        {
            key:
                value.detach().clone()
            for key, value in base_student.state_dict().items()
        }
    )

    teacher.load_state_dict(
        {
            key:
                value.detach().clone()
            for key, value in base_teacher.state_dict().items()
        }
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
        target_features,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    source_iter = iter(
        source_loader
    )

    target_iter = iter(
        target_loader
    )

    history = []

    start = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        student.train()

        source_correct = 0
        source_total = 0
        source_loss_sum = 0.0
        target_loss_sum = 0.0

        steps = min(
            len(source_loader),
            len(target_loader)
        )

        for _ in range(
            steps
        ):
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
                )
            except StopIteration:
                target_iter = iter(
                    target_loader
                )
                target_x = next(
                    target_iter
                )

            source_result = train_source_step(
                student,
                source_x,
                source_y,
                optimizer_adapter,
                optimizer_classifier,
                device
            )

            source_correct += (
                source_result["correct"]
            )

            source_total += (
                source_result["total"]
            )

            source_loss_sum += (
                source_result["loss"]
            )

            if mode == "baseline":
                target_loss = train_target_step(
                    student,
                    target_x,
                    optimizer_adapter,
                    "baseline",
                    alpha
                )
            elif mode == "rpc":
                target_loss = train_target_step(
                    student,
                    target_x,
                    optimizer_adapter,
                    "rpc",
                    alpha
                )
            elif mode == "global":
                target_loss = train_target_step(
                    student,
                    target_x,
                    optimizer_adapter,
                    "global",
                    alpha
                )
            else:
                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            target_loss_sum += (
                target_loss
            )

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

        epoch_record = {
            "epoch":
                epoch,
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
            "target_loss":
                target_loss_sum
                / max(
                    steps,
                    1
                ),
            "student_overall":
                student_metrics[
                    "overall"
                ],
            "student_mean_class":
                student_metrics[
                    "mean_class"
                ],
            "student_per_class":
                student_metrics[
                    "per_class"
                ],
            "teacher_overall":
                teacher_metrics[
                    "overall"
                ],
            "teacher_mean_class":
                teacher_metrics[
                    "mean_class"
                ],
            "teacher_per_class":
                teacher_metrics[
                    "per_class"
                ]
        }

        history.append(
            epoch_record
        )

        print(
            f"{mode.upper():8s} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Source {epoch_record['source_accuracy']:.2f}% | "
            f"Student {epoch_record['student_mean_class']:.2f}% | "
            f"Teacher {epoch_record['teacher_mean_class']:.2f}%"
        )

    elapsed = (
        time.perf_counter()
        - start
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

    return {
        "student":
            student,
        "teacher":
            teacher,
        "history":
            history,
        "final_student":
            final_student,
        "final_teacher":
            final_teacher,
        "seconds":
            elapsed
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
        "VISDA-2017 RECIPIENT PROMOTION CAPPING"
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
        f"RPC kappa={RPC_KAPPA}"
    )

    print(
        f"global_pl_fraction={GLOBAL_PL_FRACTION}"
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

    base_student = MCDModel().to(
        device
    )

    base_teacher = MCDModel().to(
        device
    )

    source_prototypes = load_checkpoint(
        base_student,
        base_teacher
    )

    print(
        "Base checkpoint loaded successfully."
    )

    base_metrics = evaluate(
        base_student,
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
        f"{base_metrics['overall']:.2f}%"
    )

    print(
        f"Mean-class: "
        f"{base_metrics['mean_class']:.2f}%"
    )

    print()

    print(
        "Auditing frozen target promotion mass..."
    )

    target_predictions = []

    base_teacher.eval()

    with torch.no_grad():
        loader = DataLoader(
            target_features,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        for x in loader:
            x = x.to(
                device
            )

            logits1, logits2 = (
                base_teacher(x)
            )

            p = (
                F.softmax(
                    logits1,
                    dim=1
                )
                + F.softmax(
                    logits2,
                    dim=1
                )
            ) / 2.0

            target_predictions.append(
                p.argmax(
                    dim=1
                ).cpu()
            )

    target_predictions = torch.cat(
        target_predictions,
        dim=0
    )

    promotion_mass, suppression_mass, audited_samples = (
        audit_target_promotion_mass(
            base_student,
            target_features,
            target_predictions,
            device
        )
    )

    alpha, median_mass, threshold = (
        build_rpc_alpha(
            promotion_mass
        )
    )

    print()
    print(
        f"Audited samples: "
        f"{audited_samples}"
    )

    print(
        f"Median promotion mass: "
        f"{median_mass.item():.10f}"
    )

    print(
        f"RPC threshold: "
        f"{threshold.item():.10f}"
    )

    print_rpc_coefficients(
        promotion_mass,
        alpha
    )

    print()
    print(
        "BASE PROMOTION / SUPPRESSION"
    )

    promotion_share = (
        promotion_mass
        / promotion_mass.sum().clamp_min(
            RPC_EPS
        )
    )

    suppression_share = (
        suppression_mass
        / suppression_mass.sum().clamp_min(
            RPC_EPS
        )
    )

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"promotion_share="
            f"{100.0 * promotion_share[class_id].item():7.3f}% | "
            f"suppression_share="
            f"{100.0 * suppression_share[class_id].item():7.3f}%"
        )

    print()
    print(
        "RUNNING BASELINE"
    )

    baseline_result = run_mode(
        "baseline",
        base_student,
        base_teacher,
        source_features,
        source_labels,
        target_features,
        target_labels,
        alpha,
        device
    )

    print()
    print(
        "RUNNING NORM-MATCHED GLOBAL ATTENUATION"
    )

    global_result = run_mode(
        "global",
        base_student,
        base_teacher,
        source_features,
        source_labels,
        target_features,
        target_labels,
        alpha,
        device
    )

    print()
    print(
        "RUNNING RECIPIENT PROMOTION CAPPING"
    )

    rpc_result = run_mode(
        "rpc",
        base_student,
        base_teacher,
        source_features,
        source_labels,
        target_features,
        target_labels,
        alpha,
        device
    )

    print()
    print("=" * 90)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 90)

    comparisons = {
        "baseline":
            baseline_result["final_student"],
        "global_attenuation":
            global_result["final_student"],
        "rpc":
            rpc_result["final_student"]
    }

    for name, metrics in comparisons.items():
        print(
            f"{name:24s} | "
            f"Overall={metrics['overall']:.2f}% | "
            f"Mean-class={metrics['mean_class']:.2f}% | "
            f"Truck={metrics['per_class'][TRUCK_ID]:.2f}% | "
            f"Car={metrics['per_class'][CAR_ID]:.2f}% | "
            f"Bus={metrics['per_class'][BUS_ID]:.2f}% | "
            f"Train={metrics['per_class'][TRAIN_ID]:.2f}%"
        )

    print()
    print(
        "RPC ALPHAS"
    )

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASSES[class_id]:12s}: "
            f"{alpha[class_id].item():.6f}"
        )

    total_seconds = (
        baseline_result["seconds"]
        + global_result["seconds"]
        + rpc_result["seconds"]
    )

    report = {
        "experiment":
            "visda_recipient_promotion_capping",
        "seed":
            SEED,
        "base_checkpoint":
            str(BASE_CHECKPOINT),
        "epochs":
            EPOCHS,
        "global_pl_fraction":
            GLOBAL_PL_FRACTION,
        "rpc_kappa":
            RPC_KAPPA,
        "promotion_mass":
            promotion_mass.tolist(),
        "suppression_mass":
            suppression_mass.tolist(),
        "promotion_share":
            promotion_share.tolist(),
        "suppression_share":
            suppression_share.tolist(),
        "rpc_alpha":
            alpha.tolist(),
        "median_promotion_mass":
            float(
                median_mass.item()
            ),
        "rpc_threshold":
            float(
                threshold.item()
            ),
        "base_metrics":
            base_metrics,
        "baseline":
            {
                "final_student":
                    baseline_result[
                        "final_student"
                    ],
                "final_teacher":
                    baseline_result[
                        "final_teacher"
                    ],
                "history":
                    baseline_result[
                        "history"
                    ],
                "seconds":
                    baseline_result[
                        "seconds"
                    ]
            },
        "global_attenuation":
            {
                "final_student":
                    global_result[
                        "final_student"
                    ],
                "final_teacher":
                    global_result[
                        "final_teacher"
                    ],
                "history":
                    global_result[
                        "history"
                    ],
                "seconds":
                    global_result[
                        "seconds"
                    ]
            },
        "rpc":
            {
                "final_student":
                    rpc_result[
                        "final_student"
                    ],
                "final_teacher":
                    rpc_result[
                        "final_teacher"
                    ],
                "history":
                    rpc_result[
                        "history"
                    ],
                "seconds":
                    rpc_result[
                        "seconds"
                    ]
            },
        "total_seconds":
            total_seconds
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

    torch.save(
        {
            "rpc_alpha":
                alpha,
            "promotion_mass":
                promotion_mass,
            "suppression_mass":
                suppression_mass,
            "base_checkpoint":
                str(BASE_CHECKPOINT),
            "rpc_kappa":
                RPC_KAPPA,
            "baseline_student_state_dict":
                baseline_result[
                    "student"
                ].state_dict(),
            "rpc_student_state_dict":
                rpc_result[
                    "student"
                ].state_dict(),
            "baseline_teacher_state_dict":
                baseline_result[
                    "teacher"
                ].state_dict(),
            "rpc_teacher_state_dict":
                rpc_result[
                    "teacher"
                ].state_dict()
        },
        OUTPUT_CHECKPOINT
    )

    print()
    print(
        "=" * 90
    )
    print(
        "EXPERIMENT COMPLETE"
    )
    print(
        "=" * 90
    )

    print(
        f"checkpoint={OUTPUT_CHECKPOINT}"
    )

    print(
        f"history={OUTPUT_HISTORY}"
    )


if __name__ == "__main__":
    main()