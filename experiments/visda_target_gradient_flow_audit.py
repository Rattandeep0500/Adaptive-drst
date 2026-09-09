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

TARGET_BATCHES = 120
FUNCTIONAL_BATCHES = 20

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
    "checkpoints/visda_target_gradient_flow_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "target_gradient_flow_audit_seed42.json"
)

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
            f"No cache files found in {cache_dir}"
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
                f"Invalid feature tensor in {path}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
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
                f"Expected 2-D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        return self.net(x)


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
                f"Expected 2-D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D input, "
                f"got {x.shape[1]}"
            )

        z = self.adapter(x)

        logits1 = self.classifier1(z)
        logits2 = self.classifier2(z)

        return logits1, logits2


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
    model
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

    model.load_state_dict(
        normalize_state_dict(
            payload[
                "student_state_dict"
            ]
        ),
        strict=True
    )


def discrepancy_loss_from_logits(
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
def collect_predictions(
    teacher,
    target_features,
    device
):
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

    prediction_parts = []
    confidence_parts = []

    for (x,) in loader:
        x = x.to(device)

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

        prediction_parts.append(
            predictions.cpu()
        )

        confidence_parts.append(
            confidence.cpu()
        )

    return (
        torch.cat(
            prediction_parts,
            dim=0
        ),
        torch.cat(
            confidence_parts,
            dim=0
        )
    )


def direct_logit_gradients(
    model,
    x
):
    model.zero_grad(
        set_to_none=True
    )

    logits1, logits2 = model(
        x
    )

    loss = discrepancy_loss_from_logits(
        logits1,
        logits2
    )

    grad1, grad2 = torch.autograd.grad(
        outputs=loss,
        inputs=(
            logits1,
            logits2
        ),
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )

    gradients = (
        grad1
        + grad2
    ) / 2.0

    return (
        gradients.detach(),
        loss.detach()
    )


def promotion_suppression_mass(
    gradients
):
    promotion = F.relu(
        -gradients
    )

    suppression = F.relu(
        gradients
    )

    promotion_mass = promotion.sum(
        dim=0
    )

    suppression_mass = suppression.sum(
        dim=0
    )

    return (
        promotion_mass,
        suppression_mass
    )


def class_conditioned_flow(
    gradients,
    predictions
):
    flow = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES
    )

    for source_class in range(
        NUM_CLASSES
    ):
        mask = (
            predictions
            == source_class
        )

        if not mask.any():
            continue

        class_grad = gradients[
            mask
        ]

        source_grad = class_grad[
            :,
            source_class
        ]

        for target_class in range(
            NUM_CLASSES
        ):
            if target_class == source_class:
                continue

            target_grad = class_grad[
                :,
                target_class
            ]

            target_vs_source = (
                target_grad
                - source_grad
            )

            flow_value = F.relu(
                -target_vs_source
            ).sum()

            flow[
                source_class,
                target_class
            ] += float(
                flow_value.item()
            )

    return flow


def safe_ratio(
    numerator,
    denominator
):
    return (
        numerator
        / denominator.clamp_min(1e-12)
    )


def print_class_mass(
    promotion,
    suppression
):
    promotion_fraction = safe_ratio(
        promotion,
        promotion.sum()
    )

    suppression_fraction = safe_ratio(
        suppression,
        suppression.sum()
    )

    print()
    print(
        "CLASS PROMOTION / SUPPRESSION"
    )

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"promo={promotion[class_id].item():.8f} "
            f"({100.0 * promotion_fraction[class_id].item():7.3f}%) | "
            f"suppress={suppression[class_id].item():.8f} "
            f"({100.0 * suppression_fraction[class_id].item():7.3f}%)"
        )


def print_flow_matrix(
    matrix,
    title
):
    print()
    print(title)

    header = (
        "source\\target".ljust(16)
        + " ".join(
            [
                name[:5].rjust(8)
                for name in CLASSES
            ]
        )
    )

    print(
        header
    )

    for source_id in range(
        NUM_CLASSES
    ):
        row = (
            CLASSES[
                source_id
            ][:14].ljust(16)
        )

        values = []

        for target_id in range(
            NUM_CLASSES
        ):
            if source_id == target_id:
                values.append(
                    "    --  "
                )
            else:
                values.append(
                    f"{matrix[source_id, target_id].item():8.3f}"
                )

        print(
            row
            + " ".join(values)
        )


def top_flows(
    matrix,
    n=20
):
    values = []

    for source_id in range(
        NUM_CLASSES
    ):
        for target_id in range(
            NUM_CLASSES
        ):
            if source_id == target_id:
                continue

            values.append(
                (
                    float(
                        matrix[
                            source_id,
                            target_id
                        ].item()
                    ),
                    source_id,
                    target_id
                )
            )

    values.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return values[
        :n
    ]


def functional_margin_flow(
    model,
    x,
    predictions,
    step_size=1e-4
):
    model.eval()

    base_logits1, base_logits2 = model(
        x
    )

    base_logits = (
        base_logits1
        + base_logits2
    ) / 2.0

    model.zero_grad(
        set_to_none=True
    )

    logits1, logits2 = model(
        x
    )

    loss = discrepancy_loss_from_logits(
        logits1,
        logits2
    )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False
    )

    saved = [
        parameter.detach().clone()
        for parameter in parameters
    ]

    with torch.no_grad():
        for parameter, gradient in zip(
            parameters,
            gradients
        ):
            parameter.add_(
                gradient,
                alpha=-step_size
            )

        updated_logits1, updated_logits2 = model(
            x
        )

        updated_logits = (
            updated_logits1
            + updated_logits2
        ) / 2.0

        for parameter, saved_value in zip(
            parameters,
            saved
        ):
            parameter.copy_(
                saved_value
            )

    delta = (
        updated_logits
        - base_logits
    ) / step_size

    flow = class_conditioned_flow(
        delta,
        predictions
    )

    return (
        flow,
        delta.detach()
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
        "VISDA-2017 TARGET GRADIENT-FLOW AUDIT"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"target_batches={TARGET_BATCHES}"
    )

    print(
        f"functional_batches={FUNCTIONAL_BATCHES}"
    )

    print(
        f"checkpoint={BASE_CHECKPOINT}"
    )

    print()

    print(
        "Loading source cache..."
    )

    source_features, source_labels = load_cache(
        SOURCE_CACHE
    )

    print(
        f"Source samples: "
        f"{len(source_features)}"
    )

    print()

    print(
        "Loading target cache..."
    )

    target_features, target_labels = load_cache(
        TARGET_CACHE
    )

    print(
        f"Target samples: "
        f"{len(target_features)}"
    )

    student = MCDModel().to(
        device
    )

    teacher = MCDModel().to(
        device
    )

    load_checkpoint(
        student
    )

    payload = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
    )

    teacher.load_state_dict(
        normalize_state_dict(
            payload[
                "teacher_state_dict"
            ]
        ),
        strict=True
    )

    student.eval()
    teacher.eval()

    print()
    print(
        "Checkpoint loaded successfully."
    )

    print()
    print(
        "Collecting target predictions..."
    )

    target_predictions, target_confidence = (
        collect_predictions(
            teacher,
            target_features,
            device
        )
    )

    target_accuracy = (
        (
            target_predictions
            == target_labels
        )
        .float()
        .mean()
        .item()
    )

    print(
        f"Teacher target accuracy: "
        f"{100.0 * target_accuracy:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{target_confidence.mean().item():.6f}"
    )

    target_loader = DataLoader(
        TensorDataset(
            target_features,
            target_predictions
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    total_promotion = torch.zeros(
        NUM_CLASSES
    )

    total_suppression = torch.zeros(
        NUM_CLASSES
    )

    total_flow = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES
    )

    audited_samples = 0
    audited_batches = 0
    target_loss_sum = 0.0

    start = time.perf_counter()

    print()
    print(
        "Computing direct logit gradients..."
    )

    for raw_x, prediction_batch in target_loader:
        if audited_batches >= TARGET_BATCHES:
            break

        raw_x = raw_x.to(
            device
        )

        prediction_batch = prediction_batch.to(
            device
        )

        if raw_x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Target input is not 2048-D"
            )

        gradients, loss = (
            direct_logit_gradients(
                student,
                raw_x
            )
        )

        promotion_mass, suppression_mass = (
            promotion_suppression_mass(
                gradients
            )
        )

        flow = class_conditioned_flow(
            gradients.cpu(),
            prediction_batch.cpu()
        )

        total_promotion += (
            promotion_mass.cpu()
        )

        total_suppression += (
            suppression_mass.cpu()
        )

        total_flow += flow

        audited_samples += (
            len(raw_x)
        )

        audited_batches += 1

        target_loss_sum += (
            loss.item()
        )

        if (
            audited_batches == 1
            or audited_batches % 20 == 0
        ):
            print(
                f"  batch "
                f"{audited_batches}/"
                f"{TARGET_BATCHES}"
            )

    gradient_seconds = (
        time.perf_counter()
        - start
    )

    mean_target_loss = (
        target_loss_sum
        / max(
            audited_batches,
            1
        )
    )

    print()
    print(
        f"Audited samples: "
        f"{audited_samples}"
    )

    print(
        f"Mean discrepancy loss: "
        f"{mean_target_loss:.8f}"
    )

    print(
        f"Audit time: "
        f"{gradient_seconds:.2f}s"
    )

    print_class_mass(
        total_promotion,
        total_suppression
    )

    print()
    print(
        "SINK CLASS GRADIENT SUMMARY"
    )

    promotion_fraction = safe_ratio(
        total_promotion,
        total_promotion.sum()
    )

    suppression_fraction = safe_ratio(
        total_suppression,
        total_suppression.sum()
    )

    for class_id in (
        TRUCK_ID,
        CAR_ID,
        BUS_ID,
        TRAIN_ID
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"promotion={total_promotion[class_id].item():.8f} | "
            f"promotion_share={100.0 * promotion_fraction[class_id].item():.4f}% | "
            f"suppression={total_suppression[class_id].item():.8f} | "
            f"suppression_share={100.0 * suppression_fraction[class_id].item():.4f}%"
        )

    print_flow_matrix(
        total_flow,
        "TARGET CLASS-CONDITIONED GRADIENT FLOW"
    )

    print()
    print(
        "TOP TARGET GRADIENT FLOWS"
    )

    for value, source_id, target_id in top_flows(
        total_flow,
        n=24
    ):
        print(
            f"{CLASSES[source_id]:12s} -> "
            f"{CLASSES[target_id]:12s} | "
            f"{value:.6f}"
        )

    print()
    print(
        "TRUCK TARGET GRADIENT FLOWS"
    )

    truck_flow = {}

    for target_id in (
        CAR_ID,
        BUS_ID,
        TRAIN_ID
    ):
        forward = total_flow[
            TRUCK_ID,
            target_id
        ].item()

        reverse = total_flow[
            target_id,
            TRUCK_ID
        ].item()

        net = (
            forward
            - reverse
        )

        truck_flow[
            CLASSES[target_id]
        ] = {
            "truck_to_target":
                float(forward),
            "target_to_truck":
                float(reverse),
            "net":
                float(net)
        }

        print(
            f"truck -> "
            f"{CLASSES[target_id]:10s}: "
            f"{forward:.6f} | "
            f"{CLASSES[target_id]} -> truck: "
            f"{reverse:.6f} | "
            f"net={net:.6f}"
        )

    print()
    print(
        "FUNCTIONAL MARGIN-FLOW AUDIT"
    )

    functional_loader = DataLoader(
        TensorDataset(
            target_features,
            target_predictions
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    functional_total_flow = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES
    )

    functional_count = 0

    functional_start = time.perf_counter()

    for raw_x, prediction_batch in functional_loader:
        if functional_count >= FUNCTIONAL_BATCHES:
            break

        raw_x = raw_x.to(
            device
        )

        prediction_batch = prediction_batch.to(
            device
        )

        flow, _ = functional_margin_flow(
            student,
            raw_x,
            prediction_batch
        )

        functional_total_flow += flow

        functional_count += 1

        if (
            functional_count == 1
            or functional_count % 5 == 0
        ):
            print(
                f"  functional batch "
                f"{functional_count}/"
                f"{FUNCTIONAL_BATCHES}"
            )

    functional_seconds = (
        time.perf_counter()
        - functional_start
    )

    print()
    print(
        f"Functional batches: "
        f"{functional_count}"
    )

    print(
        f"Functional audit time: "
        f"{functional_seconds:.2f}s"
    )

    print_flow_matrix(
        functional_total_flow,
        "FUNCTIONAL TARGET MARGIN FLOW"
    )

    print()
    print(
        "TRUCK FUNCTIONAL FLOWS"
    )

    functional_truck_flow = {}

    for target_id in (
        CAR_ID,
        BUS_ID,
        TRAIN_ID
    ):
        forward = functional_total_flow[
            TRUCK_ID,
            target_id
        ].item()

        reverse = functional_total_flow[
            target_id,
            TRUCK_ID
        ].item()

        net = (
            forward
            - reverse
        )

        functional_truck_flow[
            CLASSES[target_id]
        ] = {
            "truck_to_target":
                float(forward),
            "target_to_truck":
                float(reverse),
            "net":
                float(net)
        }

        print(
            f"truck -> "
            f"{CLASSES[target_id]:10s}: "
            f"{forward:.6f} | "
            f"{CLASSES[target_id]} -> truck: "
            f"{reverse:.6f} | "
            f"net={net:.6f}"
        )

    result = {
        "experiment":
            "visda_target_gradient_flow_audit",
        "seed":
            SEED,
        "checkpoint":
            str(BASE_CHECKPOINT),
        "target_samples":
            len(target_features),
        "target_accuracy_from_teacher":
            100.0 * target_accuracy,
        "mean_confidence":
            float(
                target_confidence.mean().item()
            ),
        "audited_batches":
            audited_batches,
        "audited_samples":
            audited_samples,
        "mean_target_discrepancy":
            mean_target_loss,
        "gradient_audit_seconds":
            gradient_seconds,
        "promotion_mass":
            total_promotion.tolist(),
        "suppression_mass":
            total_suppression.tolist(),
        "promotion_suppression_summary":
            {
                CLASSES[class_id]: {
                    "promotion":
                        float(
                            total_promotion[
                                class_id
                            ].item()
                        ),
                    "suppression":
                        float(
                            total_suppression[
                                class_id
                            ].item()
                        ),
                    "promotion_share":
                        float(
                            promotion_fraction[
                                class_id
                            ].item()
                        ),
                    "suppression_share":
                        float(
                            suppression_fraction[
                                class_id
                            ].item()
                        )
                }
                for class_id in range(
                    NUM_CLASSES
                )
            },
        "gradient_flow_matrix":
            total_flow.tolist(),
        "truck_gradient_flow":
            truck_flow,
        "functional_batches":
            functional_count,
        "functional_audit_seconds":
            functional_seconds,
        "functional_flow_matrix":
            functional_total_flow.tolist(),
        "truck_functional_flow":
            functional_truck_flow
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2
        )

    print()
    print("=" * 90)
    print(
        "AUDIT COMPLETE"
    )
    print("=" * 90)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()