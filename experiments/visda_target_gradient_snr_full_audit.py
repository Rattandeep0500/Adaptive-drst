import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SEED = 42

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/"
    "geometry_gated_mcd_seed42.pt"
)

CACHE_ROOT = Path(
    "checkpoints/visda_feature_cache"
)

TARGET_CACHE = CACHE_ROOT / "target"

OUTPUT_DIR = Path(
    "checkpoints/visda_target_gradient_snr_full"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "target_gradient_snr_full_seed42.json"
)

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10

FOCUS_IDS = [
    TRUCK_ID,
    CAR_ID,
    BUS_ID,
    TRAIN_ID,
]

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
                f"Invalid feature tensor in {path}: "
                f"{tuple(x.shape)}"
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

    if features.ndim != 2:
        raise RuntimeError(
            "Final feature tensor must be 2-D"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Final feature dimension must be "
            f"{INPUT_DIM}, got {features.shape[1]}"
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
                f"Adapter expects 2-D input, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expects {INPUT_DIM}-D input, "
                f"got {x.shape[1]}"
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
                f"Model expects 2-D input, "
                f"got {tuple(x.shape)}"
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


def load_models():
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{BASE_CHECKPOINT}"
        )

    payload = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
    )

    required = [
        "student_state_dict",
        "teacher_state_dict",
    ]

    for key in required:
        if key not in payload:
            raise RuntimeError(
                f"Checkpoint missing {key}"
            )

    student = MCDModel()
    teacher = MCDModel()

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

    student.eval()
    teacher.eval()

    return student, teacher


@torch.no_grad()
def collect_teacher_predictions(
    teacher,
    target_features,
    device
):
    teacher.eval()

    loader = DataLoader(
        target_features,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    prediction_parts = []
    confidence_parts = []

    for x in loader:
        x = x.to(device)

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Teacher received invalid input dimension"
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

        confidence, prediction = (
            probabilities.max(
                dim=1
            )
        )

        prediction_parts.append(
            prediction.cpu()
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

    logits1, logits2 = model(x)

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
        grad1 + grad2
    ) / 2.0

    return (
        gradients.detach(),
        loss.detach()
    )


def compute_batch_class_statistics(
    gradients,
    predictions
):
    results = {}

    for class_id in range(
        NUM_CLASSES
    ):
        name = CLASSES[
            class_id
        ]

        mask = (
            predictions
            == class_id
        )

        if not mask.any():
            results[name] = {
                "count": 0,
                "mean_norm": None,
                "mean_vector_norm": None,
                "rms_norm": None,
                "mean_cosine": None,
                "cosine_std": None,
                "snr": None,
                "mean_abs_gradient": None,
            }

            continue

        class_gradients = gradients[
            mask
        ]

        norms = torch.linalg.vector_norm(
            class_gradients,
            dim=1
        )

        mean_vector = (
            class_gradients.mean(
                dim=0
            )
        )

        mean_vector_norm = (
            torch.linalg.vector_norm(
                mean_vector
            )
        )

        if mean_vector_norm.item() <= 1e-12:
            cosines = torch.zeros(
                len(class_gradients)
            )
        else:
            cosines = F.cosine_similarity(
                class_gradients,
                mean_vector.unsqueeze(0),
                dim=1,
                eps=1e-8
            )

        rms_norm = torch.sqrt(
            torch.mean(
                norms.pow(2)
            )
            + 1e-12
        )

        snr = (
            mean_vector_norm
            / rms_norm.clamp_min(
                1e-12
            )
        )

        results[name] = {
            "count":
                int(
                    len(class_gradients)
                ),
            "mean_norm":
                float(
                    norms.mean().item()
                ),
            "mean_vector_norm":
                float(
                    mean_vector_norm.item()
                ),
            "rms_norm":
                float(
                    rms_norm.item()
                ),
            "mean_cosine":
                float(
                    cosines.mean().item()
                ),
            "cosine_std":
                float(
                    cosines.std(
                        unbiased=False
                    ).item()
                ),
            "snr":
                float(
                    snr.item()
                ),
            "mean_abs_gradient":
                float(
                    class_gradients.abs().mean().item()
                ),
        }

    return results


def aggregate_class_statistics(
    batch_records
):
    summaries = {}

    for class_name in CLASSES:
        snr_values = []
        norm_values = []
        cosine_values = []
        count_values = []

        for record in batch_records:
            row = record[
                class_name
            ]

            if row["count"] <= 0:
                continue

            snr_values.append(
                row["snr"]
            )

            norm_values.append(
                row["mean_norm"]
            )

            cosine_values.append(
                row["mean_cosine"]
            )

            count_values.append(
                row["count"]
            )

        if not snr_values:
            summaries[class_name] = {
                "batches":
                    0,
                "mean_snr":
                    None,
                "std_snr":
                    None,
                "median_snr":
                    None,
                "q25_snr":
                    None,
                "q75_snr":
                    None,
                "mean_norm":
                    None,
                "std_norm":
                    None,
                "median_norm":
                    None,
                "mean_cosine":
                    None,
                "std_cosine":
                    None,
                "median_cosine":
                    None,
                "mean_count":
                    None,
                "total_count":
                    0,
            }

            continue

        snr_array = np.asarray(
            snr_values,
            dtype=np.float64
        )

        norm_array = np.asarray(
            norm_values,
            dtype=np.float64
        )

        cosine_array = np.asarray(
            cosine_values,
            dtype=np.float64
        )

        count_array = np.asarray(
            count_values,
            dtype=np.float64
        )

        summaries[class_name] = {
            "batches":
                int(
                    len(snr_array)
                ),
            "mean_snr":
                float(
                    snr_array.mean()
                ),
            "std_snr":
                float(
                    snr_array.std()
                ),
            "median_snr":
                float(
                    np.median(
                        snr_array
                    )
                ),
            "q25_snr":
                float(
                    np.quantile(
                        snr_array,
                        0.25
                    )
                ),
            "q75_snr":
                float(
                    np.quantile(
                        snr_array,
                        0.75
                    )
                ),
            "mean_norm":
                float(
                    norm_array.mean()
                ),
            "std_norm":
                float(
                    norm_array.std()
                ),
            "median_norm":
                float(
                    np.median(
                        norm_array
                    )
                ),
            "mean_cosine":
                float(
                    cosine_array.mean()
                ),
            "std_cosine":
                float(
                    cosine_array.std()
                ),
            "median_cosine":
                float(
                    np.median(
                        cosine_array
                    )
                ),
            "mean_count":
                float(
                    count_array.mean()
                ),
            "total_count":
                int(
                    count_array.sum()
                ),
        }

    return summaries


def classwise_head_gradient_vectors(
    model,
    x,
    predictions
):
    model.zero_grad(
        set_to_none=True
    )

    logits1, logits2 = model(x)

    p1 = F.softmax(
        logits1,
        dim=1
    )

    p2 = F.softmax(
        logits2,
        dim=1
    )

    per_sample_loss = (
        p1 - p2
    ).abs().mean(
        dim=1
    )

    parameters = [
        model.classifier1.weight,
        model.classifier1.bias,
        model.classifier2.weight,
        model.classifier2.bias,
    ]

    vectors = {}

    for class_id in range(
        NUM_CLASSES
    ):
        name = CLASSES[
            class_id
        ]

        mask = (
            predictions
            == class_id
        )

        if not mask.any():
            vectors[name] = None
            continue

        selected_loss = (
            per_sample_loss[
                mask
            ].mean()
        )

        grads = torch.autograd.grad(
            selected_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False
        )

        vectors[name] = torch.cat(
            [
                grad.detach().reshape(-1)
                for grad in grads
            ]
        )

    return vectors


def cosine_similarity_value(
    a,
    b
):
    if a is None or b is None:
        return None

    return float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=1,
            eps=1e-8
        )[0].item()
    )


def build_alignment_table(
    vectors
):
    matrix = torch.full(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        float("nan")
    )

    for i in range(
        NUM_CLASSES
    ):
        if vectors[
            CLASSES[i]
        ] is None:
            continue

        for j in range(
            NUM_CLASSES
        ):
            if vectors[
                CLASSES[j]
            ] is None:
                continue

            value = cosine_similarity_value(
                vectors[
                    CLASSES[i]
                ],
                vectors[
                    CLASSES[j]
                ]
            )

            matrix[
                i,
                j
            ] = value

    return matrix


def print_snr_summary(
    summaries
):
    print()
    print(
        "FULL-TARGET PER-CLASS GRADIENT SNR"
    )

    for class_name in CLASSES:
        row = summaries[
            class_name
        ]

        if row["batches"] == 0:
            print(
                f"{class_name:12s} | "
                f"no observations"
            )
            continue

        print(
            f"{class_name:12s} | "
            f"batches={row['batches']:3d} | "
            f"mean_snr={row['mean_snr']:.6f} | "
            f"median_snr={row['median_snr']:.6f} | "
            f"q25={row['q25_snr']:.6f} | "
            f"q75={row['q75_snr']:.6f} | "
            f"mean_norm={row['mean_norm']:.8f} | "
            f"mean_cos={row['mean_cosine']:.6f}"
        )


def print_focus_summary(
    summaries
):
    print()
    print(
        "TRUCK VS SINK CLASSES"
    )

    for class_id in FOCUS_IDS:
        name = CLASSES[
            class_id
        ]

        row = summaries[
            name
        ]

        if row["batches"] == 0:
            print(
                f"{name:12s} | no observations"
            )
            continue

        print(
            f"{name:12s} | "
            f"mean_norm={row['mean_norm']:.8f} | "
            f"std_norm={row['std_norm']:.8f} | "
            f"mean_cos={row['mean_cosine']:.6f} | "
            f"std_cos={row['std_cosine']:.6f} | "
            f"SNR={row['mean_snr']:.6f} | "
            f"SNR_std={row['std_snr']:.6f}"
        )


def print_relative_snr(
    summaries
):
    truck = summaries[
        "truck"
    ]

    print()
    print(
        "TRUCK RELATIVE TO SINKS"
    )

    if truck["mean_snr"] is None:
        print(
            "Truck has no observations."
        )
        return

    for other_name in (
        "car",
        "bus",
        "train"
    ):
        other = summaries[
            other_name
        ]

        if (
            other["mean_snr"] is None
            or other["mean_norm"] is None
        ):
            print(
                f"truck/{other_name:5s} | unavailable"
            )
            continue

        snr_ratio = (
            truck["mean_snr"]
            / max(
                other["mean_snr"],
                1e-12
            )
        )

        norm_ratio = (
            truck["mean_norm"]
            / max(
                other["mean_norm"],
                1e-12
            )
        )

        print(
            f"truck/{other_name:5s} | "
            f"SNR ratio={snr_ratio:.6f} | "
            f"norm ratio={norm_ratio:.6f}"
        )


def print_alignment(
    matrix
):
    print()
    print(
        "CLASS HEAD-GRADIENT ALIGNMENT"
    )

    for other_id in (
        CAR_ID,
        BUS_ID,
        TRAIN_ID
    ):
        value = matrix[
            TRUCK_ID,
            other_id
        ].item()

        if np.isnan(value):
            text = "unavailable"
        else:
            text = f"{value:.6f}"

        print(
            f"truck vs "
            f"{CLASSES[other_id]:5s}: "
            f"{text}"
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
        "VISDA-2017 FULL TARGET GRADIENT SNR AUDIT"
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
        "audited_batches=ALL"
    )

    print(
        f"checkpoint={BASE_CHECKPOINT}"
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

    print()

    print(
        "Loading checkpoint..."
    )

    student, teacher = load_models()

    student = student.to(
        device
    )

    teacher = teacher.to(
        device
    )

    student.eval()
    teacher.eval()

    print(
        "Checkpoint loaded successfully."
    )

    print()

    print(
        "Collecting teacher predictions..."
    )

    target_predictions, target_confidence = (
        collect_teacher_predictions(
            teacher,
            target_features,
            device
        )
    )

    teacher_accuracy = (
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
        f"{100.0 * teacher_accuracy:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{target_confidence.mean().item():.6f}"
    )

    target_loader = DataLoader(
        target_features,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    prediction_loader = DataLoader(
        target_predictions,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    batch_records = []

    total_promotion = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    total_suppression = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    total_samples = 0
    total_batches = 0
    discrepancy_sum = 0.0

    print()
    print(
        "Computing full-target gradient statistics..."
    )

    start_time = time.perf_counter()

    for raw_x, prediction_batch in zip(
        target_loader,
        prediction_loader
    ):
        raw_x = raw_x.to(
            device
        )

        prediction_batch = prediction_batch.to(
            device
        )

        if raw_x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D raw input, "
                f"got {raw_x.shape[1]}"
            )

        gradients, loss = (
            direct_logit_gradients(
                student,
                raw_x
            )
        )

        batch_statistics = (
            compute_batch_class_statistics(
                gradients.cpu(),
                prediction_batch.cpu()
            )
        )

        batch_records.append(
            batch_statistics
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

        total_promotion += (
            promotion.detach()
            .cpu()
            .double()
        )

        total_suppression += (
            suppression.detach()
            .cpu()
            .double()
        )

        discrepancy_sum += (
            loss.item()
        )

        total_samples += len(
            raw_x
        )

        total_batches += 1

        if (
            total_batches == 1
            or total_batches % 20 == 0
        ):
            print(
                f"  batch "
                f"{total_batches}/"
                f"{len(target_loader)}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    summaries = aggregate_class_statistics(
        batch_records
    )

    print()
    print(
        f"Audited samples: "
        f"{total_samples}"
    )

    print(
        f"Audited batches: "
        f"{total_batches}"
    )

    print(
        f"Mean target discrepancy: "
        f"{discrepancy_sum / max(total_batches, 1):.8f}"
    )

    print(
        f"Audit time: "
        f"{elapsed:.2f}s"
    )

    print_snr_summary(
        summaries
    )

    print_focus_summary(
        summaries
    )

    print_relative_snr(
        summaries
    )

    promotion_share = (
        total_promotion
        / total_promotion.sum().clamp_min(
            1e-12
        )
    )

    suppression_share = (
        total_suppression
        / total_suppression.sum().clamp_min(
            1e-12
        )
    )

    print()
    print(
        "PROMOTION / SUPPRESSION SHARE"
    )

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"promotion="
            f"{100.0 * promotion_share[class_id].item():7.3f}% | "
            f"suppression="
            f"{100.0 * suppression_share[class_id].item():7.3f}%"
        )

    print()
    print(
        "TRUCK PROMOTION / SUPPRESSION"
    )

    print(
        f"truck promotion mass: "
        f"{total_promotion[TRUCK_ID].item():.10f}"
    )

    print(
        f"truck suppression mass: "
        f"{total_suppression[TRUCK_ID].item():.10f}"
    )

    print(
        f"truck promotion share: "
        f"{100.0 * promotion_share[TRUCK_ID].item():.6f}%"
    )

    print(
        f"truck suppression share: "
        f"{100.0 * suppression_share[TRUCK_ID].item():.6f}%"
    )

    print()
    print(
        "CLASS-WISE HEAD GRADIENT ALIGNMENT"
    )

    first_x = target_features[
        :BATCH_SIZE
    ].to(
        device
    )

    first_predictions = (
        target_predictions[
            :len(first_x)
        ].to(
            device
        )
    )

    vectors = classwise_head_gradient_vectors(
        student,
        first_x,
        first_predictions
    )

    alignment_matrix = build_alignment_table(
        vectors
    )

    print_alignment(
        alignment_matrix
    )

    print()
    print(
        "FULL ALIGNMENT MATRIX"
    )

    header = (
        "class".ljust(16)
        + " ".join(
            name[:5].rjust(8)
            for name in CLASSES
        )
    )

    print(
        header
    )

    for i in range(
        NUM_CLASSES
    ):
        row = (
            CLASSES[i][:14].ljust(16)
        )

        values = []

        for j in range(
            NUM_CLASSES
        ):
            value = alignment_matrix[
                i,
                j
            ].item()

            if np.isnan(value):
                values.append(
                    "     nan"
                )
            else:
                values.append(
                    f"{value:8.3f}"
                )

        print(
            row
            + " ".join(values)
        )

    truck_alignment = {}

    for other_id in (
        CAR_ID,
        BUS_ID,
        TRAIN_ID
    ):
        value = alignment_matrix[
            TRUCK_ID,
            other_id
        ].item()

        if np.isnan(value):
            truck_alignment[
                CLASSES[other_id]
            ] = None
        else:
            truck_alignment[
                CLASSES[other_id]
            ] = float(value)

    result = {
        "experiment":
            "visda_target_gradient_snr_full_audit",
        "seed":
            SEED,
        "checkpoint":
            str(BASE_CHECKPOINT),
        "target_samples":
            len(target_features),
        "batch_size":
            BATCH_SIZE,
        "audited_batches":
            total_batches,
        "teacher_target_accuracy":
            100.0 * teacher_accuracy,
        "mean_confidence":
            float(
                target_confidence.mean().item()
            ),
        "mean_target_discrepancy":
            discrepancy_sum
            / max(
                total_batches,
                1
            ),
        "audit_seconds":
            elapsed,
        "batch_summaries":
            batch_records,
        "full_target_summaries":
            summaries,
        "promotion_mass":
            total_promotion.tolist(),
        "suppression_mass":
            total_suppression.tolist(),
        "promotion_share":
            promotion_share.tolist(),
        "suppression_share":
            suppression_share.tolist(),
        "truck_promotion_share":
            float(
                promotion_share[
                    TRUCK_ID
                ].item()
            ),
        "truck_suppression_share":
            float(
                suppression_share[
                    TRUCK_ID
                ].item()
            ),
        "truck_head_gradient_alignment":
            truck_alignment,
        "class_head_gradient_alignment":
            alignment_matrix.tolist()
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
        "FULL GRADIENT SNR AUDIT COMPLETE"
    )
    print("=" * 90)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()