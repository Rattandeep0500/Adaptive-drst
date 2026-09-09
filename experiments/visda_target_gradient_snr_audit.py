import json
import random
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
TARGET_BATCHES = 1

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/"
    "geometry_gated_mcd_seed42.pt"
)

CACHE_ROOT = Path(
    "checkpoints/visda_feature_cache"
)

TARGET_CACHE = (
    CACHE_ROOT / "target"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_target_gradient_snr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "target_gradient_snr_seed42.json"
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
                f"Invalid feature tensor in {path}: "
                f"{tuple(x.shape)}"
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
            f"Final feature dimension mismatch: "
            f"{features.shape[1]}"
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

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected {HIDDEN_DIM}-D output, "
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
                f"Expected 2-D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D input, "
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


def load_checkpoint(
    student,
    teacher
):
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{BASE_CHECKPOINT}"
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
def collect_teacher_predictions(
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

        probabilities = (
            p1 + p2
        ) / 2.0

        confidence, predictions = (
            probabilities.max(
                dim=1
            )
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


def per_sample_class_statistics(
    gradients,
    predictions
):
    results = {}

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            predictions
            == class_id
        )

        name = CLASSES[
            class_id
        ]

        if not mask.any():
            results[name] = {
                "count": 0,
                "mean_norm": 0.0,
                "mean_vector_norm": 0.0,
                "rms_sample_norm": 0.0,
                "mean_cosine": 0.0,
                "cosine_std": 0.0,
                "snr": 0.0,
                "mean_abs_gradient": 0.0
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
            "rms_sample_norm":
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
                )
        }

    return results


def class_gradient_vectors(
    model,
    x,
    predictions
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

    per_sample_discrepancy = (
        p1 - p2
    ).abs().mean(
        dim=1
    )

    parameters = [
        model.classifier1.weight,
        model.classifier1.bias,
        model.classifier2.weight,
        model.classifier2.bias
    ]

    vectors = {}

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            predictions
            == class_id
        )

        name = CLASSES[
            class_id
        ]

        if not mask.any():
            vectors[name] = None
            continue

        selected_loss = (
            per_sample_discrepancy[
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

        vector = torch.cat(
            [
                grad.detach().reshape(-1)
                for grad in grads
            ]
        )

        vectors[name] = vector

    return vectors


def vector_cosine(
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


def print_class_statistics(
    statistics
):
    print()
    print(
        "PER-CLASS GRADIENT SNR"
    )

    for class_name in CLASSES:
        row = statistics[
            class_name
        ]

        if row["count"] == 0:
            print(
                f"{class_name:12s} | count=0"
            )
            continue

        print(
            f"{class_name:12s} | "
            f"n={row['count']:4d} | "
            f"mean_norm={row['mean_norm']:.8f} | "
            f"mean_cos={row['mean_cosine']:.6f} | "
            f"cos_std={row['cosine_std']:.6f} | "
            f"SNR={row['snr']:.6f}"
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
        "VISDA-2017 TARGET GRADIENT SNR AUDIT"
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
        f"target_batches={TARGET_BATCHES}"
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

    student = MCDModel().to(
        device
    )

    teacher = MCDModel().to(
        device
    )

    load_checkpoint(
        student,
        teacher
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

    (
        target_predictions,
        target_confidence
    ) = collect_teacher_predictions(
        teacher,
        target_features,
        device
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

    gradient_batches = []
    processed_batches = 0
    discrepancy_sum = 0.0
    audited_samples = 0

    print()

    print(
        "Computing direct logit gradients..."
    )

    for raw_x, prediction_batch in loader:
        if processed_batches >= TARGET_BATCHES:
            break

        raw_x = raw_x.to(
            device
        )

        prediction_batch = prediction_batch.to(
            device
        )

        if raw_x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                "Target input dimension mismatch"
            )

        gradients, loss = (
            direct_logit_gradients(
                student,
                raw_x
            )
        )

        gradient_batches.append(
            gradients.cpu()
        )

        gradient_batches.append(
            prediction_batch.cpu()
        )

        discrepancy_sum += (
            loss.item()
        )

        audited_samples += (
            len(raw_x)
        )

        processed_batches += 1

    gradient_tensors = []
    prediction_tensors = []

    for i in range(
        0,
        len(gradient_batches),
        2
    ):
        gradient_tensors.append(
            gradient_batches[i]
        )

        prediction_tensors.append(
            gradient_batches[i + 1]
        )

    gradients = torch.cat(
        gradient_tensors,
        dim=0
    )

    predictions = torch.cat(
        prediction_tensors,
        dim=0
    )

    discrepancy_mean = (
        discrepancy_sum
        / max(
            processed_batches,
            1
        )
    )

    print()
    print(
        f"Audited samples: "
        f"{audited_samples}"
    )

    print(
        f"Mean target discrepancy: "
        f"{discrepancy_mean:.8f}"
    )

    statistics = (
        per_sample_class_statistics(
            gradients,
            predictions
        )
    )

    print_class_statistics(
        statistics
    )

    print()
    print(
        "SINK VS TRUCK"
    )

    for class_id in (
        TRUCK_ID,
        CAR_ID,
        BUS_ID,
        TRAIN_ID
    ):
        name = CLASSES[
            class_id
        ]

        row = statistics[
            name
        ]

        print(
            f"{name:12s} | "
            f"mean_norm={row['mean_norm']:.8f} | "
            f"mean_cos={row['mean_cosine']:.6f} | "
            f"SNR={row['snr']:.6f}"
        )

    print()
    print(
        "CLASS-WISE HEAD GRADIENT VECTORS"
    )

    first_batch_x = target_features[
        :BATCH_SIZE
    ].to(
        device
    )

    first_batch_predictions = (
        target_predictions[
            :BATCH_SIZE
        ].to(
            device
        )
    )

    vectors = class_gradient_vectors(
        student,
        first_batch_x,
        first_batch_predictions
    )

    truck_vector = vectors[
        "truck"
    ]

    print()
    print(
        "TRUCK HEAD-GRADIENT ALIGNMENT"
    )

    alignment = {}

    for name in (
        "car",
        "bus",
        "train"
    ):
        value = vector_cosine(
            truck_vector,
            vectors[name]
        )

        alignment[
            name
        ] = value

        print(
            f"truck vs {name:5s}: "
            f"{value}"
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

    promotion_share = (
        promotion
        / promotion.sum().clamp_min(
            1e-12
        )
    )

    suppression_share = (
        suppression
        / suppression.sum().clamp_min(
            1e-12
        )
    )

    print()
    print(
        "PROMOTION / SUPPRESSION SHARES"
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

    result = {
        "experiment":
            "visda_target_gradient_snr_audit",
        "seed":
            SEED,
        "checkpoint":
            str(BASE_CHECKPOINT),
        "audited_samples":
            audited_samples,
        "audited_batches":
            processed_batches,
        "teacher_target_accuracy":
            100.0 * teacher_accuracy,
        "mean_confidence":
            float(
                target_confidence.mean().item()
            ),
        "mean_target_discrepancy":
            discrepancy_mean,
        "class_statistics":
            statistics,
        "truck_head_alignment":
            alignment,
        "promotion_mass":
            promotion.tolist(),
        "suppression_mass":
            suppression.tolist(),
        "promotion_share":
            promotion_share.tolist(),
        "suppression_share":
            suppression_share.tolist()
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
        "GRADIENT SNR AUDIT COMPLETE"
    )
    print("=" * 90)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()