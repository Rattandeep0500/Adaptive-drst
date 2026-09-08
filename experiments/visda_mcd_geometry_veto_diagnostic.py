import json
import random
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

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_mcd_geometry_veto"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 4096

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
                f"Invalid feature shape in {path}: "
                f"{tuple(batch_features.shape)}"
            )

        if batch_features.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features in {path}, "
                f"got {batch_features.shape[1]}"
            )

        if len(batch_features) != len(
            batch_labels
        ):
            raise RuntimeError(
                f"Feature/label count mismatch in {path}"
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

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Final feature dimension is "
            f"{features.shape[1]}, expected {INPUT_DIM}"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Final feature and label counts do not match"
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
                f"Adapter expected {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter expected {HIDDEN_DIM}-D output, "
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

        if logits1.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                f"Invalid classifier output shape "
                f"{tuple(logits1.shape)}"
            )

        if logits2.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                f"Invalid classifier output shape "
                f"{tuple(logits2.shape)}"
            )

        return logits1, logits2


def extract_state_dict(payload):
    if not isinstance(
        payload,
        dict
    ):
        raise RuntimeError(
            "Checkpoint payload is not a dictionary"
        )

    candidates = [
        "model_state_dict",
        "student_state_dict",
        "state_dict",
    ]

    for key in candidates:
        value = payload.get(
            key
        )

        if isinstance(
            value,
            dict
        ):
            return value

    tensor_items = {
        key: value
        for key, value in payload.items()
        if torch.is_tensor(value)
    }

    if tensor_items:
        return tensor_items

    raise RuntimeError(
        "Could not find a model state dictionary in checkpoint"
    )


def normalize_state_dict_keys(
    state_dict
):
    normalized = {}

    for key, value in state_dict.items():
        new_key = key

        prefixes = [
            "module.",
            "model.",
            "student.",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        normalized[
            new_key
        ] = value

    return normalized


def load_mcd_checkpoint(
    model
):
    if not MCD_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{MCD_CHECKPOINT}"
        )

    payload = torch.load(
        MCD_CHECKPOINT,
        map_location="cpu"
    )

    state_dict = extract_state_dict(
        payload
    )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    expected_keys = set(
        model.state_dict().keys()
    )

    actual_keys = set(
        state_dict.keys()
    )

    missing = sorted(
        expected_keys
        - actual_keys
    )

    unexpected = sorted(
        actual_keys
        - expected_keys
    )

    if missing:
        raise RuntimeError(
            "Checkpoint is missing keys:\n"
            + "\n".join(
                missing
            )
        )

    if unexpected:
        raise RuntimeError(
            "Checkpoint contains unexpected keys:\n"
            + "\n".join(
                unexpected
            )
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    return payload


@torch.no_grad()
def collect_model_outputs(
    model,
    features,
    device
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features
        ),
        batch_size=BATCH_SIZE,
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

        if x.ndim != 2:
            raise RuntimeError(
                f"Batch input must be 2D, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Batch input must be {INPUT_DIM}-D, "
                f"got {x.shape[1]}-D"
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

        probabilities = (
            p1 + p2
        ) / 2.0

        confidence, predictions = (
            probabilities.max(
                dim=1
            )
        )

        disagreement = torch.mean(
            torch.abs(
                p1 - p2
            ),
            dim=1
        )

        z = model.encode(
            x
        )

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapted feature dimension must be "
                f"{HIDDEN_DIM}, got {z.shape[1]}"
            )

        z = F.normalize(
            z,
            dim=1
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

    if len(z) != len(features):
        raise RuntimeError(
            "Collected output count does not match input count"
        )

    return (
        z,
        probabilities,
        confidence,
        predictions,
        disagreement,
    )


def compute_source_prototypes(
    source_z,
    source_labels
):
    if source_z.ndim != 2:
        raise RuntimeError(
            f"Source representation must be 2D, "
            f"got {tuple(source_z.shape)}"
        )

    if source_z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            f"Source representation must be "
            f"{HIDDEN_DIM}-D, got {source_z.shape[1]}"
        )

    prototypes = []
    counts = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            source_labels
            == class_id
        )

        count = int(
            mask.sum().item()
        )

        if count == 0:
            raise RuntimeError(
                f"No source samples for class {class_id}"
            )

        class_z = source_z[
            mask
        ]

        prototype = class_z.mean(
            dim=0
        )

        prototype = F.normalize(
            prototype,
            dim=0
        )

        prototypes.append(
            prototype
        )

        counts.append(
            count
        )

    prototypes = torch.stack(
        prototypes,
        dim=0
    )

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Invalid prototype shape "
            f"{tuple(prototypes.shape)}"
        )

    return (
        prototypes,
        counts
    )


def compute_geometry(
    target_z,
    target_probabilities,
    target_predictions,
    prototypes
):
    if target_z.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            f"Target representation must be "
            f"{HIDDEN_DIM}-D"
        )

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Invalid prototype matrix shape "
            f"{tuple(prototypes.shape)}"
        )

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

    sample_count = len(
        target_z
    )

    row_index = torch.arange(
        sample_count
    )

    predicted_similarity = (
        similarities[
            row_index,
            target_predictions
        ]
    )

    top_similarity_values, top_similarity_indices = (
        torch.topk(
            similarities,
            k=min(
                3,
                NUM_CLASSES
            ),
            dim=1
        )
    )

    predicted_rank = (
        top_similarity_indices
        == target_predictions.unsqueeze(
            1
        )
    ).float().argmax(
        dim=1
    )

    probability_order = torch.argsort(
        target_probabilities,
        dim=1,
        descending=True
    )

    top1_prediction = (
        probability_order[:, 0]
    )

    second_prediction = (
        probability_order[:, 1]
    )

    second_probability = (
        target_probabilities[
            row_index,
            second_prediction
        ]
    )

    second_similarity = (
        similarities[
            row_index,
            second_prediction
        ]
    )

    geometry_margin = (
        predicted_similarity
        - second_similarity
    )

    true_similarity = similarities

    return {
        "similarities":
            similarities,
        "top_similarity_values":
            top_similarity_values,
        "top_similarity_indices":
            top_similarity_indices,
        "predicted_similarity":
            predicted_similarity,
        "predicted_rank":
            predicted_rank,
        "second_prediction":
            second_prediction,
        "second_probability":
            second_probability,
        "second_similarity":
            second_similarity,
        "geometry_margin":
            geometry_margin,
        "true_similarity":
            true_similarity,
        "top1_prediction":
            top1_prediction,
    }


def binary_auc_for_correctness(
    score,
    correctness,
    descending=True
):
    score = score.float()
    correctness = correctness.bool()

    n = len(score)

    positive = int(
        correctness.sum().item()
    )

    negative = n - positive

    if positive == 0 or negative == 0:
        return float("nan")

    order = torch.argsort(
        score,
        descending=descending
    )

    ordered_correctness = (
        correctness[
            order
        ].float()
    )

    ranks = torch.arange(
        1,
        n + 1,
        dtype=torch.float64
    )

    positive_rank_sum = (
        ranks[
            ordered_correctness == 1
        ].sum().item()
    )

    auc = (
        positive_rank_sum
        - positive
        * (positive + 1)
        / 2.0
    ) / (
        positive
        * negative
    )

    return float(
        auc
    )


def precision_at_coverage(
    score,
    correctness,
    coverage,
    descending=True
):
    count = len(
        score
    )

    k = max(
        1,
        int(
            count
            * coverage
        )
    )

    order = torch.argsort(
        score,
        descending=descending
    )

    selected = correctness[
        order[:k]
    ].float()

    precision = float(
        selected.mean().item()
    )

    return precision


def risk_profile(
    score,
    correctness,
    descending=True
):
    auc = binary_auc_for_correctness(
        score,
        correctness,
        descending=descending
    )

    coverages = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    precision = {}

    for coverage in coverages:
        precision[
            f"{int(coverage * 100)}%"
        ] = precision_at_coverage(
            score,
            correctness,
            coverage,
            descending=descending
        )

    return {
        "auc":
            auc,
        "precision":
            precision,
    }


def evaluate_threshold(
    score,
    correctness,
    threshold,
    higher_is_better=True
):
    if higher_is_better:
        selected = (
            score
            >= threshold
        )
    else:
        selected = (
            score
            <= threshold
        )

    count = int(
        selected.sum().item()
    )

    if count == 0:
        return {
            "threshold":
                threshold,
            "selected":
                0,
            "coverage":
                0.0,
            "precision":
                float("nan"),
            "error_rate":
                float("nan"),
        }

    precision = float(
        correctness[
            selected
        ].float().mean().item()
    )

    return {
        "threshold":
            threshold,
        "selected":
            count,
        "coverage":
            count / len(score),
        "precision":
            precision,
        "error_rate":
            1.0 - precision,
    }


def classwise_geometry(
    target_predictions,
    target_labels,
    confidence,
    disagreement,
    predicted_similarity,
    geometry_margin
):
    results = {}

    for class_id, class_name in enumerate(
        CLASSES
    ):
        predicted_mask = (
            target_predictions
            == class_id
        )

        true_mask = (
            target_labels
            == class_id
        )

        predicted_count = int(
            predicted_mask.sum().item()
        )

        true_count = int(
            true_mask.sum().item()
        )

        if predicted_count == 0:
            results[
                class_name
            ] = {
                "predicted_count":
                    0,
                "true_count":
                    true_count,
            }
            continue

        class_correctness = (
            target_predictions[
                predicted_mask
            ]
            == target_labels[
                predicted_mask
            ]
        )

        results[
            class_name
        ] = {
            "predicted_count":
                predicted_count,
            "true_count":
                true_count,
            "accuracy":
                float(
                    class_correctness.float().mean().item()
                ),
            "confidence":
                float(
                    confidence[
                        predicted_mask
                    ].mean().item()
                ),
            "disagreement":
                float(
                    disagreement[
                        predicted_mask
                    ].mean().item()
                ),
            "prototype_similarity":
                float(
                    predicted_similarity[
                        predicted_mask
                    ].mean().item()
                ),
            "prototype_margin":
                float(
                    geometry_margin[
                        predicted_mask
                    ].mean().item()
                ),
        }

    return results


def confusion_geometry(
    target_predictions,
    target_labels,
    confidence,
    disagreement,
    predicted_similarity,
    geometry_margin,
    second_similarity
):
    class_to_id = {
        name: index
        for index, name in enumerate(
            CLASSES
        )
    }

    pairs = [
        (
            "truck",
            "car"
        ),
        (
            "truck",
            "bus"
        ),
        (
            "truck",
            "train"
        ),
        (
            "skateboard",
            "person"
        ),
        (
            "skateboard",
            "knife"
        ),
        (
            "bicycle",
            "motorcycle"
        ),
        (
            "person",
            "knife"
        ),
        (
            "car",
            "person"
        ),
        (
            "car",
            "knife"
        ),
    ]

    results = {}

    for true_name, pred_name in pairs:
        true_id = class_to_id[
            true_name
        ]

        pred_id = class_to_id[
            pred_name
        ]

        mask = (
            (
                target_labels
                == true_id
            )
            & (
                target_predictions
                == pred_id
            )
        )

        count = int(
            mask.sum().item()
        )

        if count == 0:
            results[
                f"{true_name}_to_{pred_name}"
            ] = {
                "count":
                    0
            }
            continue

        results[
            f"{true_name}_to_{pred_name}"
        ] = {
            "count":
                count,
            "confidence":
                float(
                    confidence[
                        mask
                    ].mean().item()
                ),
            "disagreement":
                float(
                    disagreement[
                        mask
                    ].mean().item()
                ),
            "predicted_prototype_similarity":
                float(
                    predicted_similarity[
                        mask
                    ].mean().item()
                ),
            "prototype_margin":
                float(
                    geometry_margin[
                        mask
                    ].mean().item()
                ),
            "second_prototype_similarity":
                float(
                    second_similarity[
                        mask
                    ].mean().item()
                ),
        }

    return results


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
        "VISDA-2017 MCD GEOMETRY VETO DIAGNOSTIC"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"checkpoint={MCD_CHECKPOINT}"
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

    print(
        f"Source feature dimension: "
        f"{source_features.shape[1]}"
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

    print(
        f"Target feature dimension: "
        f"{target_features.shape[1]}"
    )

    if source_features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Source cache dimension assertion failed"
        )

    if target_features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Target cache dimension assertion failed"
        )

    model = MCDModel().to(
        device
    )

    checkpoint_payload = load_mcd_checkpoint(
        model
    )

    model.eval()

    print()
    print(
        "Checkpoint loaded successfully."
    )

    print(
        "Architecture:"
    )

    print(
        f"input={INPUT_DIM}"
    )

    print(
        f"hidden={HIDDEN_DIM}"
    )

    print(
        "normalization=BatchNorm1d"
    )

    print()
    print(
        "Collecting source adapted features..."
    )

    (
        source_z,
        source_probabilities,
        source_confidence,
        source_predictions,
        source_disagreement,
    ) = collect_model_outputs(
        model,
        source_features,
        device
    )

    print(
        f"Source adapted shape: "
        f"{tuple(source_z.shape)}"
    )

    if source_z.shape != (
        len(source_features),
        HIDDEN_DIM,
    ):
        raise RuntimeError(
            f"Invalid source adapted shape: "
            f"{tuple(source_z.shape)}"
        )

    print()
    print(
        "Computing source class prototypes..."
    )

    (
        source_prototypes,
        source_counts
    ) = compute_source_prototypes(
        source_z,
        source_labels
    )

    print(
        f"Prototype shape: "
        f"{tuple(source_prototypes.shape)}"
    )

    for class_name, count in zip(
        CLASSES,
        source_counts,
    ):
        print(
            f"{class_name:12s}: "
            f"{count:6d}"
        )

    print()
    print(
        "Collecting target MCD outputs..."
    )

    (
        target_z,
        target_probabilities,
        target_confidence,
        target_predictions,
        target_disagreement,
    ) = collect_model_outputs(
        model,
        target_features,
        device
    )

    print(
        f"Target adapted shape: "
        f"{tuple(target_z.shape)}"
    )

    if target_z.shape != (
        len(target_features),
        HIDDEN_DIM,
    ):
        raise RuntimeError(
            f"Invalid target adapted shape: "
            f"{tuple(target_z.shape)}"
        )

    if len(target_predictions) != len(
        target_labels
    ):
        raise RuntimeError(
            "Target prediction count mismatch"
        )

    correctness = (
        target_predictions
        == target_labels
    )

    baseline_overall = float(
        correctness.float().mean().item()
        * 100.0
    )

    per_class_correct = []
    per_class_total = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            target_labels
            == class_id
        )

        total = int(
            mask.sum().item()
        )

        correct = int(
            (
                target_predictions[
                    mask
                ]
                == target_labels[
                    mask
                ]
            ).sum().item()
        )

        per_class_total.append(
            total
        )

        per_class_correct.append(
            correct
        )

    per_class_accuracy = (
        100.0
        * torch.tensor(
            per_class_correct,
            dtype=torch.float32
        )
        / torch.tensor(
            per_class_total,
            dtype=torch.float32
        ).clamp_min(1.0)
    )

    baseline_mean_class = float(
        per_class_accuracy.mean().item()
    )

    print()
    print("=" * 90)
    print(
        "BASELINE"
    )
    print("=" * 90)

    print(
        f"Target overall accuracy: "
        f"{baseline_overall:.2f}%"
    )

    print(
        f"Target mean-class accuracy: "
        f"{baseline_mean_class:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{target_confidence.mean().item():.6f}"
    )

    print(
        f"Mean disagreement: "
        f"{target_disagreement.mean().item():.6f}"
    )

    print()
    print(
        "PER-CLASS BASELINE"
    )

    for class_name, accuracy, total in zip(
        CLASSES,
        per_class_accuracy.tolist(),
        per_class_total,
    ):
        print(
            f"{class_name:12s} | "
            f"n={total:6d} | "
            f"accuracy={accuracy:6.2f}%"
        )

    print()
    print("=" * 90)
    print(
        "GEOMETRY"
    )
    print("=" * 90)

    geometry = compute_geometry(
        target_z,
        target_probabilities,
        target_predictions,
        source_prototypes,
    )

    predicted_similarity = (
        geometry[
            "predicted_similarity"
        ]
    )

    geometry_margin = (
        geometry[
            "geometry_margin"
        ]
    )

    second_prediction = (
        geometry[
            "second_prediction"
        ]
    )

    second_probability = (
        geometry[
            "second_probability"
        ]
    )

    second_similarity = (
        geometry[
            "second_similarity"
        ]
    )

    predicted_rank = (
        geometry[
            "predicted_rank"
        ]
    )

    print(
        f"Mean predicted-prototype similarity: "
        f"{predicted_similarity.mean().item():.6f}"
    )

    print(
        f"Mean prototype margin: "
        f"{geometry_margin.mean().item():.6f}"
    )

    print(
        f"Fraction prototype rank=1: "
        f"{100.0 * (predicted_rank == 0).float().mean().item():.2f}%"
    )

    print()
    print(
        "CORRECT VS WRONG"
    )

    correct_mask = correctness
    wrong_mask = ~correctness

    for name, values in [
        (
            "confidence",
            target_confidence,
        ),
        (
            "disagreement",
            target_disagreement,
        ),
        (
            "prototype_similarity",
            predicted_similarity,
        ),
        (
            "prototype_margin",
            geometry_margin,
        ),
    ]:
        correct_values = values[
            correct_mask
        ]

        wrong_values = values[
            wrong_mask
        ]

        print()
        print(
            name
        )

        print(
            f"correct mean: "
            f"{correct_values.mean().item():.6f}"
        )

        print(
            f"wrong mean: "
            f"{wrong_values.mean().item():.6f}"
        )

        print(
            f"correct median: "
            f"{correct_values.median().item():.6f}"
        )

        print(
            f"wrong median: "
            f"{wrong_values.median().item():.6f}"
        )

    print()
    print("=" * 90)
    print(
        "RANKING SIGNALS"
    )
    print("=" * 90)

    confidence_risk = risk_profile(
        target_confidence,
        correctness,
        descending=True,
    )

    disagreement_risk = risk_profile(
        target_disagreement,
        correctness,
        descending=False,
    )

    similarity_risk = risk_profile(
        predicted_similarity,
        correctness,
        descending=True,
    )

    margin_risk = risk_profile(
        geometry_margin,
        correctness,
        descending=True,
    )

    combined_confidence_similarity = (
        target_confidence
        * (
            predicted_similarity
            + 1.0
        )
        / 2.0
    )

    combined_confidence_margin = (
        target_confidence
        * torch.sigmoid(
            10.0
            * geometry_margin
        )
    )

    combined_similarity_margin = (
        (
            predicted_similarity
            + 1.0
        )
        / 2.0
        * torch.sigmoid(
            10.0
            * geometry_margin
        )
    )

    combined_all = (
        target_confidence
        * (
            predicted_similarity
            + 1.0
        )
        / 2.0
        * torch.sigmoid(
            10.0
            * geometry_margin
        )
    )

    combined_results = {
        "confidence_x_similarity":
            risk_profile(
                combined_confidence_similarity,
                correctness,
                descending=True,
            ),
        "confidence_x_margin":
            risk_profile(
                combined_confidence_margin,
                correctness,
                descending=True,
            ),
        "similarity_x_margin":
            risk_profile(
                combined_similarity_margin,
                correctness,
                descending=True,
            ),
        "confidence_x_similarity_x_margin":
            risk_profile(
                combined_all,
                correctness,
                descending=True,
            ),
    }

    ranking_table = [
        (
            "confidence",
            confidence_risk[
                "auc"
            ],
        ),
        (
            "disagreement",
            disagreement_risk[
                "auc"
            ],
        ),
        (
            "prototype_similarity",
            similarity_risk[
                "auc"
            ],
        ),
        (
            "prototype_margin",
            margin_risk[
                "auc"
            ],
        ),
        (
            "confidence_x_similarity",
            combined_results[
                "confidence_x_similarity"
            ]["auc"],
        ),
        (
            "confidence_x_margin",
            combined_results[
                "confidence_x_margin"
            ]["auc"],
        ),
        (
            "similarity_x_margin",
            combined_results[
                "similarity_x_margin"
            ]["auc"],
        ),
        (
            "confidence_x_similarity_x_margin",
            combined_results[
                "confidence_x_similarity_x_margin"
            ]["auc"],
        ),
    ]

    ranking_table = sorted(
        ranking_table,
        key=lambda x: (
            -x[1]
            if np.isfinite(x[1])
            else float("inf")
        ),
    )

    for index, (
        name,
        auc,
    ) in enumerate(
        ranking_table,
        start=1,
    ):
        print(
            f"{index}. "
            f"{name:36s} | "
            f"AUC={auc:.6f}"
        )

    print()
    print(
        "Precision by coverage"
    )

    for name, result in [
        (
            "confidence",
            confidence_risk,
        ),
        (
            "prototype_similarity",
            similarity_risk,
        ),
        (
            "prototype_margin",
            margin_risk,
        ),
        (
            "confidence_x_similarity",
            combined_results[
                "confidence_x_similarity"
            ],
        ),
        (
            "confidence_x_margin",
            combined_results[
                "confidence_x_margin"
            ],
        ),
        (
            "confidence_x_similarity_x_margin",
            combined_results[
                "confidence_x_similarity_x_margin"
            ],
        ),
    ]:
        print()
        print(
            name
        )

        for coverage, precision in (
            result[
                "precision"
            ].items()
        ):
            if coverage in [
                "30%",
                "50%",
                "70%",
                "90%",
            ]:
                print(
                    f"  {coverage:>4s} | "
                    f"precision={precision * 100.0:.2f}%"
                )

    print()
    print("=" * 90)
    print(
        "GEOMETRIC THRESHOLD ANALYSIS"
    )
    print("=" * 90)

    similarity_thresholds = [
        -0.25,
        -0.10,
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    for threshold in similarity_thresholds:
        result = evaluate_threshold(
            predicted_similarity,
            correctness,
            threshold,
            higher_is_better=True,
        )

        print(
            f"similarity >= "
            f"{threshold:5.2f} | "
            f"coverage={result['coverage'] * 100.0:6.2f}% | "
            f"precision={result['precision'] * 100.0:6.2f}% | "
            f"error={result['error_rate'] * 100.0:6.2f}%"
        )

    print()
    print(
        "Prototype margin thresholds"
    )

    margin_thresholds = [
        -0.30,
        -0.20,
        -0.10,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
    ]

    for threshold in margin_thresholds:
        result = evaluate_threshold(
            geometry_margin,
            correctness,
            threshold,
            higher_is_better=True,
        )

        print(
            f"margin >= "
            f"{threshold:5.2f} | "
            f"coverage={result['coverage'] * 100.0:6.2f}% | "
            f"precision={result['precision'] * 100.0:6.2f}% | "
            f"error={result['error_rate'] * 100.0:6.2f}%"
        )

    print()
    print("=" * 90)
    print(
        "PER-CLASS GEOMETRY"
    )
    print("=" * 90)

    class_geometry = classwise_geometry(
        target_predictions,
        target_labels,
        target_confidence,
        target_disagreement,
        predicted_similarity,
        geometry_margin,
    )

    for class_name in CLASSES:
        row = class_geometry[
            class_name
        ]

        if row.get(
            "predicted_count",
            0
        ) == 0:
            print(
                f"{class_name:12s} | "
                f"pred=0 | "
                f"true={row['true_count']:5d}"
            )
            continue

        print(
            f"{class_name:12s} | "
            f"pred={row['predicted_count']:5d} | "
            f"true={row['true_count']:5d} | "
            f"acc={row['accuracy']:6.2f}% | "
            f"conf={row['confidence']:.4f} | "
            f"disc={row['disagreement']:.4f} | "
            f"sim={row['prototype_similarity']:.4f} | "
            f"margin={row['prototype_margin']:.4f}"
        )

    print()
    print("=" * 90)
    print(
        "KEY CONFUSION GEOMETRY"
    )
    print("=" * 90)

    confusion_results = confusion_geometry(
        target_predictions,
        target_labels,
        target_confidence,
        target_disagreement,
        predicted_similarity,
        geometry_margin,
        second_similarity,
    )

    for pair_name, row in (
        confusion_results.items()
    ):
        if row["count"] == 0:
            print(
                f"{pair_name:32s} | n=0"
            )
            continue

        print(
            f"{pair_name:32s} | "
            f"n={row['count']:5d} | "
            f"conf={row['confidence']:.4f} | "
            f"disc={row['disagreement']:.4f} | "
            f"sim={row['predicted_prototype_similarity']:.4f} | "
            f"margin={row['prototype_margin']:.4f} | "
            f"second={row['second_prototype_similarity']:.4f}"
        )

    print()
    print("=" * 90)
    print(
        "VETO POTENTIAL"
    )
    print("=" * 90)

    low_similarity_threshold = 0.20

    wrong_low_similarity = (
        predicted_similarity[
            wrong_mask
        ]
        < low_similarity_threshold
    )

    correct_low_similarity = (
        predicted_similarity[
            correct_mask
        ]
        < low_similarity_threshold
    )

    wrong_negative_margin = (
        geometry_margin[
            wrong_mask
        ]
        < 0.0
    )

    correct_negative_margin = (
        geometry_margin[
            correct_mask
        ]
        < 0.0
    )

    print(
        f"Wrong samples with similarity < "
        f"{low_similarity_threshold:.2f}: "
        f"{100.0 * wrong_low_similarity.float().mean().item():.2f}%"
    )

    print(
        f"Correct samples with similarity < "
        f"{low_similarity_threshold:.2f}: "
        f"{100.0 * correct_low_similarity.float().mean().item():.2f}%"
    )

    print(
        f"Wrong samples with negative geometry margin: "
        f"{100.0 * wrong_negative_margin.float().mean().item():.2f}%"
    )

    print(
        f"Correct samples with negative geometry margin: "
        f"{100.0 * correct_negative_margin.float().mean().item():.2f}%"
    )

    print()
    print("=" * 90)
    print(
        "TOP-2 GEOMETRY AGREEMENT"
    )
    print("=" * 90)

    prototype_rank1 = (
        predicted_rank == 0
    )

    rank1_correct = (
        correctness[
            prototype_rank1
        ]
        .float()
        .mean()
        .item()
    )

    rank1_coverage = (
        prototype_rank1.float().mean().item()
    )

    print(
        f"Prototype rank=1 coverage: "
        f"{rank1_coverage * 100.0:.2f}%"
    )

    print(
        f"Accuracy among prototype rank=1: "
        f"{rank1_correct * 100.0:.2f}%"
    )

    print()
    print(
        "Prototype rank agreement with classifier:"
    )

    rank_agreement = (
        predicted_rank == 0
    )

    print(
        f"{100.0 * rank_agreement.float().mean().item():.2f}%"
    )

    output = {
        "experiment":
            "visda_mcd_geometry_veto_diagnostic",
        "seed":
            SEED,
        "device":
            str(device),
        "checkpoint":
            str(MCD_CHECKPOINT),
        "architecture": {
            "input_dim":
                INPUT_DIM,
            "hidden_dim":
                HIDDEN_DIM,
            "normalization":
                "BatchNorm1d",
        },
        "source_samples":
            len(source_features),
        "target_samples":
            len(target_features),
        "baseline": {
            "overall_accuracy":
                baseline_overall,
            "mean_class_accuracy":
                baseline_mean_class,
            "mean_confidence":
                float(
                    target_confidence.mean().item()
                ),
            "mean_disagreement":
                float(
                    target_disagreement.mean().item()
                ),
        },
        "prototype_shape":
            list(
                source_prototypes.shape
            ),
        "source_class_counts":
            {
                class_name:
                    count
                for class_name, count in zip(
                    CLASSES,
                    source_counts,
                )
            },
        "geometry_summary": {
            "mean_predicted_similarity":
                float(
                    predicted_similarity.mean().item()
                ),
            "mean_geometry_margin":
                float(
                    geometry_margin.mean().item()
                ),
            "prototype_rank1_fraction":
                float(
                    prototype_rank1.float().mean().item()
                ),
        },
        "correct_vs_wrong": {
            "confidence": {
                "correct_mean":
                    float(
                        target_confidence[
                            correct_mask
                        ].mean().item()
                    ),
                "wrong_mean":
                    float(
                        target_confidence[
                            wrong_mask
                        ].mean().item()
                    ),
            },
            "disagreement": {
                "correct_mean":
                    float(
                        target_disagreement[
                            correct_mask
                        ].mean().item()
                    ),
                "wrong_mean":
                    float(
                        target_disagreement[
                            wrong_mask
                        ].mean().item()
                    ),
            },
            "prototype_similarity": {
                "correct_mean":
                    float(
                        predicted_similarity[
                            correct_mask
                        ].mean().item()
                    ),
                "wrong_mean":
                    float(
                        predicted_similarity[
                            wrong_mask
                        ].mean().item()
                    ),
            },
            "prototype_margin": {
                "correct_mean":
                    float(
                        geometry_margin[
                            correct_mask
                        ].mean().item()
                    ),
                "wrong_mean":
                    float(
                        geometry_margin[
                            wrong_mask
                        ].mean().item()
                    ),
            },
        },
        "ranking": {
            "confidence":
                confidence_risk,
            "disagreement":
                disagreement_risk,
            "prototype_similarity":
                similarity_risk,
            "prototype_margin":
                margin_risk,
            "combined":
                combined_results,
        },
        "class_geometry":
            class_geometry,
        "confusion_geometry":
            confusion_results,
        "veto_potential": {
            "wrong_similarity_below_0_20":
                float(
                    wrong_low_similarity.float().mean().item()
                ),
            "correct_similarity_below_0_20":
                float(
                    correct_low_similarity.float().mean().item()
                ),
            "wrong_negative_margin":
                float(
                    wrong_negative_margin.float().mean().item()
                ),
            "correct_negative_margin":
                float(
                    correct_negative_margin.float().mean().item()
                ),
        },
    }

    output_path = (
        OUTPUT_DIR
        / "geometry_veto_seed42.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    print()
    print("=" * 90)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 90)

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()