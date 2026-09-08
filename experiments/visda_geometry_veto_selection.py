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
    "checkpoints/visda_geometry_veto_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 4096

COVERAGES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

SIMILARITY_THRESHOLDS = [
    0.40,
    0.50,
    0.60,
    0.70,
]

MARGIN_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.20,
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
            map_location="cpu",
        )

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
            )

        batch_features = payload["features"].float()
        batch_labels = payload["labels"].long()

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

        if len(batch_features) != len(batch_labels):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
            )

        features.append(batch_features)
        labels.append(batch_labels)

    features = torch.cat(
        features,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Invalid final feature dimension: "
            f"{features.shape[1]}"
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
                HIDDEN_DIM,
            ),
            nn.BatchNorm1d(
                HIDDEN_DIM,
            ),
            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected 2D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D input, got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected {HIDDEN_DIM}-D output, got {z.shape[1]}-D"
            )

        return z


class MCDModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.adapter = Adapter()

        self.classifier1 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

        self.classifier2 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

    def encode(self, x):
        return self.adapter(x)

    def forward(self, x):
        z = self.encode(x)

        return (
            self.classifier1(z),
            self.classifier2(z),
        )


def extract_state_dict(payload):
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Checkpoint payload is not a dictionary"
        )

    for key in [
        "model_state_dict",
        "student_state_dict",
        "state_dict",
    ]:
        value = payload.get(key)

        if isinstance(value, dict):
            return value

    tensor_items = {
        key: value
        for key, value in payload.items()
        if torch.is_tensor(value)
    }

    if tensor_items:
        return tensor_items

    raise RuntimeError(
        "Could not locate model state dictionary"
    )


def normalize_state_dict_keys(state_dict):
    normalized = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True

        while changed:
            changed = False

            for prefix in [
                "module.",
                "model.",
                "student.",
            ]:
                if new_key.startswith(prefix):
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        normalized[new_key] = value

    return normalized


def load_checkpoint(model):
    if not MCD_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{MCD_CHECKPOINT}"
        )

    payload = torch.load(
        MCD_CHECKPOINT,
        map_location="cpu",
    )

    state_dict = extract_state_dict(
        payload
    )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    expected = set(
        model.state_dict().keys()
    )

    actual = set(
        state_dict.keys()
    )

    missing = sorted(
        expected - actual
    )

    unexpected = sorted(
        actual - expected
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint keys:\n"
            + "\n".join(missing)
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys:\n"
            + "\n".join(unexpected)
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return payload


@torch.no_grad()
def collect_outputs(
    model,
    features,
    device,
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    z_all = []
    p_all = []
    confidence_all = []
    predictions_all = []
    disagreement_all = []

    for (x,) in loader:
        x = x.to(device)

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Input dimension changed: {x.shape[1]}"
            )

        logits1, logits2 = model(x)

        p1 = F.softmax(
            logits1,
            dim=1,
        )

        p2 = F.softmax(
            logits2,
            dim=1,
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

        z = model.encode(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapted dimension changed: {z.shape[1]}"
            )

        z = F.normalize(
            z,
            dim=1,
        )

        z_all.append(z.cpu())
        p_all.append(probabilities.cpu())
        confidence_all.append(confidence.cpu())
        predictions_all.append(predictions.cpu())
        disagreement_all.append(disagreement.cpu())

    z = torch.cat(
        z_all,
        dim=0,
    )

    probabilities = torch.cat(
        p_all,
        dim=0,
    )

    confidence = torch.cat(
        confidence_all,
        dim=0,
    )

    predictions = torch.cat(
        predictions_all,
        dim=0,
    )

    disagreement = torch.cat(
        disagreement_all,
        dim=0,
    )

    if len(z) != len(features):
        raise RuntimeError(
            "Output count does not match feature count"
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
    source_labels,
):
    prototypes = []

    for class_id in range(NUM_CLASSES):
        mask = (
            source_labels
            == class_id
        )

        if not mask.any():
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
            dim=0,
        )

        prototypes.append(
            prototype
        )

    prototypes = torch.stack(
        prototypes,
        dim=0,
    )

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM,
    ):
        raise RuntimeError(
            f"Invalid prototype shape "
            f"{tuple(prototypes.shape)}"
        )

    return prototypes


def compute_geometry(
    target_z,
    probabilities,
    predictions,
    prototypes,
):
    target_z = F.normalize(
        target_z,
        dim=1,
    )

    prototypes = F.normalize(
        prototypes,
        dim=1,
    )

    similarities = (
        target_z
        @ prototypes.t()
    )

    row_index = torch.arange(
        len(target_z)
    )

    predicted_similarity = (
        similarities[
            row_index,
            predictions,
        ]
    )

    probability_order = torch.argsort(
        probabilities,
        dim=1,
        descending=True,
    )

    second_prediction = (
        probability_order[:, 1]
    )

    second_probability = (
        probabilities[
            row_index,
            second_prediction,
        ]
    )

    second_similarity = (
        similarities[
            row_index,
            second_prediction,
        ]
    )

    geometry_margin = (
        predicted_similarity
        - second_similarity
    )

    prototype_order = torch.argsort(
        similarities,
        dim=1,
        descending=True,
    )

    predicted_rank = (
        prototype_order
        == predictions.unsqueeze(1)
    ).float().argmax(
        dim=1
    )

    return {
        "similarities":
            similarities,
        "predicted_similarity":
            predicted_similarity,
        "second_prediction":
            second_prediction,
        "second_probability":
            second_probability,
        "second_similarity":
            second_similarity,
        "geometry_margin":
            geometry_margin,
        "predicted_rank":
            predicted_rank,
    }


def select_top_fraction(
    score,
    coverage,
):
    count = len(score)

    k = max(
        1,
        int(
            count
            * coverage
        ),
    )

    order = torch.argsort(
        score,
        descending=True,
    )

    selected = torch.zeros(
        count,
        dtype=torch.bool,
    )

    selected[
        order[:k]
    ] = True

    return selected


def policy_score(
    policy,
    confidence,
    similarity,
    margin,
):
    if policy == "confidence":
        return confidence

    if policy == "confidence_similarity":
        return (
            confidence
            * (
                similarity
                + 1.0
            )
            / 2.0
        )

    if policy == "confidence_margin":
        return (
            confidence
            * torch.sigmoid(
                10.0
                * margin
            )
        )

    if policy == "confidence_similarity_margin":
        return (
            confidence
            * (
                similarity
                + 1.0
            )
            / 2.0
            * torch.sigmoid(
                10.0
                * margin
            )
        )

    raise ValueError(
        f"Unknown policy: {policy}"
    )


def select_policy(
    policy,
    confidence,
    similarity,
    margin,
    coverage,
):
    if policy == "confidence":
        score = policy_score(
            policy,
            confidence,
            similarity,
            margin,
        )

        return select_top_fraction(
            score,
            coverage,
        )

    if policy == "confidence_similarity":
        score = policy_score(
            policy,
            confidence,
            similarity,
            margin,
        )

        return select_top_fraction(
            score,
            coverage,
        )

    if policy == "confidence_margin":
        score = policy_score(
            policy,
            confidence,
            similarity,
            margin,
        )

        return select_top_fraction(
            score,
            coverage,
        )

    if policy == "confidence_similarity_margin":
        score = policy_score(
            policy,
            confidence,
            similarity,
            margin,
        )

        return select_top_fraction(
            score,
            coverage,
        )

    raise ValueError(
        f"Unknown policy: {policy}"
    )


def class_distribution(
    predictions,
    selected,
):
    counts = torch.bincount(
        predictions[
            selected
        ],
        minlength=NUM_CLASSES,
    )

    total = max(
        int(
            selected.sum().item()
        ),
        1,
    )

    result = {}

    for class_id, class_name in enumerate(
        CLASSES
    ):
        count = int(
            counts[
                class_id
            ].item()
        )

        result[
            class_name
        ] = {
            "count":
                count,
            "fraction":
                count
                / total,
        }

    return result


def compute_precision(
    predictions,
    labels,
    selected,
):
    count = int(
        selected.sum().item()
    )

    if count == 0:
        return {
            "selected":
                0,
            "precision":
                float("nan"),
            "error_rate":
                float("nan"),
        }

    correct = (
        predictions[
            selected
        ]
        == labels[
            selected
        ]
    )

    precision = float(
        correct.float().mean().item()
    )

    return {
        "selected":
            count,
        "precision":
            precision,
        "error_rate":
            1.0 - precision,
    }


def confusion_counts(
    predictions,
    labels,
    selected,
):
    pairs = [
        ("truck", "car"),
        ("truck", "bus"),
        ("truck", "train"),
        ("skateboard", "person"),
        ("skateboard", "knife"),
        ("bicycle", "motorcycle"),
        ("person", "knife"),
        ("car", "person"),
        ("car", "knife"),
    ]

    class_to_id = {
        name: index
        for index, name in enumerate(
            CLASSES
        )
    }

    results = {}

    for true_name, predicted_name in pairs:
        true_id = class_to_id[
            true_name
        ]

        predicted_id = class_to_id[
            predicted_name
        ]

        mask = (
            selected
            & (
                labels
                == true_id
            )
            & (
                predictions
                == predicted_id
            )
        )

        results[
            f"{true_name}_to_{predicted_name}"
        ] = int(
            mask.sum().item()
        )

    return results


def summarize_selection(
    policy,
    coverage,
    selected,
    predictions,
    labels,
    confidence,
    similarity,
    margin,
):
    precision = compute_precision(
        predictions,
        labels,
        selected,
    )

    selected_confidence = confidence[
        selected
    ]

    selected_similarity = similarity[
        selected
    ]

    selected_margin = margin[
        selected
    ]

    result = {
        "policy":
            policy,
        "coverage_target":
            coverage,
        "actual_coverage":
            float(
                selected.float().mean().item()
            ),
        "selected":
            precision["selected"],
        "precision":
            precision["precision"],
        "error_rate":
            precision["error_rate"],
        "mean_confidence":
            float(
                selected_confidence.mean().item()
            ),
        "mean_prototype_similarity":
            float(
                selected_similarity.mean().item()
            ),
        "mean_prototype_margin":
            float(
                selected_margin.mean().item()
            ),
        "class_distribution":
            class_distribution(
                predictions,
                selected,
            ),
        "key_confusions":
            confusion_counts(
                predictions,
                labels,
                selected,
            ),
    }

    return result


def threshold_analysis(
    confidence,
    similarity,
    margin,
    predictions,
    labels,
):
    results = []

    for threshold in SIMILARITY_THRESHOLDS:
        selected = (
            similarity
            >= threshold
        )

        precision = compute_precision(
            predictions,
            labels,
            selected,
        )

        results.append(
            {
                "type":
                    "similarity",
                "threshold":
                    threshold,
                "actual_coverage":
                    float(
                        selected.float().mean().item()
                    ),
                "selected":
                    precision["selected"],
                "precision":
                    precision["precision"],
                "error_rate":
                    precision["error_rate"],
            }
        )

    for threshold in MARGIN_THRESHOLDS:
        selected = (
            margin
            >= threshold
        )

        precision = compute_precision(
            predictions,
            labels,
            selected,
        )

        results.append(
            {
                "type":
                    "margin",
                "threshold":
                    threshold,
                "actual_coverage":
                    float(
                        selected.float().mean().item()
                    ),
                "selected":
                    precision["selected"],
                "precision":
                    precision["precision"],
                "error_rate":
                    precision["error_rate"],
            }
        )

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
        "VISDA-2017 GEOMETRY VETO SELECTION DIAGNOSTIC"
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
        f"Source dimension: "
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
        f"Target dimension: "
        f"{target_features.shape[1]}"
    )

    model = MCDModel().to(
        device
    )

    load_checkpoint(
        model
    )

    model.eval()

    print()
    print(
        "Checkpoint loaded successfully."
    )

    print()
    print(
        "Collecting source adapted features..."
    )

    (
        source_z,
        _,
        _,
        _,
        _,
    ) = collect_outputs(
        model,
        source_features,
        device,
    )

    print(
        f"Source adapted shape: "
        f"{tuple(source_z.shape)}"
    )

    print()
    print(
        "Computing source prototypes..."
    )

    source_prototypes = (
        compute_source_prototypes(
            source_z,
            source_labels,
        )
    )

    print(
        f"Prototype shape: "
        f"{tuple(source_prototypes.shape)}"
    )

    print()
    print(
        "Collecting target outputs..."
    )

    (
        target_z,
        target_probabilities,
        target_confidence,
        target_predictions,
        target_disagreement,
    ) = collect_outputs(
        model,
        target_features,
        device,
    )

    print(
        f"Target adapted shape: "
        f"{tuple(target_z.shape)}"
    )

    print()
    print(
        "Computing target geometry..."
    )

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

    predicted_rank = (
        geometry[
            "predicted_rank"
        ]
    )

    target_correct = (
        target_predictions
        == target_labels
    )

    baseline_accuracy = float(
        target_correct.float().mean().item()
        * 100.0
    )

    print()
    print("=" * 90)
    print(
        "BASELINE"
    )
    print("=" * 90)

    print(
        f"Target accuracy: "
        f"{baseline_accuracy:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{target_confidence.mean().item():.6f}"
    )

    print(
        f"Mean prototype similarity: "
        f"{predicted_similarity.mean().item():.6f}"
    )

    print(
        f"Mean prototype margin: "
        f"{geometry_margin.mean().item():.6f}"
    )

    print(
        f"Prototype rank-1 agreement: "
        f"{100.0 * (predicted_rank == 0).float().mean().item():.2f}%"
    )

    policies = [
        "confidence",
        "confidence_similarity",
        "confidence_margin",
        "confidence_similarity_margin",
    ]

    all_results = []

    print()
    print("=" * 90)
    print(
        "SELECTION COMPARISON"
    )
    print("=" * 90)

    for policy in policies:
        print()
        print(
            policy
        )

        for coverage in COVERAGES:
            selected = select_policy(
                policy,
                target_confidence,
                predicted_similarity,
                geometry_margin,
                coverage,
            )

            result = summarize_selection(
                policy,
                coverage,
                selected,
                target_predictions,
                target_labels,
                target_confidence,
                predicted_similarity,
                geometry_margin,
            )

            all_results.append(
                result
            )

            print(
                f"{int(coverage * 100):3d}% | "
                f"selected={result['selected']:5d} | "
                f"precision="
                f"{100.0 * result['precision']:6.2f}% | "
                f"conf="
                f"{result['mean_confidence']:.4f} | "
                f"sim="
                f"{result['mean_prototype_similarity']:.4f} | "
                f"margin="
                f"{result['mean_prototype_margin']:.4f}"
            )

    print()
    print("=" * 90)
    print(
        "KEY FLOW COUNTS"
    )
    print("=" * 90)

    flow_keys = [
        "truck_to_car",
        "truck_to_bus",
        "truck_to_train",
        "skateboard_to_person",
        "skateboard_to_knife",
        "bicycle_to_motorcycle",
        "person_to_knife",
        "car_to_person",
        "car_to_knife",
    ]

    for coverage in COVERAGES:
        print()
        print(
            f"Coverage {int(coverage * 100)}%"
        )

        for policy in policies:
            result = next(
                item
                for item in all_results
                if item["policy"] == policy
                and item[
                    "coverage_target"
                ] == coverage
            )

            print(
                f"{policy:32s} | "
                + " | ".join(
                    f"{key}="
                    f"{result['key_confusions'][key]}"
                    for key in flow_keys
                )
            )

    print()
    print("=" * 90)
    print(
        "SELECTION CLASS DISTRIBUTIONS"
    )
    print("=" * 90)

    for coverage in COVERAGES:
        print()
        print(
            f"Coverage {int(coverage * 100)}%"
        )

        for policy in policies:
            result = next(
                item
                for item in all_results
                if item["policy"] == policy
                and item[
                    "coverage_target"
                ] == coverage
            )

            print()
            print(
                policy
            )

            distribution = result[
                "class_distribution"
            ]

            for class_name in CLASSES:
                row = distribution[
                    class_name
                ]

                print(
                    f"  {class_name:12s}: "
                    f"{row['count']:5d} "
                    f"({100.0 * row['fraction']:6.2f}%)"
                )

    print()
    print("=" * 90)
    print(
        "GEOMETRY THRESHOLD ANALYSIS"
    )
    print("=" * 90)

    threshold_results = threshold_analysis(
        target_confidence,
        predicted_similarity,
        geometry_margin,
        target_predictions,
        target_labels,
    )

    for result in threshold_results:
        print(
            f"{result['type']:12s} | "
            f"threshold="
            f"{result['threshold']:6.2f} | "
            f"coverage="
            f"{100.0 * result['actual_coverage']:6.2f}% | "
            f"precision="
            f"{100.0 * result['precision']:6.2f}%"
        )

    best_by_30 = [
        item
        for item in all_results
        if item[
            "coverage_target"
        ] == 0.30
    ]

    best_by_30 = sorted(
        best_by_30,
        key=lambda x: x[
            "precision"
        ],
        reverse=True,
    )

    print()
    print("=" * 90)
    print(
        "BEST POLICY AT 30% COVERAGE"
    )
    print("=" * 90)

    for index, item in enumerate(
        best_by_30,
        start=1,
    ):
        print(
            f"{index}. "
            f"{item['policy']:32s} | "
            f"precision="
            f"{100.0 * item['precision']:.2f}%"
        )

    best_by_50 = [
        item
        for item in all_results
        if item[
            "coverage_target"
        ] == 0.50
    ]

    best_by_50 = sorted(
        best_by_50,
        key=lambda x: x[
            "precision"
        ],
        reverse=True,
    )

    print()
    print("=" * 90)
    print(
        "BEST POLICY AT 50% COVERAGE"
    )
    print("=" * 90)

    for index, item in enumerate(
        best_by_50,
        start=1,
    ):
        print(
            f"{index}. "
            f"{item['policy']:32s} | "
            f"precision="
            f"{100.0 * item['precision']:.2f}%"
        )

    output = {
        "experiment":
            "visda_geometry_veto_selection",
        "seed":
            SEED,
        "checkpoint":
            str(MCD_CHECKPOINT),
        "target_samples":
            len(target_features),
        "input_dim":
            INPUT_DIM,
        "hidden_dim":
            HIDDEN_DIM,
        "baseline_accuracy":
            baseline_accuracy,
        "mean_confidence":
            float(
                target_confidence.mean().item()
            ),
        "mean_disagreement":
            float(
                target_disagreement.mean().item()
            ),
        "mean_prototype_similarity":
            float(
                predicted_similarity.mean().item()
            ),
        "mean_prototype_margin":
            float(
                geometry_margin.mean().item()
            ),
        "prototype_rank1_fraction":
            float(
                (
                    predicted_rank == 0
                ).float().mean().item()
            ),
        "selection_results":
            all_results,
        "threshold_results":
            threshold_results,
        "ranking_at_30":
            [
                {
                    "rank":
                        index + 1,
                    "policy":
                        item["policy"],
                    "precision":
                        item["precision"],
                    "selected":
                        item["selected"],
                }
                for index, item in enumerate(
                    best_by_30
                )
            ],
        "ranking_at_50":
            [
                {
                    "rank":
                        index + 1,
                    "policy":
                        item["policy"],
                    "precision":
                        item["precision"],
                    "selected":
                        item["selected"],
                }
                for index, item in enumerate(
                    best_by_50
                )
            ],
    }

    output_path = (
        OUTPUT_DIR
        / "geometry_veto_selection_seed42.json"
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
        "SELECTION DIAGNOSTIC COMPLETE"
    )
    print("=" * 90)

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()