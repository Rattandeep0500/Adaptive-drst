import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

TRUCK_ID = 11
TOP_K = 3

TARGET_ANCHORS_PER_CLASS = 2000
FIT_ANCHORS_PER_CLASS = 1000
CALIBRATION_ANCHORS_PER_CLASS = 1000

COMPONENT_FRACTIONS = [
    0.01,
    0.05,
    0.10,
    0.20,
]

CANDIDATE_MASS_FRACTIONS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
]

SOURCE_CACHE = Path(
    "checkpoints/visda_feature_cache/source"
)

TARGET_CACHE = Path(
    "checkpoints/visda_feature_cache/target"
)

RPC_CHECKPOINT = Path(
    "checkpoints/visda_rpc_causal_experiment/rpc_seed42.pt"
)

GRAPH_OUTPUTS = Path(
    "checkpoints/visda_graph_semantic_diffusion/"
    "graph_semantic_outputs_seed42.pt"
)

COMPONENT_ASSIGNMENTS = Path(
    "checkpoints/visda_truck_component_structure_probe/"
    "truck_component_assignments_seed42.npz"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_residual_semantic_completion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "residual_semantic_completion_seed42.json"
)

OUTPUT_TENSOR = (
    OUTPUT_DIR
    / "residual_semantic_completion_seed42.pt"
)

OUTPUT_ASSIGNMENTS = (
    OUTPUT_DIR
    / "residual_component_assignments_seed42.npz"
)

PLOT_SCORE_PURITY = (
    OUTPUT_DIR
    / "residual_score_vs_purity_seed42.png"
)

PLOT_RECALL = (
    OUTPUT_DIR
    / "residual_precision_recall_seed42.png"
)

PLOT_COMPONENTS = (
    OUTPUT_DIR
    / "residual_component_sizes_seed42.png"
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

NON_TRUCK_CLASSES = [
    class_id
    for class_id in range(NUM_CLASSES)
    if class_id != TRUCK_ID
]

BATCH_SIZE = 4096


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
                f"Expected {INPUT_DIM}-D input, got {x.shape[1]}"
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
        z = self.adapter(x)

        return (
            self.classifier1(z),
            self.classifier2(z)
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


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
        payload = safe_load(path)

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
            )

        features = payload[
            "features"
        ].float()

        labels = payload[
            "labels"
        ].long()

        if features.ndim != 2:
            raise RuntimeError(
                f"Invalid feature tensor in {path}"
            )

        if features.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"got {features.shape[1]}"
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

    return features, labels


def normalize_state_dict(state_dict):
    output = {}

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
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        output[
            new_key
        ] = value

    return output


def extract_student_state(payload):
    for key in (
        "student_state_dict",
        "rpc_student_state_dict",
        "baseline_student_state_dict",
        "state_dict"
    ):
        if key in payload:
            return payload[key]

    if all(
        isinstance(key, str)
        and (
            key.startswith("adapter.")
            or key.startswith("classifier1.")
            or key.startswith("classifier2.")
        )
        for key in payload.keys()
    ):
        return payload

    raise RuntimeError(
        "No compatible student state dict found"
    )


def load_rpc_model():
    payload = safe_load(
        RPC_CHECKPOINT
    )

    state_dict = normalize_state_dict(
        extract_student_state(
            payload
        )
    )

    model = MCDModel()

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
            f"Missing checkpoint keys: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {unexpected}"
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


@torch.no_grad()
def collect_adapted_features(
    model,
    features,
    device
):
    parts = []

    model.eval()

    for start in range(
        0,
        len(features),
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            len(features)
        )

        x = features[
            start:end
        ].to(
            device
        )

        z = model.encode(
            x
        )

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected adapted dimension {HIDDEN_DIM}, "
                f"got {z.shape[1]}"
            )

        z = F.normalize(
            z,
            dim=1
        )

        parts.append(
            z.cpu()
        )

    return torch.cat(
        parts,
        dim=0
    )


def build_candidate_region(
    graph_scores
):
    topk = torch.topk(
        graph_scores,
        k=TOP_K,
        dim=1,
        largest=True,
        sorted=True
    ).indices

    truck_candidate = (
        topk
        == TRUCK_ID
    ).any(
        dim=1
    )

    return (
        topk,
        truck_candidate
    )


def deterministic_top_indices(
    scores,
    count,
    seed
):
    count = min(
        count,
        len(scores)
    )

    order = torch.argsort(
        scores,
        descending=True,
        stable=True
    )

    if count <= 0:
        return torch.empty(
            0,
            dtype=torch.long
        )

    return order[
        :count
    ]


def build_nontruck_support(
    adapted_target,
    graph_scores
):
    fit_anchors = {}
    calibration_anchors = {}
    counts = {}

    for class_id in NON_TRUCK_CLASSES:
        scores = graph_scores[
            :,
            class_id
        ]

        total = min(
            TARGET_ANCHORS_PER_CLASS,
            len(scores)
        )

        if total < (
            FIT_ANCHORS_PER_CLASS
            + CALIBRATION_ANCHORS_PER_CLASS
        ):
            raise RuntimeError(
                f"Insufficient target samples for "
                f"{CLASSES[class_id]}"
            )

        ranked = deterministic_top_indices(
            scores,
            total,
            SEED + class_id
        )

        fit_indices = ranked[
            :FIT_ANCHORS_PER_CLASS
        ]

        calibration_indices = ranked[
            FIT_ANCHORS_PER_CLASS:
            FIT_ANCHORS_PER_CLASS
            + CALIBRATION_ANCHORS_PER_CLASS
        ]

        fit_anchors[
            class_id
        ] = adapted_target[
            fit_indices
        ]

        calibration_anchors[
            class_id
        ] = adapted_target[
            calibration_indices
        ]

        counts[
            CLASSES[class_id]
        ] = {
            "fit":
                int(
                    len(fit_indices)
                ),
            "calibration":
                int(
                    len(calibration_indices)
                )
        }

    return (
        fit_anchors,
        calibration_anchors,
        counts
    )


def build_target_subprototypes(
    fit_anchors
):
    prototypes = {}

    for class_id in NON_TRUCK_CLASSES:
        x = fit_anchors[
            class_id
        ]

        if len(x) == 0:
            raise RuntimeError(
                f"No fit anchors for "
                f"{CLASSES[class_id]}"
            )

        generator = torch.Generator()
        generator.manual_seed(
            SEED
            + 1000
            + class_id
        )

        k = min(
            5,
            len(x)
        )

        initial = torch.randperm(
            len(x),
            generator=generator
        )[
            :k
        ]

        centers = F.normalize(
            x[
                initial
            ].clone(),
            dim=1
        )

        for _ in range(30):
            similarities = (
                x
                @ centers.T
            )

            assignments = similarities.argmax(
                dim=1
            )

            new_centers = []

            for cluster_id in range(k):
                mask = (
                    assignments
                    == cluster_id
                )

                if mask.any():
                    center = x[
                        mask
                    ].mean(
                        dim=0
                    )

                    center = F.normalize(
                        center.unsqueeze(0),
                        dim=1
                    ).squeeze(
                        0
                    )
                else:
                    center = centers[
                        cluster_id
                    ]

                new_centers.append(
                    center
                )

            new_centers = torch.stack(
                new_centers,
                dim=0
            )

            if torch.equal(
                assignments,
                torch.zeros_like(
                    assignments
                )
            ):
                pass

            if torch.allclose(
                centers,
                new_centers,
                atol=1e-6,
                rtol=1e-6
            ):
                centers = new_centers
                break

            centers = new_centers

        prototypes[
            class_id
        ] = F.normalize(
            centers,
            dim=1
        )

    return prototypes


def empirical_support_pvalue(
    distances,
    calibration_distances
):
    calibration_distances = calibration_distances.flatten()

    count = (
        calibration_distances
        >= distances.unsqueeze(1)
    ).sum(
        dim=1
    )

    return (
        1.0
        + count.float()
    ) / (
        1.0
        + len(calibration_distances)
    )


@torch.no_grad()
def compute_class_support(
    sample_features,
    prototype,
    calibration_features
):
    sample_features = F.normalize(
        sample_features,
        dim=1
    )

    prototype = F.normalize(
        prototype,
        dim=1
    )

    calibration_features = F.normalize(
        calibration_features,
        dim=1
    )

    sample_similarity = (
        sample_features
        @ prototype.T
    ).max(
        dim=1
    ).values

    calibration_similarity = (
        calibration_features
        @ prototype.T
    ).max(
        dim=1
    ).values

    sample_distance = (
        1.0
        - sample_similarity
    )

    calibration_distance = (
        1.0
        - calibration_similarity
    )

    pvalue = empirical_support_pvalue(
        sample_distance,
        calibration_distance
    )

    return (
        pvalue,
        sample_distance,
        calibration_distance
    )


def build_residual_scores(
    adapted_target,
    graph_scores,
    fit_anchors,
    calibration_anchors
):
    target_support = {}

    nontruck_best_support = torch.zeros(
        len(adapted_target)
    )

    class_supports = {}

    for class_id in NON_TRUCK_CLASSES:
        fit_features = fit_anchors[
            class_id
        ]

        calibration_features = calibration_anchors[
            class_id
        ]

        prototype = fit_features.mean(
            dim=0,
            keepdim=True
        )

        prototype = F.normalize(
            prototype,
            dim=1
        )

        pvalue, _, _ = compute_class_support(
            adapted_target,
            prototype,
            calibration_features
        )

        class_supports[
            class_id
        ] = pvalue

        nontruck_best_support = torch.maximum(
            nontruck_best_support,
            pvalue
        )

    residual = (
        1.0
        - nontruck_best_support
    )

    residual = residual.clamp(
        0.0,
        1.0
    )

    return (
        residual,
        nontruck_best_support,
        class_supports
    )


def build_component_centroids(
    adapted_target,
    candidate_indices,
    component_labels
):
    candidate_features = adapted_target[
        torch.from_numpy(
            candidate_indices
        ).long()
    ]

    unique_components = np.unique(
        component_labels
    )

    centroids = []
    metadata = []

    for component_id in unique_components:
        positions = np.flatnonzero(
            component_labels
            == component_id
        )

        x = candidate_features[
            torch.from_numpy(
                positions
            ).long()
        ]

        centroid = x.mean(
            dim=0
        )

        centroid = F.normalize(
            centroid.unsqueeze(0),
            dim=1
        ).squeeze(
            0
        )

        centroids.append(
            centroid
        )

        metadata.append(
            {
                "component_id":
                    int(component_id),
                "positions":
                    positions.astype(
                        np.int64
                    ),
                "global_indices":
                    candidate_indices[
                        positions
                    ].astype(
                        np.int64
                    ),
                "size":
                    int(len(positions))
            }
        )

    return (
        torch.stack(
            centroids,
            dim=0
        ),
        metadata
    )


def component_residual_scores(
    component_metadata,
    sample_residual
):
    scores = []

    for item in component_metadata:
        positions = torch.from_numpy(
            item["global_indices"]
        ).long()

        values = sample_residual[
            positions
        ]

        score = torch.median(
            values
        )

        scores.append(
            score
        )

    return torch.stack(
        scores
    )


def sample_ap(
    scores,
    labels
):
    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    positive = (
        labels == 1
    )

    total_positive = int(
        positive.sum()
    )

    if total_positive == 0:
        return 0.0

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    sorted_positive = positive[
        order
    ]

    cumulative = np.cumsum(
        sorted_positive
    )

    rank = np.arange(
        1,
        len(scores) + 1
    )

    precision = (
        cumulative
        / rank
    )

    return float(
        precision[
            sorted_positive
        ].sum()
        / total_positive
    )


def component_ap(
    component_scores,
    component_metadata,
    target_labels
):
    labels = []

    for item in component_metadata:
        indices = torch.from_numpy(
            item["global_indices"]
        ).long()

        labels.append(
            int(
                (
                    target_labels[
                        indices
                    ]
                    == TRUCK_ID
                ).any().item()
            )
        )

    return sample_ap(
        component_scores.numpy(),
        np.asarray(
            labels,
            dtype=np.int64
        )
    )


def ranking_metrics(
    component_scores,
    component_metadata,
    target_labels,
    truck_total,
    candidate_truck_total
):
    order = torch.argsort(
        component_scores,
        descending=True
    )

    results = {}

    for fraction in COMPONENT_FRACTIONS:
        count = max(
            1,
            int(
                len(component_metadata)
                * fraction
            )
        )

        selected_rows = order[
            :count
        ].tolist()

        selected_indices = np.concatenate(
            [
                component_metadata[
                    row
                ]["global_indices"]
                for row in selected_rows
            ]
        )

        labels = target_labels[
            torch.from_numpy(
                selected_indices
            ).long()
        ]

        selected_count = len(
            labels
        )

        truck_count = int(
            (
                labels
                == TRUCK_ID
            ).sum().item()
        )

        car_count = int(
            (
                labels
                == 3
            ).sum().item()
        )

        bus_count = int(
            (
                labels
                == 2
            ).sum().item()
        )

        train_count = int(
            (
                labels
                == 10
            ).sum().item()
        )

        other_count = (
            selected_count
            - truck_count
            - car_count
            - bus_count
            - train_count
        )

        results[
            str(fraction)
        ] = {
            "component_count":
                count,
            "selected_samples":
                int(selected_count),
            "truck_precision":
                truck_count
                / max(
                    selected_count,
                    1
                ),
            "truck_recall":
                truck_count
                / max(
                    truck_total,
                    1
                ),
            "candidate_recall":
                truck_count
                / max(
                    candidate_truck_total,
                    1
                ),
            "car_contamination":
                car_count
                / max(
                    selected_count,
                    1
                ),
            "bus_contamination":
                bus_count
                / max(
                    selected_count,
                    1
                ),
            "train_contamination":
                train_count
                / max(
                    selected_count,
                    1
                ),
            "other_contamination":
                other_count
                / max(
                    selected_count,
                    1
                )
        }

    return results


def candidate_mass_ranking(
    component_scores,
    component_metadata,
    target_labels,
    truck_total,
    candidate_truck_total
):
    order = torch.argsort(
        component_scores,
        descending=True
    )

    sizes = np.array(
        [
            item["size"]
            for item in component_metadata
        ],
        dtype=np.int64
    )

    results = {}

    for mass_fraction in CANDIDATE_MASS_FRACTIONS:
        target_mass = int(
            np.ceil(
                mass_fraction
                * sizes.sum()
            )
        )

        selected_rows = []
        accumulated = 0

        for row in order.tolist():
            selected_rows.append(
                row
            )

            accumulated += sizes[
                row
            ]

            if accumulated >= target_mass:
                break

        selected_indices = np.concatenate(
            [
                component_metadata[
                    row
                ]["global_indices"]
                for row in selected_rows
            ]
        )

        labels = target_labels[
            torch.from_numpy(
                selected_indices
            ).long()
        ]

        selected_count = len(
            labels
        )

        truck_count = int(
            (
                labels
                == TRUCK_ID
            ).sum().item()
        )

        car_count = int(
            (
                labels
                == 3
            ).sum().item()
        )

        bus_count = int(
            (
                labels
                == 2
            ).sum().item()
        )

        train_count = int(
            (
                labels
                == 10
            ).sum().item()
        )

        other_count = (
            selected_count
            - truck_count
            - car_count
            - bus_count
            - train_count
        )

        results[
            str(mass_fraction)
        ] = {
            "selected_component_count":
                len(selected_rows),
            "selected_samples":
                int(selected_count),
            "truck_precision":
                truck_count
                / max(
                    selected_count,
                    1
                ),
            "truck_recall":
                truck_count
                / max(
                    truck_total,
                    1
                ),
            "candidate_recall":
                truck_count
                / max(
                    candidate_truck_total,
                    1
                ),
            "car_contamination":
                car_count
                / max(
                    selected_count,
                    1
                ),
            "bus_contamination":
                bus_count
                / max(
                    selected_count,
                    1
                ),
            "train_contamination":
                train_count
                / max(
                    selected_count,
                    1
                ),
            "other_contamination":
                other_count
                / max(
                    selected_count,
                    1
                )
        }

    return results


def convert_predictions(
    graph_predictions,
    component_metadata,
    component_scores,
    fraction
):
    count = max(
        1,
        int(
            len(component_metadata)
            * fraction
        )
    )

    order = torch.argsort(
        component_scores,
        descending=True
    )

    selected_rows = set(
        order[
            :count
        ].tolist()
    )

    predictions = graph_predictions.clone()

    selected_global_indices = []

    for row, item in enumerate(
        component_metadata
    ):
        if row not in selected_rows:
            continue

        indices = torch.from_numpy(
            item[
                "global_indices"
            ]
        ).long()

        predictions[
            indices
        ] = TRUCK_ID

        selected_global_indices.extend(
            item[
                "global_indices"
            ].tolist()
        )

    selected_global_indices = np.asarray(
        selected_global_indices,
        dtype=np.int64
    )

    if len(selected_global_indices) > 0:
        if (
            selected_global_indices.min()
            < 0
        ):
            raise RuntimeError(
                "Negative global index"
            )

        if (
            selected_global_indices.max()
            >= len(predictions)
        ):
            raise RuntimeError(
                "Global index out of range"
            )

        if (
            len(
                np.unique(
                    selected_global_indices
                )
            )
            != len(selected_global_indices)
        ):
            raise RuntimeError(
                "Duplicate global indices"
            )

    return (
        predictions,
        selected_global_indices
    )


def multiclass_metrics(
    predictions,
    labels
):
    predictions = predictions.long()
    labels = labels.long()

    correct = (
        predictions
        == labels
    )

    overall = (
        100.0
        * correct.float().mean().item()
    )

    per_class = {}

    values = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            labels
            == class_id
        )

        value = (
            100.0
            * correct[
                mask
            ]
            .float()
            .mean()
            .item()
            if mask.any()
            else 0.0
        )

        per_class[
            CLASSES[class_id]
        ] = float(
            value
        )

        values.append(
            value
        )

    return {
        "overall":
            float(
                overall
            ),
        "mean_class":
            float(
                np.mean(values)
            ),
        "per_class":
            per_class
    }


def make_plots(
    component_sizes,
    component_scores,
    component_purities,
    ranking
):
    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        component_sizes,
        bins=30
    )

    plt.xlabel(
        "Component size"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Residual Semantic Component Sizes"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_COMPONENTS,
        dpi=160
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.scatter(
        component_scores,
        component_purities,
        s=18
    )

    plt.xlabel(
        "Label-free residual score"
    )

    plt.ylabel(
        "Truck purity"
    )

    plt.title(
        "Residual Score vs Truck Purity"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_SCORE_PURITY,
        dpi=160
    )

    plt.close()

    fractions = []
    precisions = []
    recalls = []

    for key in (
        "0.01",
        "0.05",
        "0.1",
        "0.2"
    ):
        fractions.append(
            float(key)
        )

        precisions.append(
            100.0
            * ranking[key][
                "truck_precision"
            ]
        )

        recalls.append(
            100.0
            * ranking[key][
                "truck_recall"
            ]
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        fractions,
        precisions,
        marker="o",
        label="Precision"
    )

    plt.plot(
        fractions,
        recalls,
        marker="o",
        label="Recall"
    )

    plt.xlabel(
        "Fraction of components selected"
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Residual Component Ranking"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_RECALL,
        dpi=160
    )

    plt.close()


def main():
    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 95)
    print(
        "VISDA-2017 RESIDUAL SEMANTIC COMPLETION"
    )
    print("=" * 95)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        "Target labels remain evaluation-only."
    )

    print()

    print(
        "Loading graph outputs..."
    )

    graph_payload = safe_load(
        GRAPH_OUTPUTS
    )

    if "diffused_probabilities" not in graph_payload:
        raise RuntimeError(
            "Missing diffused_probabilities"
        )

    graph_scores = (
        graph_payload[
            "diffused_probabilities"
        ]
        .float()
        .cpu()
    )

    if graph_scores.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            "Graph class dimension mismatch"
        )

    graph_predictions = (
        graph_scores.argmax(
            dim=1
        )
    )

    (
        top3,
        truck_candidate
    ) = build_candidate_region(
        graph_scores
    )

    candidate_indices = torch.nonzero(
        truck_candidate,
        as_tuple=False
    ).flatten().numpy()

    print(
        f"Target samples: "
        f"{len(graph_scores)}"
    )

    print(
        f"Truck Top-3 candidate samples: "
        f"{len(candidate_indices)}"
    )

    print()

    print(
        "Loading RPC checkpoint..."
    )

    model = load_rpc_model().to(
        device
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

    print()

    print(
        "Collecting adapted target representation..."
    )

    adapted_target = collect_adapted_features(
        model,
        target_features,
        device
    )

    print(
        f"Adapted target shape: "
        f"{tuple(adapted_target.shape)}"
    )

    print()

    print(
        "Collecting adapted source representation..."
    )

    adapted_source = collect_adapted_features(
        model,
        source_features,
        device
    )

    print(
        f"Adapted source shape: "
        f"{tuple(adapted_source.shape)}"
    )

    print()

    print(
        "Building non-truck target support..."
    )

    (
        fit_anchors,
        calibration_anchors,
        anchor_counts
    ) = build_nontruck_support(
        adapted_target,
        graph_scores
    )

    for class_id in NON_TRUCK_CLASSES:
        values = anchor_counts[
            CLASSES[class_id]
        ]

        print(
            f"{CLASSES[class_id]:12s} | "
            f"fit={values['fit']} | "
            f"calibration={values['calibration']}"
        )

    print()

    print(
        "Constructing target sub-prototypes..."
    )

    target_prototypes = build_target_subprototypes(
        fit_anchors
    )

    print(
        f"Target support classes: "
        f"{len(target_prototypes)}"
    )

    print()

    print(
        "Computing frozen non-truck support distributions..."
    )

    (
        residual_scores_full,
        best_nontruck_support,
        class_supports
    ) = build_residual_scores(
        adapted_target,
        graph_scores,
        fit_anchors,
        calibration_anchors
    )

    print(
        f"Residual score mean: "
        f"{residual_scores_full.mean().item():.6f}"
    )

    print(
        f"Residual score median: "
        f"{residual_scores_full.median().item():.6f}"
    )

    print(
        f"Best non-truck support mean: "
        f"{best_nontruck_support.mean().item():.6f}"
    )

    print()

    print(
        "Loading frozen component assignments..."
    )

    component_payload = np.load(
        COMPONENT_ASSIGNMENTS,
        allow_pickle=True
    )

    if "candidate_indices" not in component_payload:
        raise RuntimeError(
            "candidate_indices missing"
        )

    if "component_labels" not in component_payload:
        raise RuntimeError(
            "component_labels missing"
        )

    stored_candidate_indices = np.asarray(
        component_payload[
            "candidate_indices"
        ],
        dtype=np.int64
    )

    component_labels = np.asarray(
        component_payload[
            "component_labels"
        ],
        dtype=np.int64
    )

    component_payload.close()

    if not np.array_equal(
        np.sort(
            stored_candidate_indices
        ),
        np.sort(
            candidate_indices
        )
    ):
        raise RuntimeError(
            "Frozen component candidate indices do not "
            "match current graph Top-3 candidate region"
        )

    print(
        f"Candidate samples verified: "
        f"{len(stored_candidate_indices)}"
    )

    print(
        f"Frozen components: "
        f"{len(np.unique(component_labels))}"
    )

    print()

    print(
        "Computing component centroids..."
    )

    (
        component_centroids_tensor,
        component_metadata
    ) = build_component_centroids(
        adapted_target,
        candidate_indices,
        component_labels
    )

    print(
        f"Component centroid shape: "
        f"{tuple(component_centroids_tensor.shape)}"
    )

    print()

    print(
        "Computing component residual scores..."
    )

    candidate_residual = (
        residual_scores_full[
            torch.from_numpy(
                candidate_indices
            ).long()
        ]
    )

    candidate_support = (
        best_nontruck_support[
            torch.from_numpy(
                candidate_indices
            ).long()
        ]
    )

    component_scores = component_residual_scores(
        component_metadata,
        residual_scores_full
    )

    component_sizes = np.array(
        [
            item["size"]
            for item in component_metadata
        ],
        dtype=np.int64
    )

    print(
        f"Component score mean: "
        f"{component_scores.mean().item():.6f}"
    )

    print(
        f"Component score median: "
        f"{component_scores.median().item():.6f}"
    )

    print()

    print(
        "=============================================================="
    )

    print(
        "LABEL-FREE SCORE FROZEN"
    )

    print(
        "No target labels have been used to construct or rank the score."
    )

    print(
        "Loading target labels for evaluation..."
    )

    truck_total = int(
        (
            target_labels
            == TRUCK_ID
        ).sum().item()
    )

    candidate_true_truck = int(
        (
            target_labels[
                torch.from_numpy(
                    candidate_indices
                ).long()
            ]
            == TRUCK_ID
        ).sum().item()
    )

    print(
        f"True target trucks: "
        f"{truck_total}"
    )

    print(
        f"Truck candidates: "
        f"{candidate_true_truck}"
    )

    print(
        f"Candidate recall: "
        f"{100.0 * candidate_true_truck / max(truck_total, 1):.2f}%"
    )

    print()

    candidate_sample_ap = sample_ap(
        candidate_residual.numpy(),
        (
            target_labels[
                torch.from_numpy(
                    candidate_indices
                ).long()
            ]
            == TRUCK_ID
        ).numpy().astype(
            np.int64
        )
    )

    full_sample_ap = sample_ap(
        residual_scores_full.numpy(),
        (
            target_labels
            == TRUCK_ID
        ).numpy().astype(
            np.int64
        )
    )

    candidate_component_ap = component_ap(
        component_scores,
        component_metadata,
        target_labels
    )

    print(
        "RESIDUAL AP"
    )

    print(
        f"Candidate sample AP: "
        f"{candidate_sample_ap:.6f}"
    )

    print(
        f"Full-target sample AP: "
        f"{full_sample_ap:.6f}"
    )

    print(
        f"Component AP: "
        f"{candidate_component_ap:.6f}"
    )

    print()

    ranking = ranking_metrics(
        component_scores,
        component_metadata,
        target_labels,
        truck_total,
        candidate_true_truck
    )

    print(
        "COMPONENT FRACTION RANKING"
    )

    for fraction, row in ranking.items():
        print(
            f"top={100.0 * float(fraction):5.1f}% | "
            f"precision={100.0 * row['truck_precision']:6.2f}% | "
            f"recall={100.0 * row['truck_recall']:6.2f}% | "
            f"candidate-recall={100.0 * row['candidate_recall']:6.2f}% | "
            f"samples={row['selected_samples']}"
        )

    print()

    mass_ranking = candidate_mass_ranking(
        component_scores,
        component_metadata,
        target_labels,
        truck_total,
        candidate_true_truck
    )

    print(
        "CANDIDATE SAMPLE-MASS RANKING"
    )

    for fraction, row in mass_ranking.items():
        print(
            f"mass={100.0 * float(fraction):5.1f}% | "
            f"precision={100.0 * row['truck_precision']:6.2f}% | "
            f"recall={100.0 * row['truck_recall']:6.2f}% | "
            f"candidate-recall={100.0 * row['candidate_recall']:6.2f}% | "
            f"samples={row['selected_samples']} | "
            f"components={row['selected_component_count']}"
        )

    print()

    print(
        "TOP COMPONENTS BY RESIDUAL SCORE"
    )

    order = torch.argsort(
        component_scores,
        descending=True
    )

    component_records = []

    for rank, row in enumerate(
        order[
            :20
        ].tolist(),
        start=1
    ):
        item = component_metadata[
            row
        ]

        labels = target_labels[
            torch.from_numpy(
                item["global_indices"]
            ).long()
        ]

        truck_count = int(
            (
                labels
                == TRUCK_ID
            ).sum().item()
        )

        purity = (
            truck_count
            / max(
                item["size"],
                1
            )
        )

        score = float(
            component_scores[
                row
            ].item()
        )

        support = float(
            (
                1.0
                - component_scores[
                    row
                ].item()
            )
        )

        print(
            f"rank={rank:2d} | "
            f"component={item['component_id']:4d} | "
            f"size={item['size']:4d} | "
            f"purity={100.0 * purity:6.2f}% | "
            f"truck={truck_count:4d} | "
            f"residual={score:.6f} | "
            f"nontruck_support={support:.6f}"
        )

        component_records.append(
            {
                "rank":
                    rank,
                "component_id":
                    item[
                        "component_id"
                    ],
                "size":
                    item[
                        "size"
                    ],
                "truck_count":
                    truck_count,
                "truck_purity":
                    purity,
                "residual_score":
                    score,
                "best_nontruck_support":
                    support
            }
        )

    print()

    print(
        "PRIMARY LABEL-FREE CONVERSION"
    )

    (
        converted_predictions,
        selected_global_indices
    ) = convert_predictions(
        graph_predictions,
        component_metadata,
        component_scores,
        0.20
    )

    conversion_metrics = multiclass_metrics(
        converted_predictions,
        target_labels
    )

    print(
        f"Selected samples: "
        f"{len(selected_global_indices)}"
    )

    print(
        f"Converted overall: "
        f"{conversion_metrics['overall']:.2f}%"
    )

    print(
        f"Converted MCA: "
        f"{conversion_metrics['mean_class']:.2f}%"
    )

    print(
        f"Converted truck: "
        f"{conversion_metrics['per_class']['truck']:.2f}%"
    )

    print(
        f"Converted car: "
        f"{conversion_metrics['per_class']['car']:.2f}%"
    )

    print(
        f"Converted bus: "
        f"{conversion_metrics['per_class']['bus']:.2f}%"
    )

    print(
        f"Converted train: "
        f"{conversion_metrics['per_class']['train']:.2f}%"
    )

    component_purities = []

    for item in component_metadata:
        labels = target_labels[
            torch.from_numpy(
                item["global_indices"]
            ).long()
        ]

        purity = (
            (
                labels
                == TRUCK_ID
            )
            .float()
            .mean()
            .item()
        )

        component_purities.append(
            purity
        )

    component_purities = np.asarray(
        component_purities,
        dtype=np.float64
    )

    make_plots(
        component_sizes,
        component_scores.numpy(),
        component_purities,
        ranking
    )

    np.savez(
        OUTPUT_ASSIGNMENTS,
        candidate_indices=candidate_indices,
        component_labels=component_labels,
        residual_candidate_scores=candidate_residual.numpy(),
        component_residual_scores=component_scores.numpy(),
        candidate_best_nontruck_support=candidate_support.numpy()
    )

    torch.save(
        {
            "residual_scores_full":
                residual_scores_full,
            "best_nontruck_support_full":
                best_nontruck_support,
            "component_scores":
                component_scores,
            "component_centroids":
                component_centroids_tensor,
            "candidate_indices":
                torch.from_numpy(
                    candidate_indices
                ),
            "component_labels":
                torch.from_numpy(
                    component_labels
                ),
            "converted_predictions":
                converted_predictions,
            "selected_global_indices":
                torch.from_numpy(
                    selected_global_indices
                ),
            "target_labels":
                target_labels
        },
        OUTPUT_TENSOR
    )

    result = {
        "experiment":
            "visda_residual_semantic_completion",
        "seed":
            SEED,
        "device":
            str(device),
        "target_samples":
            int(
                len(target_features)
            ),
        "source_samples":
            int(
                len(source_features)
            ),
        "candidate_samples":
            int(
                len(candidate_indices)
            ),
        "component_count":
            int(
                len(component_metadata)
            ),
        "truck_total":
            truck_total,
        "candidate_true_truck":
            candidate_true_truck,
        "candidate_precision":
            candidate_true_truck
            / max(
                len(candidate_indices),
                1
            ),
        "candidate_recall":
            candidate_true_truck
            / max(
                truck_total,
                1
            ),
        "support_construction": {
            "target_anchor_count_per_class":
                TARGET_ANCHORS_PER_CLASS,
            "fit_anchors_per_class":
                FIT_ANCHORS_PER_CLASS,
            "calibration_anchors_per_class":
                CALIBRATION_ANCHORS_PER_CLASS,
            "truck_used_in_support":
                False
        },
        "sample_ap": {
            "candidate":
                candidate_sample_ap,
            "full_target":
                full_sample_ap
        },
        "component_ap":
            candidate_component_ap,
        "component_fraction_ranking":
            ranking,
        "candidate_mass_ranking":
            mass_ranking,
        "conversion": {
            "component_fraction":
                0.20,
            "selected_samples":
                int(
                    len(selected_global_indices)
                ),
            "metrics":
                conversion_metrics
        },
        "top_components":
            component_records
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2
        )

    print()
    print("=" * 95)
    print(
        "RESIDUAL SEMANTIC COMPLETION COMPLETE"
    )
    print("=" * 95)

    print(
        f"Candidate sample AP: "
        f"{candidate_sample_ap:.6f}"
    )

    print(
        f"Full-target sample AP: "
        f"{full_sample_ap:.6f}"
    )

    print(
        f"Component AP: "
        f"{candidate_component_ap:.6f}"
    )

    print(
        f"20% converted MCA: "
        f"{conversion_metrics['mean_class']:.2f}%"
    )

    print(
        f"20% converted OA: "
        f"{conversion_metrics['overall']:.2f}%"
    )

    print(
        f"20% converted truck: "
        f"{conversion_metrics['per_class']['truck']:.2f}%"
    )

    print(
        f"JSON: "
        f"{OUTPUT_JSON}"
    )

    print(
        f"NPZ: "
        f"{OUTPUT_ASSIGNMENTS}"
    )

    print(
        f"Torch: "
        f"{OUTPUT_TENSOR}"
    )


if __name__ == "__main__":
    main()