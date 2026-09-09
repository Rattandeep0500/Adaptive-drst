import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10

SUBPROTOTYPES = 5

ANCHOR_TOP_FRACTION = 0.10
MAX_TARGET_ANCHORS_PER_CLASS = 2000

TOP_FRACTIONS = [
    0.01,
    0.05,
    0.10,
    0.20,
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
    "checkpoints/visda_component_semantic_transport"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "component_semantic_transport_seed42.json"
)

OUTPUT_TENSOR = (
    OUTPUT_DIR
    / "component_semantic_transport_seed42.pt"
)

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

TRANSPORT_CLASSES = [
    class_id
    for class_id in range(NUM_CLASSES)
    if class_id != TRUCK_ID
]

SINK_CLASSES = [
    CAR_ID,
    BUS_ID,
    TRAIN_ID,
]


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

        if len(features) != len(labels):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
            )

        feature_parts.append(features)
        label_parts.append(labels)

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
                "teacher."
            ):
                if new_key.startswith(prefix):
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        result[new_key] = value

    return result


def extract_student_state(payload):
    for key in (
        "rpc_student_state_dict",
        "student_state_dict",
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

        z = model.encode(x)

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


def spherical_kmeans(
    features,
    k,
    seed
):
    features = F.normalize(
        features.float(),
        dim=1
    )

    n = len(features)

    if n == 0:
        raise RuntimeError(
            "Cannot cluster empty tensor"
        )

    k = min(
        k,
        n
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    initial = torch.randperm(
        n,
        generator=generator
    )[:k]

    centers = features[
        initial
    ].clone()

    assignments = torch.zeros(
        n,
        dtype=torch.long
    )

    for _ in range(50):
        similarities = (
            features
            @ centers.T
        )

        new_assignments = similarities.argmax(
            dim=1
        )

        new_centers = []

        for cluster_id in range(k):
            mask = (
                new_assignments
                == cluster_id
            )

            if mask.any():
                center = features[
                    mask
                ].mean(
                    dim=0
                )

                center = F.normalize(
                    center.unsqueeze(0),
                    dim=1
                ).squeeze(0)

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

        new_centers = F.normalize(
            new_centers,
            dim=1
        )

        if torch.equal(
            assignments,
            new_assignments
        ):
            centers = new_centers
            assignments = new_assignments
            break

        assignments = new_assignments
        centers = new_centers

    return centers


def build_source_subprototypes(
    adapted_source,
    source_labels
):
    prototypes = torch.zeros(
        NUM_CLASSES,
        SUBPROTOTYPES,
        HIDDEN_DIM
    )

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            source_labels
            == class_id
        )

        x = adapted_source[
            mask
        ]

        if len(x) == 0:
            raise RuntimeError(
                f"No source samples for "
                f"{CLASSES[class_id]}"
            )

        centers = spherical_kmeans(
            x,
            SUBPROTOTYPES,
            SEED + class_id
        )

        if len(centers) < SUBPROTOTYPES:
            fill = centers[
                -1
            ].unsqueeze(0).repeat(
                SUBPROTOTYPES
                - len(centers),
                1
            )

            centers = torch.cat(
                [
                    centers,
                    fill
                ],
                dim=0
            )

        prototypes[
            class_id
        ] = F.normalize(
            centers,
            dim=1
        )

    return prototypes


def build_target_subprototypes(
    adapted_target,
    graph_scores
):
    prototypes = torch.zeros(
        NUM_CLASSES,
        SUBPROTOTYPES,
        HIDDEN_DIM
    )

    anchor_counts = {}

    for class_id in range(
        NUM_CLASSES
    ):
        scores = graph_scores[
            :,
            class_id
        ]

        count = int(
            len(scores)
            * ANCHOR_TOP_FRACTION
        )

        count = min(
            count,
            MAX_TARGET_ANCHORS_PER_CLASS
        )

        count = max(
            count,
            SUBPROTOTYPES
        )

        _, indices = torch.topk(
            scores,
            k=count,
            largest=True,
            sorted=False
        )

        x = adapted_target[
            indices
        ]

        centers = spherical_kmeans(
            x,
            SUBPROTOTYPES,
            SEED + 1000 + class_id
        )

        if len(centers) < SUBPROTOTYPES:
            fill = centers[
                -1
            ].unsqueeze(0).repeat(
                SUBPROTOTYPES
                - len(centers),
                1
            )

            centers = torch.cat(
                [
                    centers,
                    fill
                ],
                dim=0
            )

        prototypes[
            class_id
        ] = F.normalize(
            centers,
            dim=1
        )

        anchor_counts[
            CLASSES[class_id]
        ] = count

    return (
        prototypes,
        anchor_counts
    )


def load_component_assignments():
    if not COMPONENT_ASSIGNMENTS.exists():
        raise FileNotFoundError(
            f"Missing component assignment file:\n"
            f"{COMPONENT_ASSIGNMENTS}"
        )

    payload = np.load(
        COMPONENT_ASSIGNMENTS,
        allow_pickle=True
    )

    if "candidate_indices" not in payload:
        raise RuntimeError(
            "candidate_indices missing"
        )

    if "component_labels" not in payload:
        raise RuntimeError(
            "component_labels missing"
        )

    candidate_indices = np.asarray(
        payload[
            "candidate_indices"
        ],
        dtype=np.int64
    )

    component_labels = np.asarray(
        payload[
            "component_labels"
        ],
        dtype=np.int64
    )

    payload.close()

    if len(candidate_indices) != len(
        component_labels
    ):
        raise RuntimeError(
            "Candidate/component length mismatch"
        )

    if len(
        np.unique(candidate_indices)
    ) != len(candidate_indices):
        raise RuntimeError(
            "Candidate indices are not unique"
        )

    if len(candidate_indices) == 0:
        raise RuntimeError(
            "Candidate region is empty"
        )

    return (
        candidate_indices,
        component_labels
    )


def build_component_centroids(
    adapted_target,
    candidate_indices,
    component_labels
):
    if adapted_target.ndim != 2:
        raise RuntimeError(
            "adapted_target must be 2-D"
        )

    if adapted_target.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            "adapted_target dimension is not 512"
        )

    candidate_features = (
        adapted_target[
            torch.from_numpy(
                candidate_indices
            ).long()
        ]
    )

    if candidate_features.ndim != 2:
        raise RuntimeError(
            "Candidate features are not 2-D"
        )

    if candidate_features.shape[0] != len(
        component_labels
    ):
        raise RuntimeError(
            "Candidate features and component labels disagree"
        )

    unique_components = np.unique(
        component_labels
    )

    centroids = []
    metadata = []

    for component_id in unique_components:
        mask = (
            component_labels
            == component_id
        )

        positions = np.flatnonzero(
            mask
        )

        if len(positions) == 0:
            continue

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
        ).squeeze(0)

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

    if not centroids:
        raise RuntimeError(
            "No components were constructed"
        )

    return (
        torch.stack(
            centroids,
            dim=0
        ),
        metadata
    )


def match_class_subprototypes(
    source_class,
    target_class
):
    similarity = (
        source_class
        @ target_class.T
    )

    source_used = set()
    target_used = set()

    pairs = []

    for _ in range(
        SUBPROTOTYPES
    ):
        best = None

        for source_id in range(
            SUBPROTOTYPES
        ):
            if source_id in source_used:
                continue

            for target_id in range(
                SUBPROTOTYPES
            ):
                if target_id in target_used:
                    continue

                value = float(
                    similarity[
                        source_id,
                        target_id
                    ].item()
                )

                if (
                    best is None
                    or value > best[0]
                ):
                    best = (
                        value,
                        source_id,
                        target_id
                    )

        if best is None:
            break

        pairs.append(best)

        source_used.add(
            best[1]
        )

        target_used.add(
            best[2]
        )

    return pairs


def fit_transport(
    source_prototypes,
    target_prototypes,
    excluded_classes
):
    excluded_classes = set(
        excluded_classes
    )

    source_points = []
    target_points = []
    pair_rows = []

    for class_id in TRANSPORT_CLASSES:
        if class_id in excluded_classes:
            continue

        source_class = source_prototypes[
            class_id
        ]

        target_class = target_prototypes[
            class_id
        ]

        pairs = match_class_subprototypes(
            source_class,
            target_class
        )

        for (
            similarity,
            source_id,
            target_id
        ) in pairs:
            source_points.append(
                source_class[
                    source_id
                ]
            )

            target_points.append(
                target_class[
                    target_id
                ]
            )

            pair_rows.append(
                {
                    "class_id":
                        int(class_id),
                    "source_subprototype":
                        int(source_id),
                    "target_subprototype":
                        int(target_id),
                    "similarity":
                        float(similarity)
                }
            )

    expected = (
        len(TRANSPORT_CLASSES)
        - len(
            excluded_classes
            & set(TRANSPORT_CLASSES)
        )
    ) * SUBPROTOTYPES

    if len(source_points) != expected:
        raise RuntimeError(
            f"Expected {expected} transport pairs, "
            f"got {len(source_points)}"
        )

    source_points = torch.stack(
        source_points,
        dim=0
    )

    target_points = torch.stack(
        target_points,
        dim=0
    )

    source_mean = source_points.mean(
        dim=0
    )

    target_mean = target_points.mean(
        dim=0
    )

    source_centered = (
        source_points
        - source_mean
    )

    target_centered = (
        target_points
        - target_mean
    )

    source_scale = torch.linalg.vector_norm(
        source_centered,
        dim=1
    ).mean().clamp_min(
        1e-12
    )

    target_scale = torch.linalg.vector_norm(
        target_centered,
        dim=1
    ).mean().clamp_min(
        1e-12
    )

    source_normalized = (
        source_centered
        / source_scale
    )

    target_normalized = (
        target_centered
        / target_scale
    )

    cross = (
        source_normalized.T
        @ target_normalized
    )

    u, _, vt = torch.linalg.svd(
        cross
    )

    rotation = (
        u
        @ vt
    )

    if (
        torch.linalg.det(
            rotation
        ).item()
        < 0
    ):
        u[:, -1] *= -1.0

        rotation = (
            u
            @ vt
        )

    scale = (
        target_scale
        / source_scale
    )

    translation = (
        target_mean
        - scale
        * (
            source_mean
            @ rotation
        )
    )

    transformed = (
        scale
        * (
            source_points
            @ rotation
        )
        + translation
    )

    transformed = F.normalize(
        transformed,
        dim=1
    )

    target_normalized_actual = F.normalize(
        target_points,
        dim=1
    )

    residual = float(
        (
            1.0
            - (
                transformed
                * target_normalized_actual
            ).sum(
                dim=1
            )
        )
        .mean()
        .item()
    )

    return (
        rotation,
        scale,
        translation,
        pair_rows,
        residual
    )


def apply_transport(
    prototypes,
    rotation,
    scale,
    translation
):
    flat = prototypes.reshape(
        NUM_CLASSES
        * SUBPROTOTYPES,
        HIDDEN_DIM
    )

    transformed = (
        scale
        * (
            flat
            @ rotation
        )
        + translation
    )

    transformed = F.normalize(
        transformed,
        dim=1
    )

    return transformed.reshape(
        NUM_CLASSES,
        SUBPROTOTYPES,
        HIDDEN_DIM
    )


def component_transport_scores(
    component_centroids,
    transported_prototypes
):
    component_centroids = F.normalize(
        component_centroids,
        dim=1
    )

    class_scores = torch.zeros(
        len(component_centroids),
        NUM_CLASSES
    )

    for class_id in range(
        NUM_CLASSES
    ):
        similarities = (
            component_centroids
            @ transported_prototypes[
                class_id
            ].T
        )

        class_scores[
            :,
            class_id
        ] = similarities.max(
            dim=1
        ).values

    truck_scores = class_scores[
        :,
        TRUCK_ID
    ]

    sink_scores = class_scores[
        :,
        SINK_CLASSES
    ].max(
        dim=1
    ).values

    truck_margin = (
        truck_scores
        - sink_scores
    )

    return (
        class_scores,
        truck_margin
    )


def binary_average_precision(
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

    ranks = np.arange(
        1,
        len(labels) + 1
    )

    precision = (
        cumulative
        / ranks
    )

    return float(
        precision[
            sorted_positive
        ].sum()
        / total_positive
    )


def component_sample_ap(
    metadata,
    scores,
    target_labels
):
    all_scores = []
    all_labels = []

    for component_row, item in enumerate(
        metadata
    ):
        indices = torch.from_numpy(
            item["global_indices"]
        ).long()

        score = float(
            scores[
                component_row
            ].item()
        )

        count = len(indices)

        all_scores.extend(
            [score] * count
        )

        labels = target_labels[
            indices
        ]

        all_labels.extend(
            (
                labels
                == TRUCK_ID
            ).long().tolist()
        )

    return binary_average_precision(
        np.asarray(
            all_scores,
            dtype=np.float64
        ),
        np.asarray(
            all_labels,
            dtype=np.int64
        )
    )


def component_binary_ap(
    metadata,
    scores,
    target_labels
):
    labels = []

    for item in metadata:
        indices = torch.from_numpy(
            item["global_indices"]
        ).long()

        component_labels = target_labels[
            indices
        ]

        labels.append(
            int(
                (
                    component_labels
                    == TRUCK_ID
                ).any().item()
            )
        )

    return binary_average_precision(
        scores.numpy(),
        np.asarray(
            labels,
            dtype=np.int64
        )
    )


def ranking_metrics(
    metadata,
    scores,
    target_labels,
    candidate_true_truck
):
    order = torch.argsort(
        scores,
        descending=True
    )

    total_truck = int(
        (
            target_labels
            == TRUCK_ID
        ).sum().item()
    )

    results = {}

    for fraction in TOP_FRACTIONS:
        component_count = max(
            1,
            int(
                len(metadata)
                * fraction
            )
        )

        selected_rows = order[
            :component_count
        ].tolist()

        selected_indices_parts = [
            metadata[
                row
            ]["global_indices"]
            for row in selected_rows
        ]

        selected_indices = np.concatenate(
            selected_indices_parts
        )

        selected_labels = target_labels[
            torch.from_numpy(
                selected_indices
            ).long()
        ]

        selected_count = len(
            selected_labels
        )

        truck_count = int(
            (
                selected_labels
                == TRUCK_ID
            ).sum().item()
        )

        car_count = int(
            (
                selected_labels
                == CAR_ID
            ).sum().item()
        )

        bus_count = int(
            (
                selected_labels
                == BUS_ID
            ).sum().item()
        )

        train_count = int(
            (
                selected_labels
                == TRAIN_ID
            ).sum().item()
        )

        other_count = (
            selected_count
            - truck_count
            - car_count
            - bus_count
            - train_count
        )

        precision = (
            truck_count
            / max(
                selected_count,
                1
            )
        )

        recall = (
            truck_count
            / max(
                total_truck,
                1
            )
        )

        candidate_recall = (
            truck_count
            / max(
                candidate_true_truck,
                1
            )
        )

        results[
            str(fraction)
        ] = {
            "component_count":
                component_count,
            "selected_samples":
                int(
                    selected_count
                ),
            "truck_precision":
                float(
                    precision
                ),
            "truck_recall":
                float(
                    recall
                ),
            "candidate_recall":
                float(
                    candidate_recall
                ),
            "car_contamination":
                float(
                    car_count
                    / max(
                        selected_count,
                        1
                    )
                ),
            "bus_contamination":
                float(
                    bus_count
                    / max(
                        selected_count,
                        1
                    )
                ),
            "train_contamination":
                float(
                    train_count
                    / max(
                        selected_count,
                        1
                    )
                ),
            "other_contamination":
                float(
                    other_count
                    / max(
                        selected_count,
                        1
                    )
                )
        }

    return results


def conversion_predictions(
    graph_predictions,
    metadata,
    scores,
    candidate_true_fraction=0.20
):
    component_count = max(
        1,
        int(
            len(metadata)
            * candidate_true_fraction
        )
    )

    order = torch.argsort(
        scores,
        descending=True
    )

    selected_rows = set(
        order[
            :component_count
        ].tolist()
    )

    predictions = graph_predictions.clone()

    selected_global_indices = []

    for row, item in enumerate(
        metadata
    ):
        if row in selected_rows:
            indices = torch.from_numpy(
                item["global_indices"]
            ).long()

            predictions[
                indices
            ] = TRUCK_ID

            selected_global_indices.extend(
                item["global_indices"].tolist()
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
                "Global index outside target range"
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
                "Duplicate global indices in component conversion"
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

    overall = float(
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

        value = float(
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
        ] = value

        values.append(
            value
        )

    return {
        "overall":
            overall,
        "mean_class":
            float(
                np.mean(
                    values
                )
            ),
        "per_class":
            per_class
    }


def leave_one_class_out_validation(
    source_prototypes,
    target_prototypes,
    adapted_target,
    target_labels,
    graph_predictions
):
    results = {}

    for held_out_class in TRANSPORT_CLASSES:
        (
            rotation,
            scale,
            translation,
            pair_rows,
            residual
        ) = fit_transport(
            source_prototypes,
            target_prototypes,
            excluded_classes=[
                TRUCK_ID,
                held_out_class
            ]
        )

        if len(pair_rows) != 50:
            raise RuntimeError(
                f"LOO {CLASSES[held_out_class]} "
                f"expected 50 pairs, got {len(pair_rows)}"
            )

        transformed = apply_transport(
            source_prototypes,
            rotation,
            scale,
            translation
        )

        class_prototypes = transformed[
            held_out_class
        ]

        graph_region = (
            graph_predictions
            == held_out_class
        )

        indices = torch.nonzero(
            graph_region,
            as_tuple=False
        ).flatten()

        if len(indices) > 0:
            x = adapted_target[
                indices
            ]

            similarities = (
                x
                @ class_prototypes.T
            )

            score = similarities.max(
                dim=1
            ).values

            labels = (
                target_labels[
                    indices
                ]
                == held_out_class
            ).long()

            retrieval_ap = binary_average_precision(
                score.numpy(),
                labels.numpy()
            )
        else:
            retrieval_ap = 0.0

        source_class = source_prototypes[
            held_out_class
        ]

        target_class = target_prototypes[
            held_out_class
        ]

        raw_similarity = (
            source_class
            @ target_class.T
        )

        raw_best = raw_similarity.max(
            dim=1
        ).values.mean().item()

        transformed_similarity = (
            transformed[
                held_out_class
            ]
            @ target_class.T
        )

        transformed_best = (
            transformed_similarity.max(
                dim=1
            ).values.mean().item()
        )

        results[
            CLASSES[held_out_class]
        ] = {
            "matched_pairs":
                len(pair_rows),
            "transport_residual":
                residual,
            "raw_source_target_similarity":
                float(raw_best),
            "transported_source_target_similarity":
                float(transformed_best),
            "similarity_delta":
                float(
                    transformed_best
                    - raw_best
                ),
            "graph_region_samples":
                int(
                    len(indices)
                ),
            "transport_retrieval_ap":
                float(
                    retrieval_ap
                )
        }

        print(
            f"holdout={CLASSES[held_out_class]:12s} | "
            f"pairs={len(pair_rows):2d} | "
            f"raw={raw_best:.6f} | "
            f"transported={transformed_best:.6f} | "
            f"delta={transformed_best - raw_best:+.6f} | "
            f"AP={retrieval_ap:.6f}"
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

    print("=" * 95)
    print(
        "VISDA-2017 CORRECTED COMPONENT-LEVEL SEMANTIC TRANSPORT PROBE"
    )
    print("=" * 95)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"subprototypes={SUBPROTOTYPES}"
    )

    print(
        f"anchor_top_fraction={ANCHOR_TOP_FRACTION}"
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
        "Loading graph outputs..."
    )

    graph_payload = safe_load(
        GRAPH_OUTPUTS
    )

    graph_scores = (
        graph_payload[
            "diffused_probabilities"
        ]
        .float()
        .cpu()
    )

    graph_predictions = (
        graph_scores
        .argmax(
            dim=1
        )
    )

    graph_metrics = multiclass_metrics(
        graph_predictions,
        target_labels
    )

    print(
        f"Graph overall="
        f"{graph_metrics['overall']:.2f}%"
    )

    print(
        f"Graph MCA="
        f"{graph_metrics['mean_class']:.2f}%"
    )

    print()

    print(
        "Loading RPC model..."
    )

    model = load_rpc_model().to(
        device
    )

    print()

    print(
        "Collecting adapted source features..."
    )

    adapted_source = collect_adapted_features(
        model,
        source_features,
        device
    )

    print(
        f"Adapted source: "
        f"{tuple(adapted_source.shape)}"
    )

    print()

    print(
        "Collecting adapted target features..."
    )

    adapted_target = collect_adapted_features(
        model,
        target_features,
        device
    )

    print(
        f"Adapted target: "
        f"{tuple(adapted_target.shape)}"
    )

    print()

    print(
        "Loading frozen component assignments..."
    )

    candidate_indices, component_labels = (
        load_component_assignments()
    )

    expected_candidate_mask = (
        torch.topk(
            graph_scores,
            k=3,
            dim=1
        ).indices
        == TRUCK_ID
    ).any(
        dim=1
    )

    expected_candidate_indices = torch.nonzero(
        expected_candidate_mask,
        as_tuple=False
    ).flatten().numpy()

    if not np.array_equal(
        np.sort(candidate_indices),
        np.sort(expected_candidate_indices)
    ):
        raise RuntimeError(
            "Frozen component candidate indices do not match "
            "the graph Top-3 candidate region"
        )

    candidate_label_tensor = target_labels[
        torch.from_numpy(
            candidate_indices
        ).long()
    ]

    truck_total = int(
        (
            target_labels
            == TRUCK_ID
        ).sum().item()
    )

    candidate_true_truck = int(
        (
            candidate_label_tensor
            == TRUCK_ID
        ).sum().item()
    )

    print(
        f"Candidate samples: "
        f"{len(candidate_indices)}"
    )

    print(
        f"Components: "
        f"{len(np.unique(component_labels))}"
    )

    print()

    print(
        "FROZEN TRUCK CANDIDATE REGION"
    )

    print(
        f"candidate precision="
        f"{100.0 * candidate_true_truck / max(len(candidate_indices), 1):.2f}%"
    )

    print(
        f"candidate recall="
        f"{100.0 * candidate_true_truck / max(truck_total, 1):.2f}%"
    )

    print()

    print(
        "Building source multi-prototypes..."
    )

    source_prototypes = build_source_subprototypes(
        adapted_source,
        source_labels
    )

    print(
        f"Source prototypes: "
        f"{tuple(source_prototypes.shape)}"
    )

    print()

    print(
        "Building target semantic anchors without target labels..."
    )

    target_prototypes, anchor_counts = (
        build_target_subprototypes(
            adapted_target,
            graph_scores
        )
    )

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"anchors={anchor_counts[CLASSES[class_id]]}"
        )

    print()

    print(
        "Computing frozen component centroids..."
    )

    component_centroid_values, metadata = (
        build_component_centroids(
            adapted_target,
            candidate_indices,
            component_labels
        )
    )

    assert component_centroid_values.ndim == 2
    assert component_centroid_values.shape[1] == HIDDEN_DIM
    assert len(metadata) == len(
        np.unique(
            component_labels
        )
    )

    print(
        f"Frozen component centroids: "
        f"{tuple(component_centroid_values.shape)}"
    )

    print()

    print(
        "=============================================================="
    )

    print(
        "PRIMARY TRUCK-EXCLUDED TRANSPORT"
    )

    (
        rotation,
        scale,
        translation,
        primary_pairs,
        primary_residual
    ) = fit_transport(
        source_prototypes,
        target_prototypes,
        excluded_classes=[
            TRUCK_ID
        ]
    )

    if len(primary_pairs) != 55:
        raise RuntimeError(
            f"Primary transport expected 55 pairs, "
            f"got {len(primary_pairs)}"
        )

    print(
        f"Matched prototype pairs: "
        f"{len(primary_pairs)}"
    )

    print(
        f"Mean transport residual: "
        f"{primary_residual:.8f}"
    )

    transported_prototypes = apply_transport(
        source_prototypes,
        rotation,
        scale,
        translation
    )

    print()

    print(
        "Computing transported component scores..."
    )

    component_scores, transported_margin = (
        component_transport_scores(
            component_centroid_values,
            transported_prototypes
        )
    )

    sample_ap = component_sample_ap(
        metadata,
        transported_margin,
        target_labels
    )

    component_ap = component_binary_ap(
        metadata,
        transported_margin,
        target_labels
    )

    print()

    print(
        "VALID TRANSPORT AP"
    )

    print(
        f"Sample AP: "
        f"{sample_ap:.6f}"
    )

    print(
        f"Component AP: "
        f"{component_ap:.6f}"
    )

    print()

    print(
        "TRANSPORT COMPONENT RANKING"
    )

    ranking = ranking_metrics(
        metadata,
        transported_margin,
        target_labels,
        candidate_true_truck
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

    print(
        "PRIMARY END-TO-END LABEL-FREE CONVERSION"
    )

    converted_predictions, selected_global_indices = (
        conversion_predictions(
            graph_predictions,
            metadata,
            transported_margin,
            candidate_true_fraction=0.20
        )
    )

    conversion_metrics = multiclass_metrics(
        converted_predictions,
        target_labels
    )

    selected_labels = target_labels[
        torch.from_numpy(
            selected_global_indices
        ).long()
    ]

    selected_truck = int(
        (
            selected_labels
            == TRUCK_ID
        ).sum().item()
    )

    converted_truck_expected_recall = (
        selected_truck
        / max(
            truck_total,
            1
        )
    )

    print(
        f"Selected component samples: "
        f"{len(selected_global_indices)}"
    )

    print(
        f"Selected truck recall: "
        f"{100.0 * converted_truck_expected_recall:.2f}%"
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

    print()

    print(
        "LEAVE-ONE-CLASS-OUT TRANSPORT VALIDATION"
    )

    loo_results = leave_one_class_out_validation(
        source_prototypes,
        target_prototypes,
        adapted_target,
        target_labels,
        graph_predictions
    )

    loo_ap = [
        row[
            "transport_retrieval_ap"
        ]
        for row in loo_results.values()
    ]

    loo_delta = [
        row[
            "similarity_delta"
        ]
        for row in loo_results.values()
    ]

    print()

    print(
        f"LOO mean retrieval AP="
        f"{np.mean(loo_ap):.6f}"
    )

    print(
        f"LOO mean similarity delta="
        f"{np.mean(loo_delta):+.6f}"
    )

    transported_rows = []

    order = torch.argsort(
        transported_margin,
        descending=True
    )

    for rank, component_row in enumerate(
        order[
            :20
        ].tolist(),
        start=1
    ):
        item = metadata[
            component_row
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
                len(labels),
                1
            )
        )

        transported_rows.append(
            {
                "rank":
                    rank,
                "component_id":
                    item["component_id"],
                "size":
                    item["size"],
                "truck_count":
                    truck_count,
                "truck_purity":
                    purity,
                "transport_truck_score":
                    float(
                        component_scores[
                            component_row,
                            TRUCK_ID
                        ].item()
                    ),
                "transport_margin":
                    float(
                        transported_margin[
                            component_row
                        ].item()
                    )
            }
        )

        print(
            f"rank={rank:2d} | "
            f"component={item['component_id']:4d} | "
            f"size={item['size']:4d} | "
            f"purity={100.0 * purity:6.2f}% | "
            f"truck={truck_count:4d} | "
            f"margin={transported_margin[component_row].item():+.6f}"
        )

    torch.save(
        {
            "source_prototypes":
                source_prototypes,
            "target_prototypes":
                target_prototypes,
            "transported_prototypes":
                transported_prototypes,
            "transport_rotation":
                rotation,
            "transport_scale":
                scale,
            "transport_translation":
                translation,
            "candidate_indices":
                torch.from_numpy(
                    candidate_indices
                ),
            "component_labels":
                torch.from_numpy(
                    component_labels
                ),
            "component_centroids":
                component_centroid_values,
            "component_scores":
                component_scores,
            "transported_truck_margin":
                transported_margin,
            "selected_global_indices":
                torch.from_numpy(
                    selected_global_indices
                ),
            "converted_predictions":
                converted_predictions,
            "target_labels":
                target_labels
        },
        OUTPUT_TENSOR
    )

    result = {
        "experiment":
            "visda_corrected_component_semantic_transport",
        "seed":
            SEED,
        "device":
            str(device),
        "source_samples":
            len(source_features),
        "target_samples":
            len(target_features),
        "candidate_samples":
            len(candidate_indices),
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
        "component_count":
            len(metadata),
        "subprototypes":
            SUBPROTOTYPES,
        "graph_baseline":
            graph_metrics,
        "primary_transport": {
            "excluded_classes":
                [
                    "truck"
                ],
            "included_class_count":
                11,
            "matched_pairs":
                len(primary_pairs),
            "expected_pairs":
                55,
            "residual":
                primary_residual,
            "sample_ap":
                sample_ap,
            "component_ap":
                component_ap,
            "ranking":
                ranking
        },
        "conversion": {
            "component_fraction":
                0.20,
            "selected_samples":
                len(
                    selected_global_indices
                ),
            "selected_truck":
                selected_truck,
            "metrics":
                conversion_metrics
        },
        "loo_validation": {
            "mean_retrieval_ap":
                float(
                    np.mean(
                        loo_ap
                    )
                ),
            "median_retrieval_ap":
                float(
                    np.median(
                        loo_ap
                    )
                ),
            "mean_similarity_delta":
                float(
                    np.mean(
                        loo_delta
                    )
                ),
            "per_class":
                loo_results
        },
        "top_transported_components":
            transported_rows
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
        "CORRECTED COMPONENT SEMANTIC TRANSPORT COMPLETE"
    )
    print("=" * 95)

    print(
        f"Primary matched pairs: "
        f"{len(primary_pairs)}"
    )

    print(
        f"Sample AP: "
        f"{sample_ap:.6f}"
    )

    print(
        f"Component AP: "
        f"{component_ap:.6f}"
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
        f"LOO mean retrieval AP: "
        f"{np.mean(loo_ap):.6f}"
    )

    print(
        f"Saved: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()