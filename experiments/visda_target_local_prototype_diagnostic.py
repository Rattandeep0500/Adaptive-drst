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
TARGET_CACHE = CACHE_ROOT / "target"

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/geometry_gated_mcd_seed42.pt"
)

KNN_GRAPH = Path(
    "checkpoints/visda_target_knn_geometry/target_knn_indices_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_target_local_prototype"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 4096

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10
SKATEBOARD_ID = 9
PERSON_ID = 7
KNIFE_ID = 5

K_VALUES = [5, 10, 20, 50]

ANCHOR_CONFIDENCE = 0.90
ANCHOR_TOP_FRACTION = 0.10
MAX_ANCHORS_PER_CLASS = 2000

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

FOCUS_CLASSES = {
    "truck": TRUCK_ID,
    "car": CAR_ID,
    "bus": BUS_ID,
    "train": TRAIN_ID,
    "skateboard": SKATEBOARD_ID,
    "person": PERSON_ID,
    "knife": KNIFE_ID,
}


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

        features = payload["features"].float()
        labels = payload["labels"].long()

        if features.ndim != 2:
            raise RuntimeError(
                f"Invalid feature shape in {path}: "
                f"{tuple(features.shape)}"
            )

        if features.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features in {path}, "
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

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected final dimension {INPUT_DIM}, "
            f"got {features.shape[1]}"
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
                f"Expected 2D input, got {tuple(x.shape)}"
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
        z = self.encode(x)

        return (
            self.classifier1(z),
            self.classifier2(z)
        )


def normalize_state_dict(state_dict):
    normalized = {}

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

        normalized[new_key] = value

    return normalized


def load_checkpoint():
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

    if "source_prototypes" not in payload:
        raise RuntimeError(
            "source_prototypes missing"
        )

    model = MCDModel()

    model.load_state_dict(
        normalize_state_dict(
            payload["student_state_dict"]
        ),
        strict=True
    )

    model.eval()

    source_prototypes = payload[
        "source_prototypes"
    ].float()

    if source_prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Expected source prototype shape "
            f"({NUM_CLASSES}, {HIDDEN_DIM}), "
            f"got {tuple(source_prototypes.shape)}"
        )

    source_prototypes = F.normalize(
        source_prototypes,
        dim=1
    )

    return model, source_prototypes


@torch.no_grad()
def collect_target_state(
    model,
    target_features
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            target_features
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    z_parts = []
    probability_parts = []
    confidence_parts = []
    prediction_parts = []
    disagreement_parts = []

    for (x,) in loader:
        z = model.encode(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected {HIDDEN_DIM}-D adapted features, "
                f"got {z.shape[1]}"
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

        disagreement = (
            p1 - p2
        ).abs().mean(
            dim=1
        )

        z = F.normalize(
            z,
            dim=1
        )

        z_parts.append(
            z.cpu()
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
                z_parts,
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
            ),
    }


def load_knn_graph(
    n_samples
):
    if not KNN_GRAPH.exists():
        raise FileNotFoundError(
            f"KNN graph not found: {KNN_GRAPH}"
        )

    payload = torch.load(
        KNN_GRAPH,
        map_location="cpu"
    )

    if "indices" not in payload:
        raise RuntimeError(
            "KNN graph missing indices"
        )

    indices = payload[
        "indices"
    ].long()

    if indices.ndim != 2:
        raise RuntimeError(
            f"Invalid KNN shape {tuple(indices.shape)}"
        )

    if indices.shape[0] != n_samples:
        raise RuntimeError(
            f"KNN rows={indices.shape[0]}, "
            f"expected {n_samples}"
        )

    if indices.shape[1] < max(K_VALUES):
        raise RuntimeError(
            f"KNN graph has only {indices.shape[1]} neighbors, "
            f"need {max(K_VALUES)}"
        )

    if (indices < 0).any():
        raise RuntimeError(
            "Negative KNN indices detected"
        )

    if (indices >= n_samples).any():
        raise RuntimeError(
            "Out-of-range KNN indices detected"
        )

    return indices


def build_anchor_mask(
    confidence,
    predictions
):
    anchors = torch.zeros(
        len(predictions),
        dtype=torch.bool
    )

    anchor_counts = torch.zeros(
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

        indices = torch.nonzero(
            class_mask,
            as_tuple=False
        ).flatten()

        if len(indices) == 0:
            continue

        desired = int(
            round(
                len(predictions)
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
            len(indices)
        )

        if len(indices) > desired:
            order = torch.argsort(
                confidence[
                    indices
                ],
                descending=True
            )

            indices = indices[
                order[:desired]
            ]

        anchors[
            indices
        ] = True

        anchor_counts[
            class_id
        ] = len(indices)

    return (
        anchors,
        anchor_counts
    )


def reconstruct_local_prototypes(
    z,
    confidence,
    predictions,
    knn_indices,
    k
):
    anchors, anchor_counts = (
        build_anchor_mask(
            confidence,
            predictions
        )
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
        class_anchors = torch.nonzero(
            anchors
            & (
                predictions
                == class_id
            ),
            as_tuple=False
        ).flatten()

        if len(class_anchors) == 0:
            continue

        neighbors = knn_indices[
            class_anchors,
            :k
        ]

        local_indices = torch.unique(
            neighbors.reshape(-1)
        )

        if len(local_indices) == 0:
            continue

        anchor_features = z[
            class_anchors
        ]

        local_features = z[
            local_indices
        ]

        anchor_confidence = confidence[
            class_anchors
        ]

        weights = (
            anchor_confidence
            / anchor_confidence.sum().clamp_min(
                1e-8
            )
        )

        anchor_center = (
            anchor_features
            * weights.unsqueeze(1)
        ).sum(
            dim=0
        )

        local_center = local_features.mean(
            dim=0
        )

        prototype = (
            0.5 * anchor_center
            + 0.5 * local_center
        )

        prototype = F.normalize(
            prototype,
            dim=0
        )

        prototypes[
            class_id
        ] = prototype

        support_counts[
            class_id
        ] = len(local_indices)

    return (
        prototypes,
        anchor_counts,
        support_counts
    )


def compute_geometry(
    z,
    labels,
    prototypes
):
    valid = (
        torch.norm(
            prototypes,
            dim=1
        )
        > 0
    )

    safe_prototypes = prototypes.clone()

    for class_id in range(
        NUM_CLASSES
    ):
        if valid[class_id]:
            safe_prototypes[
                class_id
            ] = F.normalize(
                safe_prototypes[
                    class_id
                ],
                dim=0
            )

    similarities = (
        z
        @ safe_prototypes.t()
    )

    similarities[
        :,
        ~valid
    ] = -float("inf")

    result = {}

    for class_name, class_id in (
        FOCUS_CLASSES.items()
    ):
        mask = (
            labels
            == class_id
        )

        if not mask.any():
            continue

        values = similarities[
            mask
        ]

        true_similarity = (
            values[
                :,
                class_id
            ]
        )

        competing = values.clone()

        competing[
            :,
            class_id
        ] = -float("inf")

        best_competitor = (
            competing.max(
                dim=1
            ).values
        )

        margin = (
            true_similarity
            - best_competitor
        )

        nearest_class = (
            values.argmax(
                dim=1
            )
        )

        result[
            class_name
        ] = {
            "count":
                int(
                    mask.sum().item()
                ),
            "true_similarity":
                float(
                    true_similarity.mean().item()
                ),
            "margin":
                float(
                    margin.mean().item()
                ),
            "nearest_accuracy":
                float(
                    (
                        nearest_class
                        == class_id
                    )
                    .float()
                    .mean()
                    .item()
                ),
            "positive_margin_fraction":
                float(
                    (
                        margin > 0
                    )
                    .float()
                    .mean()
                    .item()
                ),
            "margin_ge_0_10_fraction":
                float(
                    (
                        margin >= 0.10
                    )
                    .float()
                    .mean()
                    .item()
                ),
        }

    return result


def prototype_pairwise_metrics(
    prototypes
):
    result = {}

    pairs = [
        ("truck", "car"),
        ("truck", "bus"),
        ("truck", "train"),
        ("car", "bus"),
        ("car", "train"),
        ("bus", "train"),
        ("skateboard", "person"),
        ("skateboard", "knife"),
    ]

    for first_name, second_name in pairs:
        first_id = FOCUS_CLASSES[
            first_name
        ]

        second_id = FOCUS_CLASSES[
            second_name
        ]

        first_norm = torch.norm(
            prototypes[
                first_id
            ]
        )

        second_norm = torch.norm(
            prototypes[
                second_id
            ]
        )

        key = (
            f"{first_name}_to_{second_name}"
        )

        if (
            first_norm.item() == 0
            or second_norm.item() == 0
        ):
            result[key] = None
            continue

        first = F.normalize(
            prototypes[
                first_id
            ],
            dim=0
        )

        second = F.normalize(
            prototypes[
                second_id
            ],
            dim=0
        )

        similarity = torch.dot(
            first,
            second
        ).item()

        result[key] = {
            "similarity":
                float(similarity),
            "distance":
                float(
                    1.0 - similarity
                )
        }

    return result


def print_geometry(
    title,
    geometry
):
    print()
    print(
        title
    )

    for class_name in (
        "truck",
        "car",
        "bus",
        "train",
        "skateboard",
        "person"
    ):
        if class_name not in geometry:
            continue

        row = geometry[
            class_name
        ]

        print(
            f"{class_name:12s} | "
            f"sim={row['true_similarity']:.6f} | "
            f"margin={row['margin']:.6f} | "
            f"nearest="
            f"{100.0 * row['nearest_accuracy']:.2f}% | "
            f"positive="
            f"{100.0 * row['positive_margin_fraction']:.2f}%"
        )


def main():
    set_seed(
        SEED
    )

    print("=" * 90)
    print(
        "VISDA-2017 TARGET-LOCAL PROTOTYPE RECONSTRUCTION DIAGNOSTIC"
    )
    print("=" * 90)

    print(
        f"seed={SEED}"
    )

    print(
        f"checkpoint={BASE_CHECKPOINT}"
    )

    print(
        f"knn_graph={KNN_GRAPH}"
    )

    print(
        f"K values={K_VALUES}"
    )

    print(
        f"anchor confidence={ANCHOR_CONFIDENCE}"
    )

    print(
        f"anchor top fraction={ANCHOR_TOP_FRACTION}"
    )

    print(
        f"max anchors/class={MAX_ANCHORS_PER_CLASS}"
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

    if target_features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Target input dimension assertion failed"
        )

    print()
    print(
        "Loading checkpoint..."
    )

    model, source_prototypes = load_checkpoint()

    print(
        "Checkpoint loaded successfully."
    )

    print(
        f"Source prototypes: "
        f"{tuple(source_prototypes.shape)}"
    )

    print()
    print(
        "Collecting target representation..."
    )

    state = collect_target_state(
        model,
        target_features
    )

    z = state[
        "z"
    ]

    probabilities = state[
        "probabilities"
    ]

    confidence = state[
        "confidence"
    ]

    predictions = state[
        "predictions"
    ]

    disagreement = state[
        "disagreement"
    ]

    if z.shape != (
        len(target_features),
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Unexpected adapted shape {tuple(z.shape)}"
        )

    print(
        f"Adapted target shape: "
        f"{tuple(z.shape)}"
    )

    target_accuracy = (
        (
            predictions
            == target_labels
        )
        .float()
        .mean()
        .item()
    )

    print(
        f"Target accuracy: "
        f"{100.0 * target_accuracy:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{confidence.mean().item():.6f}"
    )

    print(
        f"Mean disagreement: "
        f"{disagreement.mean().item():.6f}"
    )

    print()
    print(
        "Loading target KNN graph..."
    )

    knn_indices = load_knn_graph(
        len(target_features)
    )

    print(
        f"KNN graph shape: "
        f"{tuple(knn_indices.shape)}"
    )

    print()
    print("=" * 90)
    print(
        "SOURCE-PROTOTYPE BASELINE"
    )
    print("=" * 90)

    source_geometry = compute_geometry(
        z,
        target_labels,
        source_prototypes
    )

    print_geometry(
        "Source-prototype geometry",
        source_geometry
    )

    all_results = {}

    for k in K_VALUES:
        print()
        print("=" * 90)
        print(
            f"TARGET-LOCAL RECONSTRUCTION K={k}"
        )
        print("=" * 90)

        (
            local_prototypes,
            anchor_counts,
            support_counts
        ) = reconstruct_local_prototypes(
            z,
            confidence,
            predictions,
            knn_indices,
            k
        )

        print()
        print(
            "Unlabeled anchor counts:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"{class_id:02d} "
                f"{CLASSES[class_id]:12s}: "
                f"{int(anchor_counts[class_id].item())}"
            )

        print()
        print(
            "Target-local prototype support:"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            print(
                f"{class_id:02d} "
                f"{CLASSES[class_id]:12s}: "
                f"{int(support_counts[class_id].item())}"
            )

        valid = (
            torch.norm(
                local_prototypes,
                dim=1
            )
            > 0
        )

        safe_prototypes = local_prototypes.clone()

        for class_id in range(
            NUM_CLASSES
        ):
            if valid[class_id]:
                safe_prototypes[
                    class_id
                ] = F.normalize(
                    safe_prototypes[
                        class_id
                    ],
                    dim=0
                )
            else:
                safe_prototypes[
                    class_id
                ] = source_prototypes[
                    class_id
                ]

        local_geometry = compute_geometry(
            z,
            target_labels,
            safe_prototypes
        )

        print_geometry(
            "Target-local geometry",
            local_geometry
        )

        print()
        print(
            "Prototype pairwise geometry:"
        )

        pairwise = prototype_pairwise_metrics(
            safe_prototypes
        )

        for pair_name, pair_value in pairwise.items():
            if pair_value is None:
                print(
                    f"{pair_name:30s} | unavailable"
                )
            else:
                print(
                    f"{pair_name:30s} | "
                    f"sim="
                    f"{pair_value['similarity']:.6f} | "
                    f"dist="
                    f"{pair_value['distance']:.6f}"
                )

        truck_mask = (
            target_labels
            == TRUCK_ID
        )

        truck_z = z[
            truck_mask
        ]

        local_normalized = F.normalize(
            safe_prototypes,
            dim=1
        )

        truck_similarity_matrix = (
            truck_z
            @ local_normalized.t()
        )

        truck_score = (
            truck_similarity_matrix[
                :,
                TRUCK_ID
            ]
        )

        truck_competitor = torch.stack(
            [
                truck_similarity_matrix[
                    :,
                    CAR_ID
                ],
                truck_similarity_matrix[
                    :,
                    BUS_ID
                ],
                truck_similarity_matrix[
                    :,
                    TRAIN_ID
                ],
            ],
            dim=1
        ).max(
            dim=1
        ).values

        truck_margin = (
            truck_score
            - truck_competitor
        )

        truck_metrics = {
            "truck_similarity":
                float(
                    truck_score.mean().item()
                ),
            "car_similarity":
                float(
                    truck_similarity_matrix[
                        :,
                        CAR_ID
                    ].mean().item()
                ),
            "bus_similarity":
                float(
                    truck_similarity_matrix[
                        :,
                        BUS_ID
                    ].mean().item()
                ),
            "train_similarity":
                float(
                    truck_similarity_matrix[
                        :,
                        TRAIN_ID
                    ].mean().item()
                ),
            "mean_margin":
                float(
                    truck_margin.mean().item()
                ),
            "positive_margin_fraction":
                float(
                    (
                        truck_margin > 0
                    )
                    .float()
                    .mean()
                    .item()
                ),
            "margin_ge_0_05_fraction":
                float(
                    (
                        truck_margin >= 0.05
                    )
                    .float()
                    .mean()
                    .item()
                ),
            "margin_ge_0_10_fraction":
                float(
                    (
                        truck_margin >= 0.10
                    )
                    .float()
                    .mean()
                    .item()
                )
        }

        print()
        print(
            "Truck-specific target-local geometry:"
        )

        print(
            f"truck similarity: "
            f"{truck_metrics['truck_similarity']:.6f}"
        )

        print(
            f"car similarity: "
            f"{truck_metrics['car_similarity']:.6f}"
        )

        print(
            f"bus similarity: "
            f"{truck_metrics['bus_similarity']:.6f}"
        )

        print(
            f"train similarity: "
            f"{truck_metrics['train_similarity']:.6f}"
        )

        print(
            f"truck-vs-best-competitor margin: "
            f"{truck_metrics['mean_margin']:.6f}"
        )

        print(
            f"positive margin: "
            f"{100.0 * truck_metrics['positive_margin_fraction']:.2f}%"
        )

        print(
            f"margin >= 0.05: "
            f"{100.0 * truck_metrics['margin_ge_0_05_fraction']:.2f}%"
        )

        print(
            f"margin >= 0.10: "
            f"{100.0 * truck_metrics['margin_ge_0_10_fraction']:.2f}%"
        )

        all_results[
            str(k)
        ] = {
            "anchor_counts":
                anchor_counts.tolist(),
            "support_counts":
                support_counts.tolist(),
            "valid_local_prototypes":
                valid.tolist(),
            "geometry":
                local_geometry,
            "pairwise_geometry":
                pairwise,
            "truck_specific":
                truck_metrics
        }

    source_truck = source_geometry[
        "truck"
    ]

    source_margin = (
        source_truck[
            "margin"
        ]
    )

    best_k = None
    best_margin = -float(
        "inf"
    )

    for k in K_VALUES:
        margin = all_results[
            str(k)
        ][
            "truck_specific"
        ][
            "mean_margin"
        ]

        if margin > best_margin:
            best_margin = margin
            best_k = k

    print()
    print("=" * 90)
    print(
        "SOURCE VS TARGET-LOCAL TRUCK COMPARISON"
    )
    print("=" * 90)

    print(
        f"Source truck margin: "
        f"{source_margin:.6f}"
    )

    print(
        f"Source nearest-prototype accuracy: "
        f"{100.0 * source_truck['nearest_accuracy']:.2f}%"
    )

    print(
        f"Source positive-margin fraction: "
        f"{100.0 * source_truck['positive_margin_fraction']:.2f}%"
    )

    for k in K_VALUES:
        row = all_results[
            str(k)
        ][
            "truck_specific"
        ]

        geometry_row = all_results[
            str(k)
        ][
            "geometry"
        ][
            "truck"
        ]

        print()
        print(
            f"K={k}"
        )

        print(
            f"  truck margin: "
            f"{row['mean_margin']:.6f}"
        )

        print(
            f"  improvement: "
            f"{row['mean_margin'] - source_margin:.6f}"
        )

        print(
            f"  nearest accuracy: "
            f"{100.0 * geometry_row['nearest_accuracy']:.2f}%"
        )

        print(
            f"  positive margin: "
            f"{100.0 * row['positive_margin_fraction']:.2f}%"
        )

    print()
    print(
        f"Best target-local K: "
        f"{best_k}"
    )

    print(
        f"Best target-local truck margin: "
        f"{best_margin:.6f}"
    )

    print(
        f"Best improvement over source: "
        f"{best_margin - source_margin:.6f}"
    )

    print()
    print(
        "Target labels were used only for offline evaluation."
    )

    output = {
        "experiment":
            "visda_target_local_prototype_reconstruction_diagnostic",
        "seed":
            SEED,
        "checkpoint":
            str(BASE_CHECKPOINT),
        "knn_graph":
            str(KNN_GRAPH),
        "target_samples":
            len(target_features),
        "input_dim":
            INPUT_DIM,
        "hidden_dim":
            HIDDEN_DIM,
        "target_accuracy":
            100.0 * target_accuracy,
        "mean_confidence":
            float(
                confidence.mean().item()
            ),
        "mean_disagreement":
            float(
                disagreement.mean().item()
            ),
        "anchor_confidence":
            ANCHOR_CONFIDENCE,
        "anchor_top_fraction":
            ANCHOR_TOP_FRACTION,
        "max_anchors_per_class":
            MAX_ANCHORS_PER_CLASS,
        "source_geometry":
            source_geometry,
        "results":
            all_results,
        "best_local_k":
            best_k,
        "best_local_truck_margin":
            best_margin,
        "source_truck_margin":
            source_margin,
        "target_local_improvement":
            best_margin
            - source_margin
    }

    output_path = (
        OUTPUT_DIR
        / "target_local_prototype_seed42.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2
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