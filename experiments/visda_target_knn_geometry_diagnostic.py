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

OUTPUT_DIR = Path(
    "checkpoints/visda_target_knn_geometry"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10
SKATEBOARD_ID = 9
PERSON_ID = 7

BATCH_SIZE = 4096
K_VALUES = [5, 10, 20, 50]
MAX_K = max(K_VALUES)

KNN_QUERY_CHUNK = 256

FOCUS_CLASSES = {
    "truck": TRUCK_ID,
    "car": CAR_ID,
    "bus": BUS_ID,
    "train": TRAIN_ID,
    "skateboard": SKATEBOARD_ID,
    "person": PERSON_ID,
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

        x = payload["features"].float()
        y = payload["labels"].long()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature shape in {path}: {tuple(x.shape)}"
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

        features.append(x)
        labels.append(y)

    x = torch.cat(
        features,
        dim=0
    )

    y = torch.cat(
        labels,
        dim=0
    )

    if x.ndim != 2:
        raise RuntimeError(
            f"Invalid final feature tensor: {tuple(x.shape)}"
        )

    if x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected final dimension {INPUT_DIM}, "
            f"got {x.shape[1]}"
        )

    if len(x) != len(y):
        raise RuntimeError(
            "Final feature/label count mismatch"
        )

    return x, y


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
                f"Adapter expects 2D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expects {INPUT_DIM}-D input, "
                f"got {x.shape[1]}-D"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter output expected {HIDDEN_DIM}-D, "
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


def load_checkpoint(model):
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
            "student_state_dict missing from checkpoint"
        )

    model.load_state_dict(
        normalize_state_dict(
            payload["student_state_dict"]
        ),
        strict=True
    )

    if "source_prototypes" not in payload:
        raise RuntimeError(
            "source_prototypes missing from checkpoint"
        )

    prototypes = payload[
        "source_prototypes"
    ].float()

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Expected prototype shape "
            f"({NUM_CLASSES}, {HIDDEN_DIM}), "
            f"got {tuple(prototypes.shape)}"
        )

    return prototypes


@torch.no_grad()
def collect_adapted_features(
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

    output = []

    for (x,) in loader:
        x = x.to(device)

        z = model.encode(
            x
        )

        if z.ndim != 2:
            raise RuntimeError(
                f"Unexpected adapted tensor shape "
                f"{tuple(z.shape)}"
            )

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected adapted dimension "
                f"{HIDDEN_DIM}, got {z.shape[1]}"
            )

        z = F.normalize(
            z,
            dim=1
        )

        output.append(
            z.cpu()
        )

    result = torch.cat(
        output,
        dim=0
    )

    if result.shape != (
        len(features),
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Unexpected final adapted shape "
            f"{tuple(result.shape)}"
        )

    return result


@torch.no_grad()
def collect_predictions(
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

    predictions = []
    confidence = []
    disagreement = []

    for (x,) in loader:
        x = x.to(device)

        logits1, logits2 = model(x)

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

        conf, pred = p.max(
            dim=1
        )

        disc = (
            p1 - p2
        ).abs().mean(
            dim=1
        )

        predictions.append(
            pred.cpu()
        )

        confidence.append(
            conf.cpu()
        )

        disagreement.append(
            disc.cpu()
        )

    return (
        torch.cat(
            predictions,
            dim=0
        ),
        torch.cat(
            confidence,
            dim=0
        ),
        torch.cat(
            disagreement,
            dim=0
        )
    )


@torch.no_grad()
def compute_knn_indices(
    features,
    max_k,
    query_chunk
):
    n = features.shape[0]

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2D feature matrix, got {features.shape}"
        )

    if max_k >= n:
        raise RuntimeError(
            f"max_k={max_k} must be smaller than n={n}"
        )

    features = F.normalize(
        features.float(),
        dim=1
    ).contiguous()

    knn_indices = torch.empty(
        n,
        max_k,
        dtype=torch.long
    )

    print()
    print(
        f"Computing exact cosine KNN: "
        f"N={n}, max_k={max_k}, "
        f"query_chunk={query_chunk}"
    )

    total_chunks = (
        n + query_chunk - 1
    ) // query_chunk

    for chunk_id, start in enumerate(
        range(
            0,
            n,
            query_chunk
        ),
        start=1
    ):
        end = min(
            start + query_chunk,
            n
        )

        query = features[
            start:end
        ]

        similarities = (
            query
            @ features.t()
        )

        local_rows = (
            torch.arange(
                end - start
            )
        )

        local_cols = (
            torch.arange(
                start,
                end
            )
        )

        similarities[
            local_rows,
            local_cols
        ] = -float("inf")

        values, indices = torch.topk(
            similarities,
            k=max_k,
            dim=1,
            largest=True,
            sorted=True
        )

        if not torch.isfinite(
            values
        ).all():
            raise RuntimeError(
                "Non-finite KNN similarities encountered"
            )

        knn_indices[
            start:end
        ] = indices.cpu()

        if (
            chunk_id == 1
            or chunk_id == total_chunks
            or chunk_id % 10 == 0
        ):
            print(
                f"  chunk "
                f"{chunk_id}/{total_chunks}"
            )

        del similarities
        del values
        del indices

    return knn_indices


def class_knn_statistics(
    labels,
    knn_indices,
    k
):
    results = {}

    neighbors = knn_indices[
        :,
        :k
    ]

    neighbor_labels = labels[
        neighbors
    ]

    for class_name, class_id in (
        FOCUS_CLASSES.items()
    ):
        mask = (
            labels
            == class_id
        )

        if not mask.any():
            continue

        local = neighbor_labels[
            mask
        ]

        purity = (
            (
                local
                == class_id
            ).float().mean(
                dim=1
            )
        )

        majority_labels = (
            torch.mode(
                local,
                dim=1
            ).values
        )

        majority_accuracy = (
            (
                majority_labels
                == class_id
            )
            .float()
            .mean()
        )

        true_neighbor_count = (
            (
                local
                == class_id
            )
            .sum(
                dim=1
            )
        )

        competitor_counts = []

        for competitor_id in range(
            NUM_CLASSES
        ):
            if competitor_id == class_id:
                continue

            count = (
                local
                == competitor_id
            ).sum(
                dim=1
            )

            competitor_counts.append(
                (
                    float(
                        count.float().mean().item()
                    ),
                    competitor_id
                )
            )

        competitor_counts.sort(
            reverse=True
        )

        strongest_mean_count, strongest_competitor = (
            competitor_counts[0]
        )

        flat_distribution = torch.bincount(
            local.reshape(-1),
            minlength=NUM_CLASSES
        ).float()

        flat_distribution /= (
            flat_distribution.sum()
            .clamp_min(1.0)
        )

        results[
            class_name
        ] = {
            "count":
                int(mask.sum().item()),
            "k":
                k,
            "mean_knn_purity":
                float(
                    purity.mean().item()
                ),
            "median_knn_purity":
                float(
                    purity.median().item()
                ),
            "fraction_purity_ge_50":
                float(
                    (
                        purity >= 0.50
                    ).float().mean().item()
                ),
            "fraction_purity_ge_70":
                float(
                    (
                        purity >= 0.70
                    ).float().mean().item()
                ),
            "fraction_purity_ge_90":
                float(
                    (
                        purity >= 0.90
                    ).float().mean().item()
                ),
            "majority_neighbor_accuracy":
                float(
                    majority_accuracy.item()
                ),
            "mean_true_neighbor_count":
                float(
                    true_neighbor_count.float().mean().item()
                ),
            "strongest_competitor":
                CLASSES[
                    strongest_competitor
                ],
            "strongest_competitor_mean_count":
                strongest_mean_count,
            "neighbor_distribution":
                flat_distribution.tolist()
        }

    return results


def flow_analysis(
    labels,
    predictions,
    knn_indices,
    k
):
    neighbors = knn_indices[
        :,
        :k
    ]

    neighbor_labels = labels[
        neighbors
    ]

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
    ]

    name_to_id = {
        name: idx
        for idx, name in enumerate(
            CLASSES
        )
    }

    result = {}

    for true_name, predicted_name in pairs:
        true_id = name_to_id[
            true_name
        ]

        predicted_id = name_to_id[
            predicted_name
        ]

        mask = (
            (labels == true_id)
            & (
                predictions
                == predicted_id
            )
        )

        count = int(
            mask.sum().item()
        )

        key = (
            f"{true_name}_to_{predicted_name}"
        )

        if count == 0:
            result[key] = {
                "count":
                    0,
                "true_neighbor_fraction":
                    None,
                "attractor_neighbor_fraction":
                    None
            }
            continue

        local = neighbor_labels[
            mask
        ]

        true_fraction = (
            (
                local
                == true_id
            )
            .float()
            .mean()
            .item()
        )

        attractor_fraction = (
            (
                local
                == predicted_id
            )
            .float()
            .mean()
            .item()
        )

        result[key] = {
            "count":
                count,
            "k":
                k,
            "true_neighbor_fraction":
                true_fraction,
            "attractor_neighbor_fraction":
                attractor_fraction
        }

    return result


def target_centroid_analysis(
    features,
    labels
):
    centroids = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            labels
            == class_id
        )

        if not mask.any():
            raise RuntimeError(
                f"No target samples for class "
                f"{CLASSES[class_id]}"
            )

        centroid = features[
            mask
        ].mean(
            dim=0
        )

        centroid = F.normalize(
            centroid,
            dim=0
        )

        centroids.append(
            centroid
        )

    centroids = torch.stack(
        centroids,
        dim=0
    )

    similarity = (
        centroids
        @ centroids.t()
    )

    result = {}

    for class_name, class_id in (
        FOCUS_CLASSES.items()
    ):
        competitors = []

        for other_id in range(
            NUM_CLASSES
        ):
            if other_id == class_id:
                continue

            competitors.append(
                (
                    float(
                        similarity[
                            class_id,
                            other_id
                        ].item()
                    ),
                    other_id
                )
            )

        competitors.sort(
            reverse=True
        )

        result[
            class_name
        ] = {
            "nearest_class":
                CLASSES[
                    competitors[0][1]
                ],
            "nearest_similarity":
                competitors[0][0],
            "nearest_distance":
                1.0 - competitors[0][0],
            "car_similarity":
                float(
                    similarity[
                        class_id,
                        CAR_ID
                    ].item()
                ),
            "bus_similarity":
                float(
                    similarity[
                        class_id,
                        BUS_ID
                    ].item()
                ),
            "train_similarity":
                float(
                    similarity[
                        class_id,
                        TRAIN_ID
                    ].item()
                ),
            "truck_similarity":
                float(
                    similarity[
                        class_id,
                        TRUCK_ID
                    ].item()
                )
        }

    return result


def print_knn_block(
    results,
    k
):
    print()
    print(
        f"K = {k}"
    )

    for class_name in (
        "truck",
        "car",
        "bus",
        "train",
        "skateboard",
        "person"
    ):
        row = results[
            class_name
        ]

        print(
            f"{class_name:12s} | "
            f"purity="
            f"{100.0 * row['mean_knn_purity']:6.2f}% | "
            f">=50="
            f"{100.0 * row['fraction_purity_ge_50']:6.2f}% | "
            f">=70="
            f"{100.0 * row['fraction_purity_ge_70']:6.2f}% | "
            f">=90="
            f"{100.0 * row['fraction_purity_ge_90']:6.2f}% | "
            f"majority="
            f"{100.0 * row['majority_neighbor_accuracy']:6.2f}% | "
            f"competitor="
            f"{row['strongest_competitor']}"
        )


def print_flow_block(
    flow,
    k
):
    print()
    print(
        f"Local flow at K={k}"
    )

    for key in (
        "truck_to_car",
        "truck_to_bus",
        "truck_to_train",
        "skateboard_to_person",
        "skateboard_to_knife",
        "bicycle_to_motorcycle"
    ):
        row = flow[
            key
        ]

        if row["count"] == 0:
            print(
                f"  {key}: count=0"
            )
        else:
            print(
                f"  {key}: "
                f"n={row['count']} | "
                f"true_neighbors="
                f"{100.0 * row['true_neighbor_fraction']:6.2f}% | "
                f"attractor_neighbors="
                f"{100.0 * row['attractor_neighbor_fraction']:6.2f}%"
            )


def main():
    set_seed(
        SEED
    )

    print("=" * 90)
    print(
        "VISDA-2017 TARGET-LOCAL KNN GEOMETRY DIAGNOSTIC"
    )
    print("=" * 90)

    print(
        f"seed={SEED}"
    )

    print(
        f"checkpoint={BASE_CHECKPOINT}"
    )

    print(
        f"K values={K_VALUES}"
    )

    print(
        f"query chunk={KNN_QUERY_CHUNK}"
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
        f"Target input dimension: "
        f"{target_features.shape[1]}"
    )

    model = MCDModel()

    source_prototypes = load_checkpoint(
        model
    )

    device = torch.device(
        "cpu"
    )

    model.to(
        device
    )

    print()
    print(
        "Checkpoint loaded successfully."
    )

    print(
        f"Source prototype shape: "
        f"{tuple(source_prototypes.shape)}"
    )

    print()
    print(
        "Collecting adapted target features..."
    )

    target_z = collect_adapted_features(
        model,
        target_features,
        device
    )

    print(
        f"Adapted target shape: "
        f"{tuple(target_z.shape)}"
    )

    print()
    print(
        "Collecting MCD predictions..."
    )

    (
        predictions,
        confidence,
        disagreement
    ) = collect_predictions(
        model,
        target_features,
        device
    )

    prediction_accuracy = (
        predictions
        == target_labels
    ).float().mean().item()

    print(
        f"Overall target accuracy: "
        f"{100.0 * prediction_accuracy:.2f}%"
    )

    print()
    print(
        "=" * 90
    )
    print(
        "COMPUTING TARGET KNN GRAPH"
    )
    print("=" * 90)

    knn_indices = compute_knn_indices(
        target_z,
        MAX_K,
        KNN_QUERY_CHUNK
    )

    if knn_indices.shape != (
        len(target_features),
        MAX_K
    ):
        raise RuntimeError(
            f"Invalid KNN index shape "
            f"{tuple(knn_indices.shape)}"
        )

    print()
    print(
        "KNN graph complete."
    )

    all_knn_results = {}
    all_flow_results = {}

    for k in K_VALUES:
        results = class_knn_statistics(
            target_labels,
            knn_indices,
            k
        )

        flows = flow_analysis(
            target_labels,
            predictions,
            knn_indices,
            k
        )

        all_knn_results[
            str(k)
        ] = results

        all_flow_results[
            str(k)
        ] = flows

        print_knn_block(
            results,
            k
        )

        if k in (
            5,
            20,
            50
        ):
            print_flow_block(
                flows,
                k
            )

    print()
    print("=" * 90)
    print(
        "TRUCK NEIGHBORHOOD COMPOSITION"
    )
    print("=" * 90)

    truck_mask = (
        target_labels
        == TRUCK_ID
    )

    for k in (
        5,
        10,
        20,
        50
    ):
        neighbors = knn_indices[
            truck_mask,
            :k
        ]

        neighbor_labels = target_labels[
            neighbors
        ]

        distribution = torch.bincount(
            neighbor_labels.reshape(-1),
            minlength=NUM_CLASSES
        ).float()

        distribution /= (
            distribution.sum()
            .clamp_min(1.0)
        )

        print()
        print(
            f"K={k}"
        )

        for class_id in torch.argsort(
            distribution,
            descending=True
        )[:8].tolist():
            print(
                f"  {CLASSES[class_id]:12s}: "
                f"{100.0 * distribution[class_id].item():6.2f}%"
            )

    print()
    print("=" * 90)
    print(
        "TRUCK WRONG-PREDICTION NEIGHBORHOODS"
    )
    print("=" * 90)

    for predicted_name, predicted_id in (
        ("car", CAR_ID),
        ("bus", BUS_ID),
        ("train", TRAIN_ID)
    ):
        mask = (
            (target_labels == TRUCK_ID)
            & (predictions == predicted_id)
        )

        count = int(
            mask.sum().item()
        )

        print()
        print(
            f"truck → {predicted_name}: "
            f"{count} samples"
        )

        if count == 0:
            continue

        for k in (
            5,
            20,
            50
        ):
            local = target_labels[
                knn_indices[
                    mask,
                    :k
                ]
            ]

            truck_fraction = (
                (
                    local
                    == TRUCK_ID
                )
                .float()
                .mean()
                .item()
            )

            attractor_fraction = (
                (
                    local
                    == predicted_id
                )
                .float()
                .mean()
                .item()
            )

            print(
                f"  K={k:2d} | "
                f"truck neighbors="
                f"{100.0 * truck_fraction:6.2f}% | "
                f"{predicted_name} neighbors="
                f"{100.0 * attractor_fraction:6.2f}%"
            )

    print()
    print("=" * 90)
    print(
        "SKATEBOARD WRONG-PREDICTION NEIGHBORHOODS"
    )
    print("=" * 90)

    for predicted_name, predicted_id in (
        ("person", PERSON_ID),
        ("knife", 5)
    ):
        mask = (
            (target_labels == SKATEBOARD_ID)
            & (predictions == predicted_id)
        )

        count = int(
            mask.sum().item()
        )

        print()
        print(
            f"skateboard → {predicted_name}: "
            f"{count} samples"
        )

        if count == 0:
            continue

        for k in (
            5,
            20,
            50
        ):
            local = target_labels[
                knn_indices[
                    mask,
                    :k
                ]
            ]

            skateboard_fraction = (
                (
                    local
                    == SKATEBOARD_ID
                )
                .float()
                .mean()
                .item()
            )

            attractor_fraction = (
                (
                    local
                    == predicted_id
                )
                .float()
                .mean()
                .item()
            )

            print(
                f"  K={k:2d} | "
                f"skateboard neighbors="
                f"{100.0 * skateboard_fraction:6.2f}% | "
                f"{predicted_name} neighbors="
                f"{100.0 * attractor_fraction:6.2f}%"
            )

    print()
    print("=" * 90)
    print(
        "TARGET CENTROID GEOMETRY"
    )
    print("=" * 90)

    centroid_geometry = target_centroid_analysis(
        target_z,
        target_labels
    )

    for class_name in (
        "truck",
        "car",
        "bus",
        "train",
        "skateboard",
        "person"
    ):
        row = centroid_geometry[
            class_name
        ]

        print(
            f"{class_name:12s} | "
            f"nearest="
            f"{row['nearest_class']:12s} | "
            f"sim="
            f"{row['nearest_similarity']:.6f} | "
            f"dist="
            f"{row['nearest_distance']:.6f}"
        )

        print(
            f"  truck="
            f"{row['truck_similarity']:.6f} | "
            f"car="
            f"{row['car_similarity']:.6f} | "
            f"bus="
            f"{row['bus_similarity']:.6f} | "
            f"train="
            f"{row['train_similarity']:.6f}"
        )

    print()
    print("=" * 90)
    print(
        "TRUCK SUMMARY"
    )
    print("=" * 90)

    truck_knn = all_knn_results[
        "20"
    ][
        "truck"
    ]

    truck_flow_car = all_flow_results[
        "20"
    ][
        "truck_to_car"
    ]

    truck_flow_bus = all_flow_results[
        "20"
    ][
        "truck_to_bus"
    ]

    truck_flow_train = all_flow_results[
        "20"
    ][
        "truck_to_train"
    ]

    print(
        f"K=20 truck local purity: "
        f"{100.0 * truck_knn['mean_knn_purity']:.2f}%"
    )

    print(
        f"K=20 truck samples with >=50% truck neighbors: "
        f"{100.0 * truck_knn['fraction_purity_ge_50']:.2f}%"
    )

    print(
        f"K=20 truck samples with >=70% truck neighbors: "
        f"{100.0 * truck_knn['fraction_purity_ge_70']:.2f}%"
    )

    if truck_flow_car["count"] > 0:
        print(
            f"Truck→car K=20: "
            f"true="
            f"{100.0 * truck_flow_car['true_neighbor_fraction']:.2f}% | "
            f"car="
            f"{100.0 * truck_flow_car['attractor_neighbor_fraction']:.2f}%"
        )

    if truck_flow_bus["count"] > 0:
        print(
            f"Truck→bus K=20: "
            f"true="
            f"{100.0 * truck_flow_bus['true_neighbor_fraction']:.2f}% | "
            f"bus="
            f"{100.0 * truck_flow_bus['attractor_neighbor_fraction']:.2f}%"
        )

    if truck_flow_train["count"] > 0:
        print(
            f"Truck→train K=20: "
            f"true="
            f"{100.0 * truck_flow_train['true_neighbor_fraction']:.2f}% | "
            f"train="
            f"{100.0 * truck_flow_train['attractor_neighbor_fraction']:.2f}%"
        )

    output = {
        "experiment":
            "visda_target_local_knn_geometry_diagnostic",
        "seed":
            SEED,
        "checkpoint":
            str(BASE_CHECKPOINT),
        "target_samples":
            len(target_features),
        "input_dim":
            INPUT_DIM,
        "hidden_dim":
            HIDDEN_DIM,
        "k_values":
            K_VALUES,
        "target_accuracy":
            100.0 * prediction_accuracy,
        "mean_confidence":
            float(
                confidence.mean().item()
            ),
        "mean_disagreement":
            float(
                disagreement.mean().item()
            ),
        "knn_results":
            all_knn_results,
        "local_flow_results":
            all_flow_results,
        "target_centroid_geometry":
            centroid_geometry,
        "truck_neighborhood": {
            str(k): {
                CLASSES[class_id]:
                    float(
                        (
                            (
                                target_labels[
                                    knn_indices[
                                        truck_mask,
                                        :k
                                    ]
                                ]
                                == class_id
                            )
                            .float()
                            .mean()
                            .item()
                        )
                    )
                for class_id in range(
                    NUM_CLASSES
                )
            }
            for k in K_VALUES
        }
    }

    output_path = (
        OUTPUT_DIR
        / "target_knn_geometry_seed42.json"
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

    knn_path = (
        OUTPUT_DIR
        / "target_knn_indices_seed42.pt"
    )

    torch.save(
        {
            "indices":
                knn_indices,
            "k":
                MAX_K,
            "num_samples":
                len(target_features)
        },
        knn_path
    )

    print()
    print("=" * 90)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 90)

    print(
        f"Summary: {output_path}"
    )

    print(
        f"KNN graph: {knn_path}"
    )


if __name__ == "__main__":
    main()