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

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/geometry_gated_mcd_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_truck_geometry_diagnostic"
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

FOCUS_CLASSES = {
    "truck": TRUCK_ID,
    "car": CAR_ID,
    "bus": BUS_ID,
    "train": TRAIN_ID,
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

        x = payload["features"].float()
        y = payload["labels"].long()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature shape in {path}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"got {x.shape[1]} in {path}"
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

    if x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Final feature dimension mismatch"
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
                f"Expected 2D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D input, got {x.shape[1]}"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected {HIDDEN_DIM}-D output, got {z.shape[1]}"
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

    model.load_state_dict(
        normalize_state_dict(
            payload["student_state_dict"]
        ),
        strict=True
    )

    prototypes = payload[
        "source_prototypes"
    ].float()

    if prototypes.shape != (
        NUM_CLASSES,
        HIDDEN_DIM
    ):
        raise RuntimeError(
            f"Invalid source prototype shape "
            f"{tuple(prototypes.shape)}"
        )

    return payload, prototypes


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

    outputs = []

    for (x,) in loader:
        x = x.to(device)

        z = model.encode(
            x
        )

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Invalid adapted dimension {z.shape[1]}"
            )

        z = F.normalize(
            z,
            dim=1
        )

        outputs.append(
            z.cpu()
        )

    return torch.cat(
        outputs,
        dim=0
    )


@torch.no_grad()
def compute_prediction_outputs(
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

    all_predictions = []
    all_confidence = []
    all_disagreement = []

    for (x,) in loader:
        x = x.to(device)

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

        all_predictions.append(
            predictions.cpu()
        )

        all_confidence.append(
            confidence.cpu()
        )

        all_disagreement.append(
            disagreement.cpu()
        )

    return (
        torch.cat(all_predictions, dim=0),
        torch.cat(all_confidence, dim=0),
        torch.cat(all_disagreement, dim=0)
    )


def normalized_prototypes(
    prototypes
):
    return F.normalize(
        prototypes,
        dim=1
    )


def class_centroids(
    z,
    labels
):
    centroids = []

    counts = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            labels == class_id
        )

        count = int(
            mask.sum().item()
        )

        if count == 0:
            raise RuntimeError(
                f"No samples for class {class_id}"
            )

        centroid = z[
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

        counts.append(
            count
        )

    return (
        torch.stack(
            centroids,
            dim=0
        ),
        counts
    )


def pairwise_cosine_matrix(
    centroids
):
    return centroids @ centroids.t()


def class_geometry_summary(
    z,
    labels,
    prototypes
):
    prototypes = normalized_prototypes(
        prototypes
    )

    similarities = (
        z @ prototypes.t()
    )

    result = {}

    for class_name, class_id in FOCUS_CLASSES.items():
        mask = (
            labels == class_id
        )

        class_similarity = similarities[
            mask
        ]

        true_similarity = class_similarity[
            :,
            class_id
        ]

        competing_values = class_similarity.clone()

        competing_values[
            :,
            class_id
        ] = -float("inf")

        nearest_competitor_similarity, nearest_competitor = (
            competing_values.max(
                dim=1
            )
        )

        geometry_margin = (
            true_similarity
            - nearest_competitor_similarity
        )

        nearest_source = (
            similarities[
                mask
            ].argmax(
                dim=1
            )
        )

        nearest_source_matches = (
            nearest_source
            == class_id
        )

        result[
            class_name
        ] = {
            "count":
                int(
                    mask.sum().item()
                ),
            "true_prototype_similarity_mean":
                float(
                    true_similarity.mean().item()
                ),
            "true_prototype_similarity_median":
                float(
                    true_similarity.median().item()
                ),
            "nearest_competitor_similarity_mean":
                float(
                    nearest_competitor_similarity.mean().item()
                ),
            "geometry_margin_mean":
                float(
                    geometry_margin.mean().item()
                ),
            "geometry_margin_median":
                float(
                    geometry_margin.median().item()
                ),
            "nearest_prototype_accuracy":
                float(
                    nearest_source_matches.float().mean().item()
                ),
            "nearest_prototype_distribution":
                torch.bincount(
                    nearest_source,
                    minlength=NUM_CLASSES
                ).tolist()
        }

    return result


def confusion_geometry(
    z,
    labels,
    predictions,
    prototypes
):
    prototypes = normalized_prototypes(
        prototypes
    )

    similarities = (
        z @ prototypes.t()
    )

    result = {}

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
            "car",
            "truck"
        ),
        (
            "bus",
            "truck"
        ),
        (
            "train",
            "truck"
        ),
    ]

    class_to_id = {
        name: index
        for index, name in enumerate(
            CLASSES
        )
    }

    for true_name, predicted_name in pairs:
        true_id = class_to_id[
            true_name
        ]

        predicted_id = class_to_id[
            predicted_name
        ]

        mask = (
            (labels == true_id)
            & (predictions == predicted_id)
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
                    0
            }
            continue

        pair_similarity = similarities[
            mask
        ]

        true_similarity = pair_similarity[
            :,
            true_id
        ]

        attractor_similarity = pair_similarity[
            :,
            predicted_id
        ]

        strongest_class = pair_similarity.argmax(
            dim=1
        )

        result[key] = {
            "count":
                count,
            "true_prototype_similarity_mean":
                float(
                    true_similarity.mean().item()
                ),
            "attractor_prototype_similarity_mean":
                float(
                    attractor_similarity.mean().item()
                ),
            "true_minus_attractor_mean":
                float(
                    (
                        true_similarity
                        - attractor_similarity
                    ).mean().item()
                ),
            "true_minus_attractor_median":
                float(
                    (
                        true_similarity
                        - attractor_similarity
                    ).median().item()
                ),
            "fraction_nearest_true_prototype":
                float(
                    (
                        strongest_class
                        == true_id
                    ).float().mean().item()
                ),
            "fraction_nearest_attractor_prototype":
                float(
                    (
                        strongest_class
                        == predicted_id
                    ).float().mean().item()
                )
        }

    return result


def truck_competitor_profile(
    z,
    labels,
    prototypes
):
    prototypes = normalized_prototypes(
        prototypes
    )

    similarities = (
        z @ prototypes.t()
    )

    rows = []

    for class_name, class_id in FOCUS_CLASSES.items():
        mask = (
            labels == class_id
        )

        values = similarities[
            mask
        ]

        row = {
            "class":
                class_name,
            "count":
                int(mask.sum().item())
        }

        for competitor_name, competitor_id in (
            ("truck", TRUCK_ID),
            ("car", CAR_ID),
            ("bus", BUS_ID),
            ("train", TRAIN_ID),
        ):
            score = values[
                :,
                competitor_id
            ]

            row[
                f"sim_to_{competitor_name}"
            ] = float(
                score.mean().item()
            )

        truck_score = values[
            :,
            TRUCK_ID
        ]

        car_score = values[
            :,
            CAR_ID
        ]

        bus_score = values[
            :,
            BUS_ID
        ]

        train_score = values[
            :,
            TRAIN_ID
        ]

        competitor_max = torch.maximum(
            torch.maximum(
                car_score,
                bus_score
            ),
            train_score
        )

        row[
            "truck_vs_best_competitor"
        ] = float(
            (
                truck_score
                - competitor_max
            ).mean().item()
        )

        rows.append(
            row
        )

    return rows


def centroid_pairwise_analysis(
    z,
    labels
):
    centroids, counts = class_centroids(
        z,
        labels
    )

    distances = (
        1.0
        - pairwise_cosine_matrix(
            centroids
        )
    )

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
            "car",
            "bus"
        ),
        (
            "car",
            "train"
        ),
        (
            "bus",
            "train"
        ),
    ]

    class_to_id = {
        name: index
        for index, name in enumerate(
            CLASSES
        )
    }

    result = {}

    for a_name, b_name in pairs:
        a = class_to_id[a_name]
        b = class_to_id[b_name]

        result[
            f"{a_name}_to_{b_name}"
        ] = {
            "cosine_distance":
                float(
                    distances[a, b].item()
                ),
            "cosine_similarity":
                float(
                    (1.0 - distances[a, b]).item()
                )
        }

    return (
        result,
        centroids,
        counts
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
        "VISDA-2017 TRUCK STRUCTURAL GEOMETRY DIAGNOSTIC"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
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

    model = MCDModel().to(
        device
    )

    _, source_prototypes = load_checkpoint(
        model
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
        "Collecting adapted source features..."
    )

    source_z = collect_adapted_features(
        model,
        source_features,
        device
    )

    print(
        f"Source adapted shape: "
        f"{tuple(source_z.shape)}"
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
        f"Target adapted shape: "
        f"{tuple(target_z.shape)}"
    )

    print()
    print(
        "Collecting target predictions..."
    )

    (
        target_predictions,
        target_confidence,
        target_disagreement
    ) = compute_prediction_outputs(
        model,
        target_features,
        device
    )

    print()
    print(
        "=" * 90
    )
    print(
        "TRUCK / CAR / BUS / TRAIN GEOMETRY"
    )
    print("=" * 90)

    geometry_summary = class_geometry_summary(
        target_z,
        target_labels,
        source_prototypes
    )

    for class_name in (
        "truck",
        "car",
        "bus",
        "train"
    ):
        row = geometry_summary[
            class_name
        ]

        print()
        print(
            class_name
        )

        print(
            f"count: "
            f"{row['count']}"
        )

        print(
            f"true prototype similarity: "
            f"{row['true_prototype_similarity_mean']:.6f}"
        )

        print(
            f"nearest competitor similarity: "
            f"{row['nearest_competitor_similarity_mean']:.6f}"
        )

        print(
            f"geometry margin: "
            f"{row['geometry_margin_mean']:.6f}"
        )

        print(
            f"nearest source prototype accuracy: "
            f"{100.0 * row['nearest_prototype_accuracy']:.2f}%"
        )

    print()
    print("=" * 90)
    print(
        "TRUCK COMPETITOR PROFILE"
    )
    print("=" * 90)

    competitor_profile = truck_competitor_profile(
        target_z,
        target_labels,
        source_prototypes
    )

    for row in competitor_profile:
        print()
        print(
            row["class"]
        )

        print(
            f"truck similarity: "
            f"{row['sim_to_truck']:.6f}"
        )

        print(
            f"car similarity: "
            f"{row['sim_to_car']:.6f}"
        )

        print(
            f"bus similarity: "
            f"{row['sim_to_bus']:.6f}"
        )

        print(
            f"train similarity: "
            f"{row['sim_to_train']:.6f}"
        )

        print(
            f"truck vs best competitor: "
            f"{row['truck_vs_best_competitor']:.6f}"
        )

    print()
    print("=" * 90)
    print(
        "CONFUSION GEOMETRY"
    )
    print("=" * 90)

    confusion = confusion_geometry(
        target_z,
        target_labels,
        target_predictions,
        source_prototypes
    )

    for key, row in confusion.items():
        print()
        print(
            key
        )

        if row["count"] == 0:
            print(
                "count: 0"
            )
            continue

        print(
            f"count: "
            f"{row['count']}"
        )

        print(
            f"true prototype similarity: "
            f"{row['true_prototype_similarity_mean']:.6f}"
        )

        print(
            f"attractor prototype similarity: "
            f"{row['attractor_prototype_similarity_mean']:.6f}"
        )

        print(
            f"true minus attractor: "
            f"{row['true_minus_attractor_mean']:.6f}"
        )

        print(
            f"nearest true prototype: "
            f"{100.0 * row['fraction_nearest_true_prototype']:.2f}%"
        )

        print(
            f"nearest attractor prototype: "
            f"{100.0 * row['fraction_nearest_attractor_prototype']:.2f}%"
        )

    print()
    print("=" * 90)
    print(
        "TRUCK GEOMETRIC SEPARABILITY"
    )
    print("=" * 90)

    truck_mask = (
        target_labels
        == TRUCK_ID
    )

    truck_similarity = (
        F.normalize(
            target_z,
            dim=1
        )
        @ F.normalize(
            source_prototypes,
            dim=1
        ).t()
    )[truck_mask]

    truck_true = truck_similarity[
        :,
        TRUCK_ID
    ]

    truck_car = truck_similarity[
        :,
        CAR_ID
    ]

    truck_bus = truck_similarity[
        :,
        BUS_ID
    ]

    truck_train = truck_similarity[
        :,
        TRAIN_ID
    ]

    truck_best_competitor = torch.maximum(
        torch.maximum(
            truck_car,
            truck_bus
        ),
        truck_train
    )

    truck_margin = (
        truck_true
        - truck_best_competitor
    )

    print(
        f"True truck similarity mean: "
        f"{truck_true.mean().item():.6f}"
    )

    print(
        f"Car similarity mean: "
        f"{truck_car.mean().item():.6f}"
    )

    print(
        f"Bus similarity mean: "
        f"{truck_bus.mean().item():.6f}"
    )

    print(
        f"Train similarity mean: "
        f"{truck_train.mean().item():.6f}"
    )

    print(
        f"Truck vs best competitor margin: "
        f"{truck_margin.mean().item():.6f}"
    )

    print(
        f"Truck samples with positive margin: "
        f"{100.0 * (truck_margin > 0).float().mean().item():.2f}%"
    )

    print(
        f"Truck samples with margin >= 0.10: "
        f"{100.0 * (truck_margin >= 0.10).float().mean().item():.2f}%"
    )

    print(
        f"Truck samples with margin >= 0.20: "
        f"{100.0 * (truck_margin >= 0.20).float().mean().item():.2f}%"
    )

    print()
    print("=" * 90)
    print(
        "CENTROID PAIRWISE DISTANCES"
    )
    print("=" * 90)

    (
        centroid_pairs,
        target_centroids,
        target_counts
    ) = centroid_pairwise_analysis(
        target_z,
        target_labels
    )

    for pair, row in centroid_pairs.items():
        print(
            f"{pair:22s} | "
            f"cosine similarity="
            f"{row['cosine_similarity']:.6f} | "
            f"distance="
            f"{row['cosine_distance']:.6f}"
        )

    print()
    print("=" * 90)
    print(
        "PREDICTION VS GEOMETRY"
    )
    print("=" * 90)

    prediction_accuracy = float(
        (
            target_predictions
            == target_labels
        ).float().mean().item()
    )

    truck_prediction = (
        target_predictions[
            truck_mask
        ]
        == TRUCK_ID
    )

    truck_accuracy = float(
        truck_prediction.float().mean().item()
    )

    print(
        f"Overall target accuracy: "
        f"{100.0 * prediction_accuracy:.2f}%"
    )

    print(
        f"Truck accuracy: "
        f"{100.0 * truck_accuracy:.2f}%"
    )

    print(
        f"Truck mean confidence: "
        f"{target_confidence[truck_mask].mean().item():.6f}"
    )

    print(
        f"Truck mean disagreement: "
        f"{target_disagreement[truck_mask].mean().item():.6f}"
    )

    print(
        f"Truck geometric nearest-prototype rate: "
        f"{100.0 * geometry_summary['truck']['nearest_prototype_accuracy']:.2f}%"
    )

    output = {
        "experiment":
            "visda_truck_structural_geometry_diagnostic",
        "seed":
            SEED,
        "checkpoint":
            str(BASE_CHECKPOINT),
        "target_accuracy":
            100.0 * prediction_accuracy,
        "truck_accuracy":
            100.0 * truck_accuracy,
        "truck_confidence":
            float(
                target_confidence[
                    truck_mask
                ].mean().item()
            ),
        "truck_disagreement":
            float(
                target_disagreement[
                    truck_mask
                ].mean().item()
            ),
        "geometry_summary":
            geometry_summary,
        "truck_competitor_profile":
            competitor_profile,
        "confusion_geometry":
            confusion,
        "truck_geometric_separability": {
            "truck_similarity":
                float(
                    truck_true.mean().item()
                ),
            "car_similarity":
                float(
                    truck_car.mean().item()
                ),
            "bus_similarity":
                float(
                    truck_bus.mean().item()
                ),
            "train_similarity":
                float(
                    truck_train.mean().item()
                ),
            "truck_vs_best_competitor":
                float(
                    truck_margin.mean().item()
                ),
            "positive_margin_fraction":
                float(
                    (
                        truck_margin > 0
                    ).float().mean().item()
                ),
            "margin_0_10_fraction":
                float(
                    (
                        truck_margin >= 0.10
                    ).float().mean().item()
                ),
            "margin_0_20_fraction":
                float(
                    (
                        truck_margin >= 0.20
                    ).float().mean().item()
                )
        },
        "centroid_pairwise":
            centroid_pairs,
        "target_centroid_counts":
            {
                CLASSES[i]:
                    int(target_counts[i])
                for i in range(NUM_CLASSES)
            }
    }

    output_path = (
        OUTPUT_DIR
        / "truck_geometry_diagnostic_seed42.json"
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