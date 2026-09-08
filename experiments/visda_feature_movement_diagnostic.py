import json
from pathlib import Path

import numpy as np
import torch


NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_feature_movement"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM,
            ),
            torch.nn.BatchNorm1d(
                HIDDEN_DIM
            ),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MCDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.adapter = Adapter()

        self.classifier1 = torch.nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

        self.classifier2 = torch.nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

    def forward(self, x):
        z = self.adapter(x)

        return (
            self.classifier1(z),
            self.classifier2(z),
            z,
        )


def load_chunks(cache_dir):
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No chunks found in {cache_dir}"
        )

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu",
        )

        yield (
            payload["features"].float(),
            payload["labels"].long(),
        )


def load_mcd():
    checkpoint = torch.load(
        MCD_CHECKPOINT,
        map_location="cpu",
    )

    model = MCDModel()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def compute_class_centroids():
    source_sum = torch.zeros(
        NUM_CLASSES,
        INPUT_DIM,
        dtype=torch.float64,
    )

    target_sum = torch.zeros(
        NUM_CLASSES,
        INPUT_DIM,
        dtype=torch.float64,
    )

    source_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    target_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        SOURCE_CACHE
    ):
        features = features.double()

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                source_sum[class_id] += (
                    features[mask].sum(dim=0)
                )

                source_count[class_id] += int(
                    mask.sum().item()
                )

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        features = features.double()

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                target_sum[class_id] += (
                    features[mask].sum(dim=0)
                )

                target_count[class_id] += int(
                    mask.sum().item()
                )

    source_centroids = (
        source_sum
        / source_count.clamp_min(1).unsqueeze(1)
    )

    target_centroids = (
        target_sum
        / target_count.clamp_min(1).unsqueeze(1)
    )

    return (
        source_centroids,
        target_centroids,
    )


def compute_mcd_centroids(model):
    source_sum = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64,
    )

    target_sum = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64,
    )

    source_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    target_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        SOURCE_CACHE
    ):
        with torch.no_grad():
            z = model.adapter(
                features
            )

        z = z.double()

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                source_sum[class_id] += (
                    z[mask].sum(dim=0)
                )

                source_count[class_id] += int(
                    mask.sum().item()
                )

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        with torch.no_grad():
            z = model.adapter(
                features
            )

        z = z.double()

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                target_sum[class_id] += (
                    z[mask].sum(dim=0)
                )

                target_count[class_id] += int(
                    mask.sum().item()
                )

    source_centroids = (
        source_sum
        / source_count.clamp_min(1).unsqueeze(1)
    )

    target_centroids = (
        target_sum
        / target_count.clamp_min(1).unsqueeze(1)
    )

    return (
        source_centroids,
        target_centroids,
    )


def analyze_target_movement(
    model,
    original_source_centroids,
    original_target_centroids,
    mcd_source_centroids,
    mcd_target_centroids,
):
    results = []

    for class_id in range(NUM_CLASSES):
        original_source = (
            original_source_centroids[
                class_id
            ]
        )

        original_target = (
            original_target_centroids[
                class_id
            ]
        )

        mcd_source = (
            mcd_source_centroids[
                class_id
            ]
        )

        mcd_target = (
            mcd_target_centroids[
                class_id
            ]
        )

        original_shift = torch.norm(
            original_target
            - original_source
        ).item()

        mcd_shift = torch.norm(
            mcd_target
            - mcd_source
        ).item()

        target_movement = torch.norm(
            mcd_target
            - original_target[:HIDDEN_DIM]
        ).item()

        source_movement = torch.norm(
            mcd_source
            - original_source[:HIDDEN_DIM]
        ).item()

        target_to_true_before = (
            torch.norm(
                original_target
                - original_source
            ).item()
        )

        target_to_true_after = (
            torch.norm(
                mcd_target
                - mcd_source
            ).item()
        )

        results.append(
            {
                "class": CLASSES[class_id],
                "original_target_source_shift":
                    original_shift,
                "mcd_target_source_shift":
                    mcd_shift,
                "target_centroid_movement":
                    target_movement,
                "source_centroid_movement":
                    source_movement,
                "target_true_class_distance_before":
                    target_to_true_before,
                "target_true_class_distance_after":
                    target_to_true_after,
            }
        )

    return results


def compute_directional_alignment(
    model,
    original_target_centroids,
    mcd_source_centroids,
):
    rows = []

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        with torch.no_grad():
            z = model.adapter(
                features
            )

            logits1 = model.classifier1(
                z
            )

            logits2 = model.classifier2(
                z
            )

            probabilities = (
                torch.softmax(
                    logits1,
                    dim=1,
                )
                + torch.softmax(
                    logits2,
                    dim=1,
                )
            ) / 2.0

            predictions = probabilities.argmax(
                dim=1
            )

        for i in range(
            features.shape[0]
        ):
            true_class = int(
                labels[i].item()
            )

            predicted_class = int(
                predictions[i].item()
            )

            original_target = (
                original_target_centroids[
                    true_class
                ]
            )

            z_i = z[i].double()

            if z_i.shape[0] != HIDDEN_DIM:
                continue

            predicted_centroid = (
                mcd_source_centroids[
                    predicted_class
                ]
            )

            true_centroid = (
                mcd_source_centroids[
                    true_class
                ]
            )

            toward_predicted = torch.norm(
                z_i
                - predicted_centroid
            ).item()

            toward_true = torch.norm(
                z_i
                - true_centroid
            ).item()

            rows.append(
                (
                    true_class,
                    predicted_class,
                    toward_true,
                    toward_predicted,
                )
            )

    return rows


def main():
    print("=" * 90)
    print("VISDA-2017 FEATURE MOVEMENT DIAGNOSTIC")
    print("=" * 90)

    print(
        "Target labels are used only for diagnostics."
    )

    print()
    print(
        "Loading corrected MCD model..."
    )

    model = load_mcd()

    print(
        "Computing original feature centroids..."
    )

    (
        original_source_centroids,
        original_target_centroids,
    ) = compute_class_centroids()

    print(
        "Computing MCD feature centroids..."
    )

    (
        mcd_source_centroids,
        mcd_target_centroids,
    ) = compute_mcd_centroids(
        model
    )

    print()
    print("=" * 90)
    print("CLASS-CENTROID MOVEMENT")
    print("=" * 90)

    centroid_results = []

    for class_id, class_name in enumerate(
        CLASSES
    ):
        source_original = (
            original_source_centroids[
                class_id
            ]
        )

        target_original = (
            original_target_centroids[
                class_id
            ]
        )

        source_mcd = (
            mcd_source_centroids[
                class_id
            ]
        )

        target_mcd = (
            mcd_target_centroids[
                class_id
            ]
        )

        original_shift = torch.norm(
            target_original
            - source_original
        ).item()

        mcd_shift = torch.norm(
            target_mcd
            - source_mcd
        ).item()

        source_movement = torch.norm(
            source_mcd
            - source_original[:HIDDEN_DIM]
        ).item()

        target_movement = torch.norm(
            target_mcd
            - target_original[:HIDDEN_DIM]
        ).item()

        print(
            f"{class_name:12s} | "
            f"original shift "
            f"{original_shift:10.3f} | "
            f"MCD shift "
            f"{mcd_shift:10.3f} | "
            f"source move "
            f"{source_movement:10.3f} | "
            f"target move "
            f"{target_movement:10.3f}"
        )

        centroid_results.append(
            {
                "class": class_name,
                "original_shift":
                    original_shift,
                "mcd_shift":
                    mcd_shift,
                "source_movement":
                    source_movement,
                "target_movement":
                    target_movement,
            }
        )

    print()
    print("=" * 90)
    print("MCD SHIFT CHANGE")
    print("=" * 90)

    for row in centroid_results:
        delta = (
            row["mcd_shift"]
            - row["original_shift"]
        )

        print(
            f"{row['class']:12s} | "
            f"{row['original_shift']:10.3f} → "
            f"{row['mcd_shift']:10.3f} | "
            f"delta {delta:+10.3f}"
        )

    print()
    print("=" * 90)
    print("TARGET CENTROID NEAREST SOURCE CLASS")
    print("=" * 90)

    for class_id, class_name in enumerate(
        CLASSES
    ):
        target_centroid = (
            mcd_target_centroids[
                class_id
            ]
        )

        distances = torch.norm(
            mcd_source_centroids
            - target_centroid.unsqueeze(0),
            dim=1,
        )

        nearest = int(
            distances.argmin().item()
        )

        true_distance = (
            distances[class_id].item()
        )

        nearest_distance = (
            distances[nearest].item()
        )

        print(
            f"{class_name:12s} | "
            f"nearest source class "
            f"{CLASSES[nearest]:12s} | "
            f"true dist "
            f"{true_distance:10.3f} | "
            f"nearest dist "
            f"{nearest_distance:10.3f}"
        )

    print()
    print("=" * 90)
    print("PROBLEMATIC CLASS PAIRS")
    print("=" * 90)

    pairs = [
        ("truck", "car"),
        ("truck", "train"),
        ("truck", "bus"),
        ("skateboard", "person"),
        ("skateboard", "knife"),
        ("bicycle", "motorcycle"),
        ("person", "knife"),
    ]

    name_to_idx = {
        name: i
        for i, name in enumerate(CLASSES)
    }

    for class_a, class_b in pairs:
        a = name_to_idx[class_a]
        b = name_to_idx[class_b]

        target_a = mcd_target_centroids[a]
        target_b = mcd_target_centroids[b]

        source_a = mcd_source_centroids[a]
        source_b = mcd_source_centroids[b]

        target_distance = torch.norm(
            target_a
            - target_b
        ).item()

        source_distance = torch.norm(
            source_a
            - source_b
        ).item()

        print(
            f"{class_a:12s} ↔ "
            f"{class_b:12s} | "
            f"source distance "
            f"{source_distance:10.3f} | "
            f"target distance "
            f"{target_distance:10.3f}"
        )

    report = {
        "experiment":
            "visda_feature_movement_diagnostic",
        "seed":
            42,
        "target_labels_used_only_for_diagnostics":
            True,
        "centroid_results":
            centroid_results,
    }

    report_path = (
        OUTPUT_DIR
        / "feature_movement_seed42.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print()
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )


if __name__ == "__main__":
    main()