import json
from pathlib import Path

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

FLOW_REPORT = Path(
    "checkpoints/visda_directional_flow/directional_flow_seed42.json"
)

RESIDUAL_REPORT = Path(
    "checkpoints/visda_mcd_residual_diagnostics/"
    "mcd_residual_diagnostics_seed42.json"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_class_separation"
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
            torch.nn.ReLU(
                inplace=True
            ),
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


def compute_target_centroids_original():
    sums = torch.zeros(
        NUM_CLASSES,
        INPUT_DIM,
        dtype=torch.float64,
    )

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        features = features.double()

        for class_id in range(
            NUM_CLASSES
        ):
            mask = labels == class_id

            if mask.any():
                sums[class_id] += (
                    features[mask].sum(
                        dim=0
                    )
                )

                counts[class_id] += int(
                    mask.sum().item()
                )

    centroids = (
        sums
        / counts.clamp_min(1).unsqueeze(1)
    )

    return centroids, counts


def compute_target_centroids_mcd(model):
    sums = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64,
    )

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        with torch.no_grad():
            z = model.adapter(
                features
            )

        z = z.double()

        for class_id in range(
            NUM_CLASSES
        ):
            mask = labels == class_id

            if mask.any():
                sums[class_id] += (
                    z[mask].sum(
                        dim=0
                    )
                )

                counts[class_id] += int(
                    mask.sum().item()
                )

    centroids = (
        sums
        / counts.clamp_min(1).unsqueeze(1)
    )

    return centroids, counts


def compute_source_centroids_original():
    sums = torch.zeros(
        NUM_CLASSES,
        INPUT_DIM,
        dtype=torch.float64,
    )

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        SOURCE_CACHE
    ):
        features = features.double()

        for class_id in range(
            NUM_CLASSES
        ):
            mask = labels == class_id

            if mask.any():
                sums[class_id] += (
                    features[mask].sum(
                        dim=0
                    )
                )

                counts[class_id] += int(
                    mask.sum().item()
                )

    centroids = (
        sums
        / counts.clamp_min(1).unsqueeze(1)
    )

    return centroids, counts


def compute_source_centroids_mcd(model):
    sums = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64,
    )

    counts = torch.zeros(
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

        for class_id in range(
            NUM_CLASSES
        ):
            mask = labels == class_id

            if mask.any():
                sums[class_id] += (
                    z[mask].sum(
                        dim=0
                    )
                )

                counts[class_id] += int(
                    mask.sum().item()
                )

    centroids = (
        sums
        / counts.clamp_min(1).unsqueeze(1)
    )

    return centroids, counts


def pairwise_distances(centroids):
    distances = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.float64,
    )

    for i in range(
        NUM_CLASSES
    ):
        for j in range(
            NUM_CLASSES
        ):
            if i != j:
                distances[i, j] = torch.norm(
                    centroids[i]
                    - centroids[j]
                )

    return distances


def summarize_pairs(
    before,
    after,
):
    rows = []

    for i in range(
        NUM_CLASSES
    ):
        for j in range(
            i + 1,
            NUM_CLASSES
        ):
            b = before[i, j].item()
            a = after[i, j].item()

            if b == 0.0:
                change_pct = 0.0
            else:
                change_pct = (
                    100.0
                    * (a - b)
                    / b
                )

            rows.append(
                {
                    "class_a":
                        CLASSES[i],
                    "class_b":
                        CLASSES[j],
                    "before":
                        b,
                    "after":
                        a,
                    "delta":
                        a - b,
                    "relative_change_percent":
                        change_pct,
                }
            )

    return rows


def map_confusions():
    with open(
        FLOW_REPORT,
        "r",
        encoding="utf-8",
    ) as handle:
        flow = json.load(handle)

    with open(
        RESIDUAL_REPORT,
        "r",
        encoding="utf-8",
    ) as handle:
        residual = json.load(handle)

    return flow, residual


def main():
    print("=" * 90)
    print("VISDA-2017 CLASS SEPARATION DIAGNOSTIC")
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
        "Computing original source centroids..."
    )

    source_original, source_original_counts = (
        compute_source_centroids_original()
    )

    print(
        "Computing original target centroids..."
    )

    target_original, target_original_counts = (
        compute_target_centroids_original()
    )

    print(
        "Computing MCD source centroids..."
    )

    source_mcd, source_mcd_counts = (
        compute_source_centroids_mcd(
            model
        )
    )

    print(
        "Computing MCD target centroids..."
    )

    target_mcd, target_mcd_counts = (
        compute_target_centroids_mcd(
            model
        )
    )

    source_original_distances = (
        pairwise_distances(
            source_original
        )
    )

    target_original_distances = (
        pairwise_distances(
            target_original
        )
    )

    source_mcd_distances = (
        pairwise_distances(
            source_mcd
        )
    )

    target_mcd_distances = (
        pairwise_distances(
            target_mcd
        )
    )

    source_distance_change = (
        source_mcd_distances
        - source_original_distances
    )

    target_distance_change = (
        target_mcd_distances
        - target_original_distances
    )

    source_pair_rows = summarize_pairs(
        source_original_distances,
        source_mcd_distances,
    )

    target_pair_rows = summarize_pairs(
        target_original_distances,
        target_mcd_distances,
    )

    print()
    print("=" * 90)
    print("TARGET PAIRWISE SEPARATION CHANGES")
    print("=" * 90)

    target_pair_rows_sorted = sorted(
        target_pair_rows,
        key=lambda x: x["delta"],
    )

    print(
        f"{'Pair':28s}"
        f"{'Before':>12s}"
        f"{'After':>12s}"
        f"{'Delta':>12s}"
        f"{'Rel %':>12s}"
    )

    for row in target_pair_rows_sorted[:20]:
        pair = (
            row["class_a"]
            + " ↔ "
            + row["class_b"]
        )

        print(
            f"{pair:28s}"
            f"{row['before']:11.3f}"
            f"{row['after']:11.3f}"
            f"{row['delta']:+11.3f}"
            f"{row['relative_change_percent']:+11.2f}%"
        )

    print()
    print("=" * 90)
    print("LARGEST TARGET SEPARATION INCREASES")
    print("=" * 90)

    for row in sorted(
        target_pair_rows,
        key=lambda x: x["delta"],
        reverse=True,
    )[:15]:
        pair = (
            row["class_a"]
            + " ↔ "
            + row["class_b"]
        )

        print(
            f"{pair:28s} | "
            f"{row['before']:10.3f} → "
            f"{row['after']:10.3f} | "
            f"{row['delta']:+9.3f}"
        )

    print()
    print("=" * 90)
    print("SOURCE PAIRWISE SEPARATION CHANGES")
    print("=" * 90)

    source_pair_rows_sorted = sorted(
        source_pair_rows,
        key=lambda x: x["delta"],
    )

    print(
        f"{'Pair':28s}"
        f"{'Before':>12s}"
        f"{'After':>12s}"
        f"{'Delta':>12s}"
    )

    for row in source_pair_rows_sorted[:20]:
        pair = (
            row["class_a"]
            + " ↔ "
            + row["class_b"]
        )

        print(
            f"{pair:28s}"
            f"{row['before']:11.3f}"
            f"{row['after']:11.3f}"
            f"{row['delta']:+11.3f}"
        )

    print()
    print("=" * 90)
    print("SPECIFIC PAIRS")
    print("=" * 90)

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
        ("bus", "train"),
    ]

    name_to_idx = {
        name: i
        for i, name in enumerate(CLASSES)
    }

    for class_a, class_b in pairs:
        a = name_to_idx[class_a]
        b = name_to_idx[class_b]

        original_target_distance = (
            target_original_distances[a, b].item()
        )

        mcd_target_distance = (
            target_mcd_distances[a, b].item()
        )

        original_source_distance = (
            source_original_distances[a, b].item()
        )

        mcd_source_distance = (
            source_mcd_distances[a, b].item()
        )

        print(
            f"{class_a:12s} ↔ "
            f"{class_b:12s} | "
            f"source "
            f"{original_source_distance:.3f} → "
            f"{mcd_source_distance:.3f} | "
            f"target "
            f"{original_target_distance:.3f} → "
            f"{mcd_target_distance:.3f}"
        )

    print()
    print("=" * 90)
    print("TARGET SEPARATION LOSS BY CLASS")
    print("=" * 90)

    class_loss = []

    for i, class_name in enumerate(
        CLASSES
    ):
        values = []

        for j in range(
            NUM_CLASSES
        ):
            if i == j:
                continue

            before = target_original_distances[
                i, j
            ].item()

            after = target_mcd_distances[
                i, j
            ].item()

            values.append(
                after - before
            )

        mean_change = (
            sum(values)
            / len(values)
        )

        negative_count = sum(
            value < 0
            for value in values
        )

        class_loss.append(
            {
                "class":
                    class_name,
                "mean_pairwise_change":
                    mean_change,
                "pairs_with_reduced_separation":
                    negative_count,
                "pairs_total":
                    len(values),
            }
        )

    for row in sorted(
        class_loss,
        key=lambda x:
            x["mean_pairwise_change"],
    ):
        print(
            f"{row['class']:12s} | "
            f"mean delta "
            f"{row['mean_pairwise_change']:+10.3f} | "
            f"reduced "
            f"{row['pairs_with_reduced_separation']:2d}/"
            f"{row['pairs_total']:2d}"
        )

    print()
    print("=" * 90)
    print("SEPARATION VS CONFUSION SIGNAL")
    print("=" * 90)

    flow, residual = map_confusions()

    confusion_delta = torch.tensor(
        flow["flow_delta_rates"],
        dtype=torch.float64,
    )

    pair_rows = []

    for i in range(
        NUM_CLASSES
    ):
        for j in range(
            NUM_CLASSES
        ):
            if i == j:
                continue

            separation_delta = (
                target_distance_change[i, j]
                .item()
            )

            confusion_delta_value = (
                confusion_delta[i, j]
                .item()
            )

            pair_rows.append(
                {
                    "true_class":
                        CLASSES[i],
                    "predicted_class":
                        CLASSES[j],
                    "separation_delta":
                        separation_delta,
                    "confusion_delta":
                        confusion_delta_value,
                }
            )

    separation_values = torch.tensor(
        [
            row["separation_delta"]
            for row in pair_rows
        ],
        dtype=torch.float64,
    )

    confusion_values = torch.tensor(
        [
            row["confusion_delta"]
            for row in pair_rows
        ],
        dtype=torch.float64,
    )

    if (
        separation_values.std().item() > 0
        and confusion_values.std().item() > 0
    ):
        correlation = torch.corrcoef(
            torch.stack(
                [
                    separation_values,
                    confusion_values,
                ]
            )
        )[0, 1].item()
    else:
        correlation = 0.0

    print(
        f"Correlation between pairwise "
        f"separation change and confusion-flow change: "
        f"{correlation:.6f}"
    )

    print()
    print("=" * 90)
    print("MOST INTERESTING PAIRS")
    print("=" * 90)

    interesting = sorted(
        pair_rows,
        key=lambda x:
            abs(
                x["confusion_delta"]
            ),
        reverse=True,
    )

    for row in interesting[:20]:
        print(
            f"{row['true_class']:12s} → "
            f"{row['predicted_class']:12s} | "
            f"separation "
            f"{row['separation_delta']:+9.3f} | "
            f"flow "
            f"{row['confusion_delta']:+8.3f} pp"
        )

    report = {
        "experiment":
            "visda_class_separation_diagnostic",
        "seed":
            42,
        "target_labels_used_only_for_diagnostics":
            True,
        "source_pairwise_before":
            source_original_distances.tolist(),
        "source_pairwise_after":
            source_mcd_distances.tolist(),
        "target_pairwise_before":
            target_original_distances.tolist(),
        "target_pairwise_after":
            target_mcd_distances.tolist(),
        "source_pairwise_delta":
            source_distance_change.tolist(),
        "target_pairwise_delta":
            target_distance_change.tolist(),
        "target_pair_rows":
            target_pair_rows,
        "source_pair_rows":
            source_pair_rows,
        "class_separation_loss":
            class_loss,
        "separation_confusion_correlation":
            correlation,
        "flow_pairs":
            pair_rows,
        "source_original_counts":
            source_original_counts.tolist(),
        "target_original_counts":
            target_original_counts.tolist(),
        "source_mcd_counts":
            source_mcd_counts.tolist(),
        "target_mcd_counts":
            target_mcd_counts.tolist(),
    }

    report_path = (
        OUTPUT_DIR
        / "class_separation_seed42.json"
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