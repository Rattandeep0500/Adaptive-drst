import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SEED = 42

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

PROBE_CHECKPOINT = Path(
    "checkpoints/visda_cached_source_only/linear_probe_seed42.pt"
)

OUTPUT_DIR = Path("checkpoints/visda_feature_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 12
FEATURE_DIM = 2048

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


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES,
        )

    def forward(self, x):
        return self.classifier(x)


def iter_chunks(cache_dir):
    files = sorted(cache_dir.glob("chunk_*.pt"))

    if not files:
        raise RuntimeError(
            f"No cache chunks found in {cache_dir}"
        )

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu",
        )

        features = payload["features"].float()
        labels = payload["labels"].long()

        yield path, features, labels


def load_probe():
    if not PROBE_CHECKPOINT.exists():
        raise RuntimeError(
            f"Probe checkpoint not found: {PROBE_CHECKPOINT.resolve()}"
        )

    checkpoint = torch.load(
        PROBE_CHECKPOINT,
        map_location="cpu",
    )

    model = LinearProbe()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def compute_source_statistics():
    sums = torch.zeros(
        NUM_CLASSES,
        FEATURE_DIM,
        dtype=torch.float64,
    )

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for _, features, labels in iter_chunks(
        SOURCE_CACHE
    ):
        features = features.double()

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                sums[class_id] += features[mask].sum(
                    dim=0
                )

                counts[class_id] += int(
                    mask.sum().item()
                )

    centroids = (
        sums
        / counts.clamp_min(1).unsqueeze(1)
    )

    scatter_sum = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    scatter_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for _, features, labels in iter_chunks(
        SOURCE_CACHE
    ):
        features = features.double()

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                x = features[mask]
                centroid = centroids[class_id]

                dist2 = (
                    (x - centroid) ** 2
                ).sum(dim=1)

                scatter_sum[class_id] += dist2.sum()
                scatter_count[class_id] += dist2.numel()

    within = (
        scatter_sum
        / scatter_count.clamp_min(1)
    )

    return centroids, counts, within


def analyze_target(model, source_centroids):
    confusion = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
    )

    true_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    predicted_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    correct_per_class = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    total_per_class = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    target_sums = torch.zeros(
        NUM_CLASSES,
        FEATURE_DIM,
        dtype=torch.float64,
    )

    target_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    true_source_distance_sum = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    predicted_source_distance_sum = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    distance_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    margin_sum = 0.0
    margin_sq_sum = 0.0

    correct_margin_sum = 0.0
    correct_margin_count = 0

    wrong_margin_sum = 0.0
    wrong_margin_count = 0

    total_samples = 0

    for _, features, labels in iter_chunks(
        TARGET_CACHE
    ):
        with torch.no_grad():
            logits = model(features)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            top_values, top_indices = torch.topk(
                probabilities,
                k=2,
                dim=1,
            )

            predictions = top_indices[:, 0]

            margins = (
                top_values[:, 0]
                - top_values[:, 1]
            )

        total_samples += labels.size(0)

        predicted_counts += torch.bincount(
            predictions,
            minlength=NUM_CLASSES,
        )

        for true_class in range(NUM_CLASSES):
            mask = labels == true_class

            if not mask.any():
                continue

            true_count = int(
                mask.sum().item()
            )

            true_counts[true_class] += true_count
            total_per_class[true_class] += true_count

            predicted_for_class = predictions[mask]

            correct_count = int(
                (
                    predicted_for_class
                    == true_class
                )
                .sum()
                .item()
            )

            correct_per_class[true_class] += correct_count

            confusion[true_class] += torch.bincount(
                predicted_for_class,
                minlength=NUM_CLASSES,
            )

            class_features = features[mask].double()

            target_sums[true_class] += (
                class_features.sum(dim=0)
            )

            target_counts[true_class] += true_count

            source_centroid = source_centroids[
                true_class
            ]

            true_dist = torch.sqrt(
                (
                    class_features
                    - source_centroid
                ).pow(2).sum(dim=1)
                + 1e-12
            )

            true_source_distance_sum[
                true_class
            ] += true_dist.sum()

            predicted_centroids = source_centroids[
                predictions[mask]
            ]

            predicted_dist = torch.sqrt(
                (
                    class_features
                    - predicted_centroids
                ).pow(2).sum(dim=1)
                + 1e-12
            )

            predicted_source_distance_sum[
                true_class
            ] += predicted_dist.sum()

            distance_count[true_class] += true_count

        margin_sum += margins.sum().item()

        margin_sq_sum += (
            margins.pow(2).sum().item()
        )

        correct_mask = predictions == labels

        if correct_mask.any():
            correct_margin_sum += (
                margins[correct_mask]
                .sum()
                .item()
            )

            correct_margin_count += int(
                correct_mask.sum().item()
            )

        wrong_mask = ~correct_mask

        if wrong_mask.any():
            wrong_margin_sum += (
                margins[wrong_mask]
                .sum()
                .item()
            )

            wrong_margin_count += int(
                wrong_mask.sum().item()
            )

    target_centroids = (
        target_sums
        / target_counts.clamp_min(1).unsqueeze(1)
    )

    target_scatter_sum = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    target_scatter_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for _, features, labels in iter_chunks(
        TARGET_CACHE
    ):
        features = features.double()

        for class_id in range(NUM_CLASSES):
            mask = labels == class_id

            if mask.any():
                x = features[mask]
                centroid = target_centroids[class_id]

                dist2 = (
                    (x - centroid) ** 2
                ).sum(dim=1)

                target_scatter_sum[class_id] += (
                    dist2.sum()
                )

                target_scatter_count[class_id] += (
                    dist2.numel()
                )

    target_within = (
        target_scatter_sum
        / target_scatter_count.clamp_min(1)
    )

    centroid_shift = torch.norm(
        target_centroids
        - source_centroids,
        dim=1,
    )

    true_source_distance = (
        true_source_distance_sum
        / distance_count.clamp_min(1)
    )

    predicted_source_distance = (
        predicted_source_distance_sum
        / distance_count.clamp_min(1)
    )

    per_class_accuracy = (
        100.0
        * correct_per_class.float()
        / total_per_class.clamp_min(1)
    )

    overall_accuracy = (
        100.0
        * correct_per_class.sum().item()
        / max(total_samples, 1)
    )

    mean_class_accuracy = float(
        per_class_accuracy.mean().item()
    )

    prediction_distribution = (
        100.0
        * predicted_counts.float()
        / max(total_samples, 1)
    )

    margin_mean = (
        margin_sum
        / max(total_samples, 1)
    )

    margin_variance = (
        margin_sq_sum
        / max(total_samples, 1)
        - margin_mean ** 2
    )

    margin_std = float(
        np.sqrt(max(margin_variance, 0.0))
    )

    correct_margin_mean = (
        correct_margin_sum
        / max(correct_margin_count, 1)
    )

    wrong_margin_mean = (
        wrong_margin_sum
        / max(wrong_margin_count, 1)
    )

    return {
        "confusion": confusion,
        "true_counts": true_counts,
        "predicted_counts": predicted_counts,
        "prediction_distribution": prediction_distribution,
        "correct_per_class": correct_per_class,
        "total_per_class": total_per_class,
        "per_class_accuracy": per_class_accuracy,
        "overall_accuracy": overall_accuracy,
        "mean_class_accuracy": mean_class_accuracy,
        "target_centroids": target_centroids,
        "target_within": target_within,
        "centroid_shift": centroid_shift,
        "true_source_distance": true_source_distance,
        "predicted_source_distance": predicted_source_distance,
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "correct_margin_mean": correct_margin_mean,
        "wrong_margin_mean": wrong_margin_mean,
    }


def centroid_distance_matrix(centroids):
    matrix = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.float64,
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j:
                matrix[i, j] = torch.norm(
                    centroids[i]
                    - centroids[j]
                )

    return matrix


def print_confusion_matrix(confusion):
    print()
    print("=" * 90)
    print("TARGET CONFUSION MATRIX")
    print("=" * 90)
    print("Rows = true class | Columns = predicted class")

    header = (
        "true\\pred".ljust(14)
        + "".join(
            f"{i:>10}"
            for i in range(NUM_CLASSES)
        )
    )

    print(header)

    for i, class_name in enumerate(CLASSES):
        row = confusion[i].tolist()

        text = class_name.ljust(14)

        text += "".join(
            f"{value:>10}"
            for value in row
        )

        print(text)


def tensor_to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    return value


def main():
    print("=" * 90)
    print("VISDA-2017 FEATURE-SPACE DIAGNOSTICS")
    print("=" * 90)

    print(
        "Target labels are used only for diagnostics."
    )

    print(
        "No training or checkpoint selection occurs here."
    )

    model = load_probe()

    print()
    print(
        "Computing source centroids and compactness..."
    )

    (
        source_centroids,
        source_counts,
        source_within,
    ) = compute_source_statistics()

    print(
        "Analyzing target predictions and geometry..."
    )

    target_stats = analyze_target(
        model,
        source_centroids,
    )

    source_centroid_distances = (
        centroid_distance_matrix(
            source_centroids
        )
    )

    target_centroid_distances = (
        centroid_distance_matrix(
            target_stats["target_centroids"]
        )
    )

    print()
    print("=" * 90)
    print("CLASS PERFORMANCE")
    print("=" * 90)

    for i, class_name in enumerate(CLASSES):
        print(
            f"{class_name:12s} | "
            f"accuracy "
            f"{target_stats['per_class_accuracy'][i].item():6.2f}% | "
            f"true n "
            f"{int(target_stats['true_counts'][i]):6d} | "
            f"pred n "
            f"{int(target_stats['predicted_counts'][i]):6d}"
        )

    print()
    print(
        f"Target overall accuracy: "
        f"{target_stats['overall_accuracy']:.2f}%"
    )

    print(
        f"Target mean-class accuracy: "
        f"{target_stats['mean_class_accuracy']:.2f}%"
    )

    print()
    print("=" * 90)
    print("PREDICTION DISTRIBUTION")
    print("=" * 90)

    for i, class_name in enumerate(CLASSES):
        print(
            f"{class_name:12s}: "
            f"{target_stats['prediction_distribution'][i].item():6.2f}%"
        )

    print()
    print("=" * 90)
    print("CLASSIFIER MARGIN DIAGNOSTIC")
    print("=" * 90)

    print(
        f"Mean margin overall: "
        f"{target_stats['margin_mean']:.6f}"
    )

    print(
        f"Margin std: "
        f"{target_stats['margin_std']:.6f}"
    )

    print(
        f"Mean margin on correct predictions: "
        f"{target_stats['correct_margin_mean']:.6f}"
    )

    print(
        f"Mean margin on wrong predictions: "
        f"{target_stats['wrong_margin_mean']:.6f}"
    )

    print()
    print("=" * 90)
    print("SOURCE / TARGET GEOMETRY")
    print("=" * 90)

    for i, class_name in enumerate(CLASSES):
        print(
            f"{class_name:12s} | "
            f"source within "
            f"{source_within[i].item():12.2f} | "
            f"target within "
            f"{target_stats['target_within'][i].item():12.2f} | "
            f"centroid shift "
            f"{target_stats['centroid_shift'][i].item():12.2f} | "
            f"target→true-source "
            f"{target_stats['true_source_distance'][i].item():12.2f} | "
            f"target→pred-source "
            f"{target_stats['predicted_source_distance'][i].item():12.2f}"
        )

    print()
    print("=" * 90)
    print("HARDEST CLASSES")
    print("=" * 90)

    hardest = sorted(
        [
            (
                float(
                    target_stats["per_class_accuracy"][i]
                ),
                CLASSES[i],
            )
            for i in range(NUM_CLASSES)
        ]
    )

    for accuracy, class_name in hardest:
        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}%"
        )

    print()
    print("=" * 90)
    print("STRONGEST TARGET CONFUSIONS")
    print("=" * 90)

    confusion = target_stats["confusion"]

    pairs = []

    for i in range(NUM_CLASSES):
        row_total = int(
            confusion[i].sum().item()
        )

        for j in range(NUM_CLASSES):
            if i == j:
                continue

            count = int(
                confusion[i, j].item()
            )

            rate = (
                100.0 * count / row_total
                if row_total > 0
                else 0.0
            )

            pairs.append(
                (
                    rate,
                    count,
                    CLASSES[i],
                    CLASSES[j],
                )
            )

    pairs.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    for rate, count, true_name, pred_name in pairs[:20]:
        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{rate:6.2f}% | "
            f"{count:6d} samples"
        )

    print_confusion_matrix(confusion)

    report = {
        "experiment": "visda_feature_space_diagnostics",
        "seed": SEED,
        "feature_dim": FEATURE_DIM,
        "classes": CLASSES,
        "probe_checkpoint": str(
            PROBE_CHECKPOINT
        ),
        "target_labels_used_for_diagnostics_only": True,
        "source_counts": tensor_to_list(
            source_counts
        ),
        "source_within_class_distance": tensor_to_list(
            source_within
        ),
        "target_within_class_distance": tensor_to_list(
            target_stats["target_within"]
        ),
        "centroid_shift": tensor_to_list(
            target_stats["centroid_shift"]
        ),
        "true_source_distance": tensor_to_list(
            target_stats["true_source_distance"]
        ),
        "predicted_source_distance": tensor_to_list(
            target_stats["predicted_source_distance"]
        ),
        "per_class_accuracy": tensor_to_list(
            target_stats["per_class_accuracy"]
        ),
        "overall_accuracy":
            target_stats["overall_accuracy"],
        "mean_class_accuracy":
            target_stats["mean_class_accuracy"],
        "true_counts": tensor_to_list(
            target_stats["true_counts"]
        ),
        "predicted_counts": tensor_to_list(
            target_stats["predicted_counts"]
        ),
        "prediction_distribution": tensor_to_list(
            target_stats["prediction_distribution"]
        ),
        "margin_mean":
            target_stats["margin_mean"],
        "margin_std":
            target_stats["margin_std"],
        "correct_margin_mean":
            target_stats["correct_margin_mean"],
        "wrong_margin_mean":
            target_stats["wrong_margin_mean"],
        "confusion_matrix": tensor_to_list(
            confusion
        ),
        "source_centroid_distance_matrix":
            tensor_to_list(
                source_centroid_distances
            ),
        "target_centroid_distance_matrix":
            tensor_to_list(
                target_centroid_distances
            ),
        "strongest_confusions": [
            {
                "rate_percent": rate,
                "count": count,
                "true_class": true_name,
                "predicted_class": pred_name,
            }
            for rate, count, true_name, pred_name in pairs[:20]
        ],
    }

    report_path = (
        OUTPUT_DIR
        / "feature_diagnostics_seed42.json"
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
    print("DIAGNOSTIC REPORT")
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )

    print()
    print("DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()