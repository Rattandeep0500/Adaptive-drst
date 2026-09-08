import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

SOURCE_PROBE = Path(
    "checkpoints/visda_cached_source_only/linear_probe_seed42.pt"
)

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_mcd_residual_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 2048

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
            INPUT_DIM,
            NUM_CLASSES,
        )

    def forward(self, x):
        return self.classifier(x)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM,
            ),
            nn.BatchNorm1d(
                HIDDEN_DIM
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


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


def load_source_probe():
    checkpoint = torch.load(
        SOURCE_PROBE,
        map_location="cpu",
    )

    model = LinearProbe()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


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


def analyze_baseline(model):
    confusion = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
    )

    correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    totals = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    predicted = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    margin_sum = 0.0
    confidence_sum = 0.0
    total = 0

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        with torch.no_grad():
            logits = model(features)

            probabilities = F.softmax(
                logits,
                dim=1,
            )

            top_values, top_indices = torch.topk(
                probabilities,
                2,
                dim=1,
            )

            predictions = top_indices[:, 0]

            margins = (
                top_values[:, 0]
                - top_values[:, 1]
            )

            confidence = top_values[:, 0]

        total += labels.size(0)

        predicted += torch.bincount(
            predictions,
            minlength=NUM_CLASSES,
        )

        for true_class in range(
            NUM_CLASSES
        ):
            mask = labels == true_class

            if mask.any():
                count = int(
                    mask.sum().item()
                )

                totals[true_class] += count

                correct[true_class] += int(
                    (
                        predictions[mask]
                        == true_class
                    )
                    .sum()
                    .item()
                )

                confusion[true_class] += (
                    torch.bincount(
                        predictions[mask],
                        minlength=NUM_CLASSES,
                    )
                )

        margin_sum += margins.sum().item()
        confidence_sum += confidence.sum().item()

    per_class = (
        100.0
        * correct.float()
        / totals.clamp_min(1)
    )

    mean_class = float(
        per_class.mean().item()
    )

    overall = (
        100.0
        * correct.sum().item()
        / total
    )

    return {
        "confusion": confusion,
        "correct": correct,
        "totals": totals,
        "predicted": predicted,
        "per_class": per_class,
        "mean_class": mean_class,
        "overall": overall,
        "mean_margin": (
            margin_sum / total
        ),
        "mean_confidence": (
            confidence_sum / total
        ),
    }


def analyze_mcd(model):
    confusion = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
    )

    correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    totals = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    predicted = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    margin_sum = 0.0
    confidence_sum = 0.0
    disagreement_sum = 0.0
    total = 0

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
        TARGET_CACHE
    ):
        with torch.no_grad():
            logits1, logits2, z = model(
                features
            )

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

            top_values, top_indices = torch.topk(
                probabilities,
                2,
                dim=1,
            )

            predictions = top_indices[:, 0]

            margins = (
                top_values[:, 0]
                - top_values[:, 1]
            )

            confidence = top_values[:, 0]

            disagreement = (
                torch.abs(p1 - p2)
                .mean(dim=1)
            )

        total += labels.size(0)

        predicted += torch.bincount(
            predictions,
            minlength=NUM_CLASSES,
        )

        for true_class in range(
            NUM_CLASSES
        ):
            mask = labels == true_class

            if mask.any():
                count = int(
                    mask.sum().item()
                )

                totals[true_class] += count

                correct[true_class] += int(
                    (
                        predictions[mask]
                        == true_class
                    )
                    .sum()
                    .item()
                )

                confusion[true_class] += (
                    torch.bincount(
                        predictions[mask],
                        minlength=NUM_CLASSES,
                    )
                )

                target_sum[true_class] += (
                    z[mask]
                    .double()
                    .sum(dim=0)
                )

                target_count[true_class] += count

        margin_sum += margins.sum().item()
        confidence_sum += confidence.sum().item()
        disagreement_sum += disagreement.sum().item()

    per_class = (
        100.0
        * correct.float()
        / totals.clamp_min(1)
    )

    mean_class = float(
        per_class.mean().item()
    )

    overall = (
        100.0
        * correct.sum().item()
        / total
    )

    target_centroids = (
        target_sum
        / target_count.clamp_min(1).unsqueeze(1)
    )

    return {
        "confusion": confusion,
        "correct": correct,
        "totals": totals,
        "predicted": predicted,
        "per_class": per_class,
        "mean_class": mean_class,
        "overall": overall,
        "mean_margin": (
            margin_sum / total
        ),
        "mean_confidence": (
            confidence_sum / total
        ),
        "mean_disagreement": (
            disagreement_sum / total
        ),
        "target_centroids": target_centroids,
    }


def compute_source_mcd_geometry(model):
    source_sum = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64,
    )

    source_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        SOURCE_CACHE
    ):
        with torch.no_grad():
            z = model.adapter(features)

        for class_id in range(
            NUM_CLASSES
        ):
            mask = labels == class_id

            if mask.any():
                source_sum[class_id] += (
                    z[mask]
                    .double()
                    .sum(dim=0)
                )

                source_count[class_id] += int(
                    mask.sum().item()
                )

    source_centroids = (
        source_sum
        / source_count.clamp_min(1).unsqueeze(1)
    )

    source_scatter_sum = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    source_scatter_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        SOURCE_CACHE
    ):
        with torch.no_grad():
            z = model.adapter(features)

        for class_id in range(
            NUM_CLASSES
        ):
            mask = labels == class_id

            if mask.any():
                x = z[mask].double()

                centroid = (
                    source_centroids[
                        class_id
                    ]
                )

                dist2 = (
                    (x - centroid) ** 2
                ).sum(dim=1)

                source_scatter_sum[
                    class_id
                ] += dist2.sum()

                source_scatter_count[
                    class_id
                ] += dist2.numel()

    source_within = (
        source_scatter_sum
        / source_scatter_count.clamp_min(1)
    )

    return (
        source_centroids,
        source_within,
    )


def strongest_confusions(confusion):
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
                100.0
                * count
                / row_total
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
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    return pairs


def attractor_table(result):
    true_total = result["totals"].float()
    predicted_total = result["predicted"].float()

    true_share = (
        true_total
        / true_total.sum()
    )

    predicted_share = (
        predicted_total
        / predicted_total.sum()
    )

    ratio = (
        predicted_share
        / true_share.clamp_min(1e-9)
    )

    return [
        (
            CLASSES[i],
            float(
                true_share[i].item()
                * 100.0
            ),
            float(
                predicted_share[i].item()
                * 100.0
            ),
            float(
                ratio[i].item()
            ),
        )
        for i in range(NUM_CLASSES)
    ]


def tensor_list(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()

    return x


def main():
    print("=" * 90)
    print("VISDA-2017 MCD RESIDUAL DIAGNOSTICS")
    print("=" * 90)

    print(
        "Target labels are used only for diagnostics."
    )

    print()
    print("Loading source-only probe...")

    source_probe = load_source_probe()

    print(
        "Loading corrected MCD model..."
    )

    mcd_model = load_mcd()

    print()
    print("Analyzing source-only predictions...")

    baseline = analyze_baseline(
        source_probe
    )

    print(
        "Analyzing corrected MCD predictions..."
    )

    mcd = analyze_mcd(
        mcd_model
    )

    print()
    print("=" * 90)
    print("HEADLINE COMPARISON")
    print("=" * 90)

    print(
        f"{'Metric':32s}"
        f"{'Source-only':>18s}"
        f"{'MCD':>18s}"
    )

    print(
        f"{'Mean-class accuracy':32s}"
        f"{baseline['mean_class']:17.2f}%"
        f"{mcd['mean_class']:17.2f}%"
    )

    print(
        f"{'Overall accuracy':32s}"
        f"{baseline['overall']:17.2f}%"
        f"{mcd['overall']:17.2f}%"
    )

    print(
        f"{'Mean confidence':32s}"
        f"{baseline['mean_confidence']:18.4f}"
        f"{mcd['mean_confidence']:18.4f}"
    )

    print(
        f"{'Mean margin':32s}"
        f"{baseline['mean_margin']:18.4f}"
        f"{mcd['mean_margin']:18.4f}"
    )

    print(
        f"{'Mean disagreement':32s}"
        f"{'N/A':>18s}"
        f"{mcd['mean_disagreement']:18.6f}"
    )

    print()
    print("=" * 90)
    print("PER-CLASS COMPARISON")
    print("=" * 90)

    for i, class_name in enumerate(CLASSES):
        old_acc = float(
            baseline["per_class"][i].item()
        )

        new_acc = float(
            mcd["per_class"][i].item()
        )

        delta = new_acc - old_acc

        print(
            f"{class_name:12s} | "
            f"Source {old_acc:6.2f}% | "
            f"MCD {new_acc:6.2f}% | "
            f"Delta {delta:+7.2f} pp"
        )

    print()
    print("=" * 90)
    print("PREDICTION ATTRACTORS")
    print("=" * 90)

    print(
        f"{'Class':12s}"
        f"{'True %':>12s}"
        f"{'Pred %':>12s}"
        f"{'Pred/True':>14s}"
    )

    for (
        class_name,
        true_share,
        pred_share,
        ratio,
    ) in attractor_table(mcd):
        print(
            f"{class_name:12s}"
            f"{true_share:11.2f}%"
            f"{pred_share:11.2f}%"
            f"{ratio:13.2f}x"
        )

    print()
    print("=" * 90)
    print("SOURCE-ONLY STRONGEST CONFUSIONS")
    print("=" * 90)

    for (
        rate,
        count,
        true_name,
        pred_name,
    ) in strongest_confusions(
        baseline["confusion"]
    )[:15]:
        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{rate:6.2f}% | "
            f"{count:6d}"
        )

    print()
    print("=" * 90)
    print("MCD STRONGEST CONFUSIONS")
    print("=" * 90)

    for (
        rate,
        count,
        true_name,
        pred_name,
    ) in strongest_confusions(
        mcd["confusion"]
    )[:15]:
        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{rate:6.2f}% | "
            f"{count:6d}"
        )

    baseline_conf = baseline["confusion"].float()
    mcd_conf = mcd["confusion"].float()

    baseline_rates = (
        baseline_conf
        / baseline_conf.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)
        * 100.0
    )

    mcd_rates = (
        mcd_conf
        / mcd_conf.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)
        * 100.0
    )

    delta_rates = (
        mcd_rates
        - baseline_rates
    )

    delta_pairs = []

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                continue

            delta_pairs.append(
                (
                    float(
                        delta_rates[i, j].item()
                    ),
                    int(
                        mcd_conf[i, j].item()
                    ),
                    CLASSES[i],
                    CLASSES[j],
                )
            )

    print()
    print("=" * 90)
    print("LARGEST CONFUSION INCREASES")
    print("=" * 90)

    for (
        delta,
        count,
        true_name,
        pred_name,
    ) in sorted(
        delta_pairs,
        reverse=True,
    )[:15]:
        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{delta:+7.2f} pp | "
            f"MCD count {count:6d}"
        )

    print()
    print("=" * 90)
    print("LARGEST CONFUSION REDUCTIONS")
    print("=" * 90)

    for (
        delta,
        count,
        true_name,
        pred_name,
    ) in sorted(
        delta_pairs,
        key=lambda x: x[0],
    )[:15]:
        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{delta:+7.2f} pp | "
            f"MCD count {count:6d}"
        )

    print()
    print("=" * 90)
    print("MCD FEATURE GEOMETRY")
    print("=" * 90)

    (
        source_centroids,
        source_within,
    ) = compute_source_mcd_geometry(
        mcd_model
    )

    target_centroids = (
        mcd["target_centroids"]
    )

    centroid_shift = torch.norm(
        target_centroids
        - source_centroids,
        dim=1,
    )

    target_scatter_sum = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    target_scatter_count = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        with torch.no_grad():
            z = mcd_model.adapter(
                features
            )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = labels == class_id

            if mask.any():
                x = z[mask].double()

                centroid = (
                    target_centroids[
                        class_id
                    ]
                )

                dist2 = (
                    (x - centroid) ** 2
                ).sum(dim=1)

                target_scatter_sum[
                    class_id
                ] += dist2.sum()

                target_scatter_count[
                    class_id
                ] += dist2.numel()

    target_within = (
        target_scatter_sum
        / target_scatter_count.clamp_min(1)
    )

    for i, class_name in enumerate(
        CLASSES
    ):
        print(
            f"{class_name:12s} | "
            f"source within "
            f"{source_within[i].item():12.2f} | "
            f"target within "
            f"{target_within[i].item():12.2f} | "
            f"centroid shift "
            f"{centroid_shift[i].item():12.2f}"
        )

    report = {
        "experiment":
            "visda_mcd_residual_diagnostics",
        "seed":
            SEED,
        "target_labels_used_only_for_diagnostics":
            True,
        "baseline": {
            "overall_accuracy":
                baseline["overall"],
            "mean_class_accuracy":
                baseline["mean_class"],
            "mean_confidence":
                baseline["mean_confidence"],
            "mean_margin":
                baseline["mean_margin"],
            "per_class_accuracy":
                tensor_list(
                    baseline["per_class"]
                ),
            "confusion":
                tensor_list(
                    baseline["confusion"]
                ),
            "predicted_counts":
                tensor_list(
                    baseline["predicted"]
                ),
        },
        "mcd": {
            "overall_accuracy":
                mcd["overall"],
            "mean_class_accuracy":
                mcd["mean_class"],
            "mean_confidence":
                mcd["mean_confidence"],
            "mean_margin":
                mcd["mean_margin"],
            "mean_disagreement":
                mcd["mean_disagreement"],
            "per_class_accuracy":
                tensor_list(
                    mcd["per_class"]
                ),
            "confusion":
                tensor_list(
                    mcd["confusion"]
                ),
            "predicted_counts":
                tensor_list(
                    mcd["predicted"]
                ),
            "source_within_class":
                tensor_list(
                    source_within
                ),
            "target_within_class":
                tensor_list(
                    target_within
                ),
            "centroid_shift":
                tensor_list(
                    centroid_shift
                ),
        },
        "class_accuracy_delta":
            tensor_list(
                mcd["per_class"]
                - baseline["per_class"]
            ),
        "confusion_delta_rates":
            tensor_list(
                delta_rates
            ),
        "attractors":
            attractor_table(mcd),
    }

    report_path = (
        OUTPUT_DIR
        / "mcd_residual_diagnostics_seed42.json"
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
    print("DIAGNOSTICS COMPLETE")
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )


if __name__ == "__main__":
    main()