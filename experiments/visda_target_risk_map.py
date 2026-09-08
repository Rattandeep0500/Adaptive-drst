import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
TARGET_CACHE = CACHE_ROOT / "target"

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_target_risk"
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

BATCH_SIZE = 4096
NUM_BINS = 10

RISK_NAMES = [
    "confidence",
    "margin",
    "disagreement",
    "entropy",
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
        )


def load_model():
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


def load_target_chunks():
    files = sorted(
        TARGET_CACHE.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No target chunks found in {TARGET_CACHE}"
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


def collect_outputs(model):
    all_labels = []
    all_predictions = []
    all_confidence = []
    all_margin = []
    all_disagreement = []
    all_entropy = []

    total = 0

    for features, labels in load_target_chunks():
        for start in range(
            0,
            len(features),
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                len(features),
            )

            x = features[start:end]
            y = labels[start:end]

            with torch.no_grad():
                logits1, logits2 = model(x)

                p1 = F.softmax(
                    logits1,
                    dim=1,
                )

                p2 = F.softmax(
                    logits2,
                    dim=1,
                )

                p = (
                    p1 + p2
                ) / 2.0

                top = torch.topk(
                    p,
                    k=2,
                    dim=1,
                )

                confidence = top.values[:, 0]

                margin = (
                    top.values[:, 0]
                    - top.values[:, 1]
                )

                predictions = (
                    top.indices[:, 0]
                )

                disagreement = (
                    torch.abs(
                        p1 - p2
                    ).mean(dim=1)
                )

                entropy = -(
                    p
                    * torch.log(
                        p.clamp_min(1e-12)
                    )
                ).sum(dim=1)

            all_labels.append(
                y
            )

            all_predictions.append(
                predictions
            )

            all_confidence.append(
                confidence
            )

            all_margin.append(
                margin
            )

            all_disagreement.append(
                disagreement
            )

            all_entropy.append(
                entropy
            )

            total += y.size(0)

    return {
        "labels": torch.cat(
            all_labels
        ),
        "predictions": torch.cat(
            all_predictions
        ),
        "confidence": torch.cat(
            all_confidence
        ),
        "margin": torch.cat(
            all_margin
        ),
        "disagreement": torch.cat(
            all_disagreement
        ),
        "entropy": torch.cat(
            all_entropy
        ),
        "total": total,
    }


def error_mask(data):
    return (
        data["predictions"]
        != data["labels"]
    )


def quantile_edges(values):
    q = torch.linspace(
        0.0,
        1.0,
        NUM_BINS + 1,
    )

    edges = torch.quantile(
        values,
        q,
    )

    edges = torch.unique(edges)

    if len(edges) < 2:
        low = values.min().item()
        high = values.max().item()

        if low == high:
            high = low + 1e-6

        edges = torch.tensor(
            [low, high],
            dtype=torch.float32,
        )

    return edges


def assign_bins(values, edges):
    bins = torch.bucketize(
        values,
        edges[1:-1],
        right=False,
    )

    return bins.clamp(
        0,
        len(edges) - 2,
    )


def analyze_global_risk(
    values,
    errors,
):
    edges = quantile_edges(
        values
    )

    bins = assign_bins(
        values,
        edges,
    )

    rows = []

    for bin_id in range(
        len(edges) - 1
    ):
        mask = bins == bin_id

        count = int(
            mask.sum().item()
        )

        if count == 0:
            continue

        error_rate = (
            100.0
            * errors[mask]
            .float()
            .mean()
            .item()
        )

        mean_value = (
            values[mask]
            .mean()
            .item()
        )

        rows.append(
            {
                "bin": bin_id,
                "count": count,
                "error_rate_percent":
                    error_rate,
                "mean_value":
                    mean_value,
                "low":
                    edges[bin_id].item(),
                "high":
                    edges[bin_id + 1].item(),
            }
        )

    return rows


def correlation_with_error(
    values,
    errors,
):
    x = values.double()

    y = errors.double()

    x_centered = (
        x - x.mean()
    )

    y_centered = (
        y - y.mean()
    )

    numerator = (
        x_centered
        * y_centered
    ).sum()

    denominator = (
        torch.sqrt(
            (
                x_centered
                .pow(2)
            ).sum()
        )
        * torch.sqrt(
            (
                y_centered
                .pow(2)
            ).sum()
        )
    )

    if denominator.item() == 0:
        return 0.0

    return (
        numerator / denominator
    ).item()


def analyze_class_risk(
    predictions,
    errors,
    confidence,
    margin,
    disagreement,
    entropy,
):
    rows = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            predictions
            == class_id
        )

        count = int(
            mask.sum().item()
        )

        if count == 0:
            rows.append(
                {
                    "class":
                        CLASSES[class_id],
                    "count":
                        0,
                    "error_rate_percent":
                        0.0,
                    "confidence":
                        0.0,
                    "margin":
                        0.0,
                    "disagreement":
                        0.0,
                    "entropy":
                        0.0,
                }
            )

            continue

        rows.append(
            {
                "class":
                    CLASSES[class_id],
                "count":
                    count,
                "error_rate_percent":
                    100.0
                    * errors[mask]
                    .float()
                    .mean()
                    .item(),
                "confidence":
                    confidence[mask]
                    .mean()
                    .item(),
                "margin":
                    margin[mask]
                    .mean()
                    .item(),
                "disagreement":
                    disagreement[mask]
                    .mean()
                    .item(),
                "entropy":
                    entropy[mask]
                    .mean()
                    .item(),
            }
        )

    return rows


def analyze_class_risk_bins(
    data,
    risk_name,
):
    values = data[risk_name]

    errors = error_mask(
        data
    )

    predictions = data[
        "predictions"
    ]

    results = []

    for class_id in range(
        NUM_CLASSES
    ):
        class_mask = (
            predictions
            == class_id
        )

        if not class_mask.any():
            continue

        class_values = values[
            class_mask
        ]

        class_errors = errors[
            class_mask
        ]

        if class_values.numel() < NUM_BINS:
            continue

        edges = quantile_edges(
            class_values
        )

        bins = assign_bins(
            class_values,
            edges,
        )

        for bin_id in range(
            len(edges) - 1
        ):
            mask = bins == bin_id

            count = int(
                mask.sum().item()
            )

            if count == 0:
                continue

            results.append(
                {
                    "class":
                        CLASSES[class_id],
                    "bin":
                        bin_id,
                    "count":
                        count,
                    "error_rate_percent":
                        100.0
                        * class_errors[
                            mask
                        ]
                        .float()
                        .mean()
                        .item(),
                    "mean_risk_value":
                        class_values[
                            mask
                        ]
                        .mean()
                        .item(),
                }
            )

    return results


def compute_safe_pool_statistics(
    data,
):
    errors = error_mask(
        data
    )

    confidence = data[
        "confidence"
    ]

    margin = data[
        "margin"
    ]

    disagreement = data[
        "disagreement"
    ]

    entropy = data[
        "entropy"
    ]

    thresholds = {
        "confidence_0.80":
            confidence >= 0.80,
        "confidence_0.90":
            confidence >= 0.90,
        "confidence_0.95":
            confidence >= 0.95,
        "margin_0.30":
            margin >= 0.30,
        "margin_0.50":
            margin >= 0.50,
        "margin_0.70":
            margin >= 0.70,
        "disagreement_0.02":
            disagreement <= 0.02,
        "disagreement_0.05":
            disagreement <= 0.05,
        "disagreement_0.10":
            disagreement <= 0.10,
        "entropy_0.30":
            entropy <= 0.30,
        "entropy_0.50":
            entropy <= 0.50,
        "entropy_0.70":
            entropy <= 0.70,
    }

    rows = []

    for name, mask in thresholds.items():
        count = int(
            mask.sum().item()
        )

        if count == 0:
            error_rate = 0.0
        else:
            error_rate = (
                100.0
                * errors[mask]
                .float()
                .mean()
                .item()
            )

        rows.append(
            {
                "rule":
                    name,
                "selected":
                    count,
                "coverage_percent":
                    100.0
                    * count
                    / len(errors),
                "error_rate_percent":
                    error_rate,
                "precision_percent":
                    100.0
                    - error_rate,
            }
        )

    return rows


def print_global_risk(
    name,
    rows,
):
    print()
    print("=" * 90)
    print(
        f"{name.upper()} RISK BINS"
    )
    print("=" * 90)

    print(
        f"{'Bin':8s}"
        f"{'Count':12s}"
        f"{'Mean':14s}"
        f"{'Error %':14s}"
        f"{'Range':26s}"
    )

    for row in rows:
        print(
            f"{row['bin']:<8d}"
            f"{row['count']:<12d}"
            f"{row['mean_value']:12.6f}  "
            f"{row['error_rate_percent']:11.2f}%  "
            f"[{row['low']:.6f}, "
            f"{row['high']:.6f}]"
        )


def main():
    print("=" * 90)
    print("VISDA-2017 TARGET-ONLY RISK MAP")
    print("=" * 90)

    print(
        "Risk features use model outputs only."
    )

    print(
        "Target labels are used only afterward "
        "to measure predictive value."
    )

    model = load_model()

    print()
    print(
        "Collecting target model outputs..."
    )

    data = collect_outputs(
        model
    )

    errors = error_mask(
        data
    )

    print()
    print("=" * 90)
    print("BASELINE")
    print("=" * 90)

    total = data["total"]

    accuracy = (
        100.0
        * (~errors)
        .float()
        .mean()
        .item()
    )

    print(
        f"Samples: {total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print(
        f"Error rate: "
        f"{100.0 - accuracy:.2f}%"
    )

    print()
    print("=" * 90)
    print("GLOBAL RISK SIGNALS")
    print("=" * 90)

    correlations = {}

    for risk_name in RISK_NAMES:
        corr = correlation_with_error(
            data[risk_name],
            errors,
        )

        correlations[
            risk_name
        ] = corr

        print(
            f"{risk_name:16s}: "
            f"{corr:+.6f}"
        )

    global_bins = {}

    for risk_name in RISK_NAMES:
        rows = analyze_global_risk(
            data[risk_name],
            errors,
        )

        global_bins[
            risk_name
        ] = rows

        print_global_risk(
            risk_name,
            rows,
        )

    print()
    print("=" * 90)
    print("PREDICTED-CLASS RISK")
    print("=" * 90)

    class_risk = analyze_class_risk(
        data["predictions"],
        errors,
        data["confidence"],
        data["margin"],
        data["disagreement"],
        data["entropy"],
    )

    print(
        f"{'Class':12s}"
        f"{'Count':10s}"
        f"{'Error %':12s}"
        f"{'Conf':12s}"
        f"{'Margin':12s}"
        f"{'Disc':12s}"
        f"{'Entropy':12s}"
    )

    for row in class_risk:
        print(
            f"{row['class']:12s}"
            f"{row['count']:9d}"
            f"{row['error_rate_percent']:11.2f}%"
            f"{row['confidence']:11.4f}"
            f"{row['margin']:11.4f}"
            f"{row['disagreement']:11.4f}"
            f"{row['entropy']:11.4f}"
        )

    print()
    print("=" * 90)
    print("TARGET POOL QUALITY")
    print("=" * 90)

    pool_stats = (
        compute_safe_pool_statistics(
            data
        )
    )

    print(
        f"{'Rule':24s}"
        f"{'Selected':12s}"
        f"{'Coverage':14s}"
        f"{'Error %':14s}"
        f"{'Precision':14s}"
    )

    for row in pool_stats:
        print(
            f"{row['rule']:24s}"
            f"{row['selected']:11d}"
            f"{row['coverage_percent']:12.2f}%"
            f"{row['error_rate_percent']:12.2f}%"
            f"{row['precision_percent']:12.2f}%"
        )

    print()
    print("=" * 90)
    print("CLASS-CONDITIONAL RISK BINS")
    print("=" * 90)

    class_bin_results = {}

    for risk_name in RISK_NAMES:
        rows = analyze_class_risk_bins(
            data,
            risk_name,
        )

        class_bin_results[
            risk_name
        ] = rows

        print()
        print(
            f"RISK = {risk_name}"
        )

        for row in rows:
            if row["bin"] in (
                0,
                NUM_BINS - 1,
            ):
                print(
                    f"{row['class']:12s} | "
                    f"bin {row['bin']:02d} | "
                    f"n {row['count']:6d} | "
                    f"risk {row['mean_risk_value']:.6f} | "
                    f"error {row['error_rate_percent']:.2f}%"
                )

    class_rank = sorted(
        class_risk,
        key=lambda x:
            x["error_rate_percent"],
        reverse=True,
    )

    print()
    print("=" * 90)
    print("HIGHEST-RISK PREDICTED CLASSES")
    print("=" * 90)

    for row in class_rank:
        print(
            f"{row['class']:12s}: "
            f"{row['error_rate_percent']:.2f}% error "
            f"| n={row['count']}"
        )

    risk_rank = sorted(
        correlations.items(),
        key=lambda x:
            abs(x[1]),
        reverse=True,
    )

    print()
    print("=" * 90)
    print("RISK SIGNAL RANKING")
    print("=" * 90)

    for rank, (
        name,
        corr,
    ) in enumerate(
        risk_rank,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{name}: "
            f"{corr:+.6f}"
        )

    report = {
        "experiment":
            "visda_target_only_risk_map",
        "seed":
            42,
        "target_labels_used_only_for_evaluation":
            True,
        "total_samples":
            total,
        "accuracy":
            accuracy,
        "correlations_with_error":
            correlations,
        "global_bins":
            global_bins,
        "predicted_class_risk":
            class_risk,
        "target_pool_quality":
            pool_stats,
        "class_conditional_bins":
            class_bin_results,
        "risk_ranking":
            [
                {
                    "rank":
                        i + 1,
                    "signal":
                        name,
                    "correlation":
                        corr,
                }
                for i, (
                    name,
                    corr,
                ) in enumerate(
                    risk_rank
                )
            ],
    }

    report_path = (
        OUTPUT_DIR
        / "target_risk_map_seed42.json"
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
    print("RISK MAP COMPLETE")
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )


if __name__ == "__main__":
    main()