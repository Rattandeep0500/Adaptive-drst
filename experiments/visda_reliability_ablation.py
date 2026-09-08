import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512
BATCH_SIZE = 4096

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
TARGET_CACHE = CACHE_ROOT / "target"

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_reliability_ablation"
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
                inplace=True,
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


def load_target():
    files = sorted(
        TARGET_CACHE.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No target chunks found in {TARGET_CACHE}"
        )

    features = []
    labels = []

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu",
        )

        features.append(
            payload["features"].float()
        )

        labels.append(
            payload["labels"].long()
        )

    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
    )


def collect_signals(
    model,
    features,
):
    confidences = []
    margins = []
    disagreements = []
    entropies = []
    predictions = []

    with torch.no_grad():
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

            prediction = top.indices[:, 0]

            confidences.append(
                confidence
            )

            margins.append(
                margin
            )

            disagreements.append(
                disagreement
            )

            entropies.append(
                entropy
            )

            predictions.append(
                prediction
            )

    return {
        "confidence": torch.cat(
            confidences
        ),
        "margin": torch.cat(
            margins
        ),
        "disagreement": torch.cat(
            disagreements
        ),
        "entropy": torch.cat(
            entropies
        ),
        "prediction": torch.cat(
            predictions
        ),
    }


def rank_transform(values):
    order = torch.argsort(
        torch.argsort(values)
    )

    denominator = max(
        len(values) - 1,
        1,
    )

    return order.float() / denominator


def zscore(values):
    mean = values.mean()

    std = values.std().clamp_min(
        1e-8
    )

    return (
        values - mean
    ) / std


def binary_auc(scores, labels):
    scores = scores.double()
    labels = labels.long()

    positives = labels == 1
    negatives = labels == 0

    n_pos = int(
        positives.sum().item()
    )

    n_neg = int(
        negatives.sum().item()
    )

    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_scores = scores[positives]
    neg_scores = scores[negatives]

    comparisons = (
        pos_scores.unsqueeze(1)
        > neg_scores.unsqueeze(0)
    )

    ties = (
        pos_scores.unsqueeze(1)
        == neg_scores.unsqueeze(0)
    )

    auc = (
        comparisons.float().sum()
        + 0.5 * ties.float().sum()
    ) / (
        n_pos * n_neg
    )

    return float(
        auc.item()
    )


def quantile_threshold(
    values,
    fraction,
):
    return torch.quantile(
        values,
        1.0 - fraction,
    )


def precision_at_coverage(
    score,
    correct,
    coverage,
):
    n = len(score)

    k = max(
        1,
        int(
            round(
                n * coverage
            )
        ),
    )

    order = torch.argsort(
        score,
        descending=True,
    )

    selected = order[:k]

    precision = (
        correct[selected]
        .float()
        .mean()
        .item()
    )

    return {
        "coverage_percent":
            100.0
            * k
            / n,
        "selected":
            k,
        "precision_percent":
            100.0
            * precision,
    }


def build_scores(signals):
    confidence = signals[
        "confidence"
    ]

    margin = signals[
        "margin"
    ]

    disagreement = signals[
        "disagreement"
    ]

    entropy = signals[
        "entropy"
    ]

    confidence_rank = rank_transform(
        confidence
    )

    margin_rank = rank_transform(
        margin
    )

    disagreement_good_rank = (
        1.0
        - rank_transform(
            disagreement
        )
    )

    entropy_good_rank = (
        1.0
        - rank_transform(
            entropy
        )
    )

    confidence_margin = (
        confidence_rank
        + margin_rank
    ) / 2.0

    confidence_class = (
        confidence_rank.clone()
    )

    combined_equal = (
        confidence_rank
        + margin_rank
        + disagreement_good_rank
        + entropy_good_rank
    ) / 4.0

    combined_weighted = (
        0.40 * confidence_rank
        + 0.20 * margin_rank
        + 0.20 * disagreement_good_rank
        + 0.20 * entropy_good_rank
    )

    combined_z = (
        zscore(confidence)
        + zscore(margin)
        - zscore(disagreement)
        - zscore(entropy)
    )

    return {
        "confidence":
            confidence_rank,
        "confidence_margin":
            confidence_margin,
        "combined_equal":
            combined_equal,
        "combined_weighted":
            combined_weighted,
        "combined_z":
            combined_z,
    }


def build_class_conditional_scores(
    signals
):
    predictions = signals[
        "prediction"
    ]

    confidence = signals[
        "confidence"
    ]

    margin = signals[
        "margin"
    ]

    disagreement = signals[
        "disagreement"
    ]

    entropy = signals[
        "entropy"
    ]

    n = len(predictions)

    scores = {
        "class_conditioned_confidence":
            torch.zeros(n),
        "class_conditioned_combined":
            torch.zeros(n),
    }

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            predictions
            == class_id
        )

        if not mask.any():
            continue

        c = confidence[mask]
        m = margin[mask]
        d = disagreement[mask]
        e = entropy[mask]

        c_rank = rank_transform(c)
        m_rank = rank_transform(m)

        d_rank = (
            1.0
            - rank_transform(d)
        )

        e_rank = (
            1.0
            - rank_transform(e)
        )

        scores[
            "class_conditioned_confidence"
        ][mask] = c_rank

        scores[
            "class_conditioned_combined"
        ][mask] = (
            0.40 * c_rank
            + 0.20 * m_rank
            + 0.20 * d_rank
            + 0.20 * e_rank
        )

    return scores


def evaluate_score_family(
    name,
    score,
    correct,
):
    error = ~correct

    auc = binary_auc(
        score,
        correct.long(),
    )

    print()
    print(
        f"{name}"
    )

    print(
        f"AUC for correctness: "
        f"{auc:.6f}"
    )

    rows = []

    for coverage in (
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
    ):
        result = precision_at_coverage(
            score,
            correct,
            coverage,
        )

        rows.append(
            {
                "coverage_percent":
                    result[
                        "coverage_percent"
                    ],
                "selected":
                    result["selected"],
                "precision_percent":
                    result["precision_percent"],
            }
        )

        print(
            f"Coverage "
            f"{result['coverage_percent']:6.2f}% "
            f"| "
            f"Precision "
            f"{result['precision_percent']:6.2f}%"
        )

    return {
        "auc":
            auc,
        "coverage_results":
            rows,
    }


def evaluate_class_conditional(
    score,
    predictions,
    correct,
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
            continue

        class_score = score[
            mask
        ]

        class_correct = correct[
            mask
        ]

        auc = binary_auc(
            class_score,
            class_correct.long(),
        )

        top_30 = precision_at_coverage(
            class_score,
            class_correct,
            0.30,
        )

        top_50 = precision_at_coverage(
            class_score,
            class_correct,
            0.50,
        )

        rows.append(
            {
                "class":
                    CLASSES[class_id],
                "count":
                    count,
                "auc":
                    auc,
                "top_30_precision":
                    top_30[
                        "precision_percent"
                    ],
                "top_50_precision":
                    top_50[
                        "precision_percent"
                    ],
            }
        )

    return rows


def main():
    print("=" * 90)
    print("VISDA-2017 RELIABILITY ABLATION")
    print("=" * 90)

    print(
        "Scores use model outputs only."
    )

    print(
        "Target labels are used only to evaluate "
        "whether the scores predict correctness."
    )

    model = load_model()

    features, labels = load_target()

    print(
        f"Target samples: {len(features)}"
    )

    signals = collect_signals(
        model,
        features,
    )

    correct = (
        signals["prediction"]
        == labels
    )

    print()
    print("=" * 90)
    print("BASELINE")
    print("=" * 90)

    print(
        f"Accuracy: "
        f"{100.0 * correct.float().mean().item():.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{signals['confidence'].mean().item():.6f}"
    )

    print(
        f"Mean margin: "
        f"{signals['margin'].mean().item():.6f}"
    )

    print(
        f"Mean disagreement: "
        f"{signals['disagreement'].mean().item():.6f}"
    )

    print(
        f"Mean entropy: "
        f"{signals['entropy'].mean().item():.6f}"
    )

    scores = build_scores(
        signals
    )

    class_scores = (
        build_class_conditional_scores(
            signals
        )
    )

    results = {}

    for name, score in scores.items():
        results[name] = (
            evaluate_score_family(
                name,
                score,
                correct,
            )
        )

    for name, score in class_scores.items():
        results[name] = (
            evaluate_score_family(
                name,
                score,
                correct,
            )
        )

    print()
    print("=" * 90)
    print("CLASS-CONDITIONAL PERFORMANCE")
    print("=" * 90)

    class_results = {}

    for name, score in class_scores.items():
        rows = evaluate_class_conditional(
            score,
            signals["prediction"],
            correct,
        )

        class_results[name] = rows

        print()
        print(
            name
        )

        print(
            f"{'Class':12s}"
            f"{'Count':10s}"
            f"{'AUC':12s}"
            f"{'Top30 %':12s}"
            f"{'Top50 %':12s}"
        )

        for row in rows:
            print(
                f"{row['class']:12s}"
                f"{row['count']:9d}"
                f"{row['auc']:11.4f}"
                f"{row['top_30_precision']:11.2f}%"
                f"{row['top_50_precision']:11.2f}%"
            )

    print()
    print("=" * 90)
    print("BEST SIGNAL COMPARISON")
    print("=" * 90)

    ranking = sorted(
        [
            (
                name,
                values["auc"],
            )
            for name, values
            in results.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    for rank, (
        name,
        auc,
    ) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{name:30s} "
            f"AUC={auc:.6f}"
        )

    print()
    print("=" * 90)
    print("INCREMENTAL VALUE")
    print("=" * 90)

    confidence_auc = results[
        "confidence"
    ]["auc"]

    for name, values in ranking:
        gain = (
            values
            - confidence_auc
        )

        print(
            f"{name:30s} "
            f"gain_vs_confidence="
            f"{gain:+.6f}"
        )

    report = {
        "experiment":
            "visda_reliability_ablation",
        "target_labels_used_only_for_evaluation":
            True,
        "total_samples":
            len(features),
        "accuracy":
            float(
                correct.float()
                .mean()
                .item()
            ),
        "signals": {
            name:
                float(
                    signals[name]
                    .mean()
                    .item()
                )
            for name in (
                "confidence",
                "margin",
                "disagreement",
                "entropy",
            )
        },
        "results":
            results,
        "class_conditional_results":
            class_results,
        "ranking":
            [
                {
                    "rank":
                        index + 1,
                    "signal":
                        name,
                    "auc":
                        auc,
                }
                for index, (
                    name,
                    auc,
                ) in enumerate(
                    ranking
                )
            ],
        "incremental_gain_vs_confidence":
            {
                name:
                    values
                    - confidence_auc
                for name, values in ranking
            },
    }

    report_path = (
        OUTPUT_DIR
        / "reliability_ablation_seed42.json"
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
    print("RELIABILITY ABLATION COMPLETE")
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )


if __name__ == "__main__":
    main()