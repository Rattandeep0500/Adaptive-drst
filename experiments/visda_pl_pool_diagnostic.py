import json
from pathlib import Path

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

CURRICULUM_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_ema_curriculum/"
    "mcd_ema_curriculum_seed42.pt"
)

CURRICULUM_HISTORY = Path(
    "checkpoints/visda_cached_mcd_ema_curriculum/"
    "mcd_ema_curriculum_seed42.json"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_pl_pool_diagnostic"
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
        )


def load_cache(cache_dir):
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No chunks found in {cache_dir}"
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


def load_curriculum():
    checkpoint = torch.load(
        CURRICULUM_CHECKPOINT,
        map_location="cpu",
    )

    model = MCDModel()

    model.load_state_dict(
        checkpoint["teacher_state_dict"]
    )

    model.eval()

    return model


def load_history():
    if not CURRICULUM_HISTORY.exists():
        return None

    with open(
        CURRICULUM_HISTORY,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def collect_model_outputs(
    model,
    features,
):
    predictions = []
    confidence = []
    disagreement = []
    entropy = []

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

            conf, pred = p.max(
                dim=1
            )

            disc = (
                torch.abs(
                    p1 - p2
                ).mean(dim=1)
            )

            ent = -(
                p
                * torch.log(
                    p.clamp_min(1e-12)
                )
            ).sum(dim=1)

            predictions.append(
                pred
            )

            confidence.append(
                conf
            )

            disagreement.append(
                disc
            )

            entropy.append(
                ent
            )

    return {
        "prediction":
            torch.cat(predictions),
        "confidence":
            torch.cat(confidence),
        "disagreement":
            torch.cat(disagreement),
        "entropy":
            torch.cat(entropy),
    }


def threshold_for_epoch(
    epoch,
    warmup,
    total_epochs,
    start,
    end,
):
    if epoch <= warmup:
        return float("inf")

    span = max(
        total_epochs - warmup,
        1,
    )

    progress = (
        epoch - warmup
    ) / span

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return (
        start
        + (
            end - start
        )
        * progress
    )


def selected_mask(
    confidence,
    threshold,
    fraction,
):
    n = len(confidence)

    if not torch.isfinite(
        torch.tensor(threshold)
    ):
        return torch.zeros(
            n,
            dtype=torch.bool,
        )

    mask = (
        confidence
        >= threshold
    )

    indices = torch.nonzero(
        mask,
        as_tuple=False,
    ).flatten()

    max_count = max(
        1,
        int(
            n
            * fraction
        ),
    )

    if len(indices) <= max_count:
        return mask

    selected_conf = confidence[
        indices
    ]

    keep = torch.topk(
        selected_conf,
        k=max_count,
        largest=True,
    ).indices

    chosen = indices[
        keep
    ]

    output = torch.zeros(
        n,
        dtype=torch.bool,
    )

    output[
        chosen
    ] = True

    return output


def evaluate_pool(
    prediction,
    confidence,
    disagreement,
    entropy,
    labels,
    mask,
):
    selected = int(
        mask.sum().item()
    )

    if selected == 0:
        return {
            "selected": 0,
            "coverage_percent": 0.0,
            "pseudo_accuracy_percent": 0.0,
            "mean_confidence": 0.0,
            "mean_disagreement": 0.0,
            "mean_entropy": 0.0,
            "predicted_counts": [
                0
                for _ in range(
                    NUM_CLASSES
                )
            ],
            "true_counts": [
                0
                for _ in range(
                    NUM_CLASSES
                )
            ],
        }

    pseudo_accuracy = (
        100.0
        * (
            prediction[mask]
            == labels[mask]
        )
        .float()
        .mean()
        .item()
    )

    pred_counts = torch.bincount(
        prediction[mask],
        minlength=NUM_CLASSES,
    )

    true_counts = torch.bincount(
        labels[mask],
        minlength=NUM_CLASSES,
    )

    return {
        "selected": selected,
        "coverage_percent":
            100.0
            * selected
            / len(labels),
        "pseudo_accuracy_percent":
            pseudo_accuracy,
        "mean_confidence":
            confidence[mask]
            .mean()
            .item(),
        "mean_disagreement":
            disagreement[mask]
            .mean()
            .item(),
        "mean_entropy":
            entropy[mask]
            .mean()
            .item(),
        "predicted_counts":
            pred_counts.tolist(),
        "true_counts":
            true_counts.tolist(),
    }


def print_distribution(
    counts,
    total,
):
    for i, name in enumerate(
        CLASSES
    ):
        percentage = (
            100.0
            * counts[i]
            / max(total, 1)
        )

        print(
            f"{name:12s}: "
            f"{counts[i]:6d} "
            f"({percentage:6.2f}%)"
        )


def compare_distributions(
    predicted,
    true,
):
    rows = []

    pred_total = sum(
        predicted
    )

    true_total = sum(
        true
    )

    for i, name in enumerate(
        CLASSES
    ):
        pred_pct = (
            100.0
            * predicted[i]
            / max(
                pred_total,
                1,
            )
        )

        true_pct = (
            100.0
            * true[i]
            / max(
                true_total,
                1,
            )
        )

        rows.append(
            {
                "class":
                    name,
                "predicted_percent":
                    pred_pct,
                "true_percent":
                    true_pct,
                "difference_pp":
                    pred_pct - true_pct,
            }
        )

    return rows


def main():
    print("=" * 90)
    print("VISDA-2017 PSEUDO-LABEL POOL AUTOPSY")
    print("=" * 90)

    print(
        "Target labels are used only to evaluate pseudo-label quality."
    )

    print()
    print(
        "Loading target cache..."
    )

    features, labels = load_cache(
        TARGET_CACHE
    )

    print(
        f"Target samples: {len(features)}"
    )

    print()
    print(
        "Loading corrected MCD..."
    )

    mcd = load_mcd()

    print(
        "Loading curriculum teacher..."
    )

    teacher = load_curriculum()

    print()
    print(
        "Collecting corrected-MCD outputs..."
    )

    mcd_outputs = collect_model_outputs(
        mcd,
        features,
    )

    print(
        "Collecting final curriculum-teacher outputs..."
    )

    teacher_outputs = collect_model_outputs(
        teacher,
        features,
    )

    history = load_history()

    print()
    print("=" * 90)
    print("FINAL CURRICULUM TEACHER")
    print("=" * 90)

    teacher_accuracy = (
        100.0
        * (
            teacher_outputs["prediction"]
            == labels
        )
        .float()
        .mean()
        .item()
    )

    print(
        f"Target accuracy: "
        f"{teacher_accuracy:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{teacher_outputs['confidence'].mean().item():.6f}"
    )

    print(
        f"Mean disagreement: "
        f"{teacher_outputs['disagreement'].mean().item():.6f}"
    )

    print(
        f"Mean entropy: "
        f"{teacher_outputs['entropy'].mean().item():.6f}"
    )

    print()
    print("=" * 90)
    print("FINAL TEACHER PREDICTION DISTRIBUTION")
    print("=" * 90)

    teacher_pred_counts = torch.bincount(
        teacher_outputs["prediction"],
        minlength=NUM_CLASSES,
    ).tolist()

    teacher_true_counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES,
    ).tolist()

    print("Predicted:")
    print_distribution(
        teacher_pred_counts,
        len(labels),
    )

    print()
    print("True target:")
    print_distribution(
        teacher_true_counts,
        len(labels),
    )

    print()
    print("=" * 90)
    print("PREDICTED VS TRUE DISTRIBUTION")
    print("=" * 90)

    final_distribution_comparison = (
        compare_distributions(
            teacher_pred_counts,
            teacher_true_counts,
        )
    )

    for row in final_distribution_comparison:
        print(
            f"{row['class']:12s} | "
            f"pred "
            f"{row['predicted_percent']:6.2f}% | "
            f"true "
            f"{row['true_percent']:6.2f}% | "
            f"delta "
            f"{row['difference_pp']:+7.2f} pp"
        )

    print()
    print("=" * 90)
    print("FINAL CURRICULUM POOL")
    print("=" * 90)

    final_mask = selected_mask(
        teacher_outputs["confidence"],
        0.90,
        0.40,
    )

    final_pool = evaluate_pool(
        teacher_outputs["prediction"],
        teacher_outputs["confidence"],
        teacher_outputs["disagreement"],
        teacher_outputs["entropy"],
        labels,
        final_mask,
    )

    print(
        f"Selected: "
        f"{final_pool['selected']}"
    )

    print(
        f"Coverage: "
        f"{final_pool['coverage_percent']:.2f}%"
    )

    print(
        f"Actual pseudo-label accuracy: "
        f"{final_pool['pseudo_accuracy_percent']:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{final_pool['mean_confidence']:.6f}"
    )

    print()
    print("Predicted class distribution:")
    print_distribution(
        final_pool["predicted_counts"],
        final_pool["selected"],
    )

    print()
    print("True class distribution:")
    print_distribution(
        final_pool["true_counts"],
        final_pool["selected"],
    )

    print()
    print("=" * 90)
    print("TARGET POOL BY THRESHOLD")
    print("=" * 90)

    thresholds = [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.98,
        0.99,
    ]

    threshold_results = []

    for threshold in thresholds:
        mask = (
            teacher_outputs["confidence"]
            >= threshold
        )

        result = evaluate_pool(
            teacher_outputs["prediction"],
            teacher_outputs["confidence"],
            teacher_outputs["disagreement"],
            teacher_outputs["entropy"],
            labels,
            mask,
        )

        threshold_results.append(
            {
                "threshold":
                    threshold,
                **result,
            }
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"coverage "
            f"{result['coverage_percent']:.2f}% | "
            f"pseudo-accuracy "
            f"{result['pseudo_accuracy_percent']:.2f}% | "
            f"mean conf "
            f"{result['mean_confidence']:.4f}"
        )

    print()
    print("=" * 90)
    print("CURRICULUM HISTORY")
    print("=" * 90)

    history_rows = []

    if history is not None:
        for record in history.get(
            "history",
            [],
        ):
            epoch = record[
                "epoch"
            ]

            threshold = (
                record["threshold"]
            )

            coverage = (
                record[
                    "pseudo_coverage_percent"
                ]
            )

            confidence = (
                record[
                    "mean_selected_confidence"
                ]
            )

            print(
                f"Epoch {epoch:02d} | "
                f"threshold "
                f"{threshold if threshold is not None else 'WARMUP'} | "
                f"coverage "
                f"{coverage:.2f}% | "
                f"selected confidence "
                f"{confidence:.6f}"
            )

            history_rows.append(
                {
                    "epoch":
                        epoch,
                    "threshold":
                        threshold,
                    "coverage":
                        coverage,
                    "selected_confidence":
                        confidence,
                    "student_target_mean_class":
                        record[
                            "student_target_mean_class"
                        ],
                    "teacher_target_mean_class":
                        record[
                            "teacher_target_mean_class"
                        ],
                }
            )

    print()
    print("=" * 90)
    print("MCD VS CURRICULUM TEACHER")
    print("=" * 90)

    mcd_accuracy = (
        100.0
        * (
            mcd_outputs["prediction"]
            == labels
        )
        .float()
        .mean()
        .item()
    )

    teacher_mcd_delta = (
        teacher_accuracy
        - mcd_accuracy
    )

    print(
        f"MCD accuracy: "
        f"{mcd_accuracy:.2f}%"
    )

    print(
        f"Teacher accuracy: "
        f"{teacher_accuracy:.2f}%"
    )

    print(
        f"Teacher delta: "
        f"{teacher_mcd_delta:+.2f} pp"
    )

    mcd_counts = torch.bincount(
        mcd_outputs["prediction"],
        minlength=NUM_CLASSES,
    ).tolist()

    print()
    print(
        "MCD predicted distribution:"
    )

    print_distribution(
        mcd_counts,
        len(labels),
    )

    print()
    print(
        "Teacher predicted distribution:"
    )

    print_distribution(
        teacher_pred_counts,
        len(labels),
    )

    report = {
        "experiment":
            "visda_pl_pool_diagnostic",
        "target_labels_used_only_for_evaluation":
            True,
        "target_samples":
            len(labels),
        "mcd_accuracy":
            mcd_accuracy,
        "teacher_accuracy":
            teacher_accuracy,
        "teacher_minus_mcd":
            teacher_mcd_delta,
        "mcd_prediction_counts":
            mcd_counts,
        "teacher_prediction_counts":
            teacher_pred_counts,
        "target_true_counts":
            teacher_true_counts,
        "final_distribution_comparison":
            final_distribution_comparison,
        "final_pool":
            final_pool,
        "threshold_results":
            threshold_results,
        "history":
            history_rows,
    }

    report_path = (
        OUTPUT_DIR
        / "pl_pool_autopsy_seed42.json"
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
    print("AUTOPSY COMPLETE")
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )


if __name__ == "__main__":
    main()