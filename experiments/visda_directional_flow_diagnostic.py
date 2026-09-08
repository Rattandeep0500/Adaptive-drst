import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512
BATCH_SIZE = 4096

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
TARGET_CACHE = CACHE_ROOT / "target"

SOURCE_PROBE = Path(
    "checkpoints/visda_cached_source_only/linear_probe_seed42.pt"
)

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_directional_flow"
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
            nn.BatchNorm1d(HIDDEN_DIM),
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
            f"No cache chunks found in {cache_dir}"
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


def collect_predictions(model, mcd_mode):
    rows = []

    for features, labels in load_chunks(
        TARGET_CACHE
    ):
        with torch.no_grad():
            if not mcd_mode:
                logits = model(features)
                probabilities = F.softmax(
                    logits,
                    dim=1,
                )
                confidence, predictions = probabilities.max(
                    dim=1
                )
                second_values = torch.topk(
                    probabilities,
                    k=2,
                    dim=1,
                ).values

                margins = (
                    second_values[:, 0]
                    - second_values[:, 1]
                )

                disagreement = torch.zeros_like(
                    confidence
                )
            else:
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

                confidence, predictions = probabilities.max(
                    dim=1
                )

                second_values = torch.topk(
                    probabilities,
                    k=2,
                    dim=1,
                ).values

                margins = (
                    second_values[:, 0]
                    - second_values[:, 1]
                )

                disagreement = (
                    torch.abs(
                        p1 - p2
                    ).mean(dim=1)
                )

            rows.append(
                {
                    "labels": labels,
                    "predictions": predictions,
                    "confidence": confidence,
                    "margin": margins,
                    "disagreement": disagreement,
                }
            )

    labels = torch.cat(
        [r["labels"] for r in rows]
    )

    predictions = torch.cat(
        [r["predictions"] for r in rows]
    )

    confidence = torch.cat(
        [r["confidence"] for r in rows]
    )

    margins = torch.cat(
        [r["margin"] for r in rows]
    )

    disagreement = torch.cat(
        [r["disagreement"] for r in rows]
    )

    return {
        "labels": labels,
        "predictions": predictions,
        "confidence": confidence,
        "margin": margins,
        "disagreement": disagreement,
    }


def build_flow_matrix(
    labels_before,
    predictions_before,
    labels_after,
    predictions_after,
):
    before = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
    )

    after = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
    )

    for i in range(NUM_CLASSES):
        mask_before = (
            labels_before == i
        )

        mask_after = (
            labels_after == i
        )

        before[i] = torch.bincount(
            predictions_before[
                mask_before
            ],
            minlength=NUM_CLASSES,
        )

        after[i] = torch.bincount(
            predictions_after[
                mask_after
            ],
            minlength=NUM_CLASSES,
        )

    return before, after


def normalized_rows(matrix):
    totals = matrix.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1)

    return (
        matrix.float()
        / totals.float()
        * 100.0
    )


def strongest_changes(
    before_rates,
    after_rates,
):
    rows = []

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                continue

            delta = (
                after_rates[i, j]
                - before_rates[i, j]
            ).item()

            rows.append(
                (
                    delta,
                    int(
                        after_rates[i, j].item()
                    ),
                    CLASSES[i],
                    CLASSES[j],
                )
            )

    return sorted(
        rows,
        key=lambda x: x[0],
        reverse=True,
    )


def attractor_scores(predictions, labels):
    true_counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES,
    ).float()

    pred_counts = torch.bincount(
        predictions,
        minlength=NUM_CLASSES,
    ).float()

    true_share = (
        true_counts
        / true_counts.sum()
    )

    pred_share = (
        pred_counts
        / pred_counts.sum()
    )

    ratio = (
        pred_share
        / true_share.clamp_min(1e-9)
    )

    return (
        true_share,
        pred_share,
        ratio,
    )


def conditional_margin_stats(
    labels,
    predictions,
    margins,
    disagreement,
):
    results = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = labels == class_id

        correct = mask & (
            predictions == class_id
        )

        wrong = mask & (
            predictions != class_id
        )

        result = {
            "class": CLASSES[class_id],
            "correct_count": int(
                correct.sum().item()
            ),
            "wrong_count": int(
                wrong.sum().item()
            ),
            "correct_margin": float(
                margins[correct].mean().item()
            )
            if correct.any()
            else 0.0,
            "wrong_margin": float(
                margins[wrong].mean().item()
            )
            if wrong.any()
            else 0.0,
            "correct_disagreement": float(
                disagreement[correct].mean().item()
            )
            if correct.any()
            else 0.0,
            "wrong_disagreement": float(
                disagreement[wrong].mean().item()
            )
            if wrong.any()
            else 0.0,
        }

        results.append(result)

    return results


def main():
    print("=" * 90)
    print("VISDA-2017 DIRECTIONAL FLOW DIAGNOSTIC")
    print("=" * 90)

    print(
        "Target labels are used only for diagnostics."
    )

    print()
    print(
        "Loading source-only model..."
    )

    source_model = load_source_probe()

    print(
        "Loading corrected MCD model..."
    )

    mcd_model = load_mcd()

    print()
    print(
        "Collecting source-only target predictions..."
    )

    source_result = collect_predictions(
        source_model,
        False,
    )

    print(
        "Collecting MCD target predictions..."
    )

    mcd_result = collect_predictions(
        mcd_model,
        True,
    )

    labels = source_result["labels"]

    print()
    print("=" * 90)
    print("TARGET FLOW SUMMARY")
    print("=" * 90)

    print(
        f"Source-only mean confidence: "
        f"{source_result['confidence'].mean().item():.6f}"
    )

    print(
        f"MCD mean confidence: "
        f"{mcd_result['confidence'].mean().item():.6f}"
    )

    print(
        f"Source-only mean margin: "
        f"{source_result['margin'].mean().item():.6f}"
    )

    print(
        f"MCD mean margin: "
        f"{mcd_result['margin'].mean().item():.6f}"
    )

    print(
        f"MCD mean disagreement: "
        f"{mcd_result['disagreement'].mean().item():.6f}"
    )

    source_before, mcd_after = build_flow_matrix(
        labels,
        source_result["predictions"],
        labels,
        mcd_result["predictions"],
    )

    source_rates = normalized_rows(
        source_before
    )

    mcd_rates = normalized_rows(
        mcd_after
    )

    delta_rates = (
        mcd_rates
        - source_rates
    )

    print()
    print("=" * 90)
    print("PER-CLASS FLOW CHANGE")
    print("=" * 90)

    print(
        f"{'Class':12s}"
        f"{'Before wrong %':>18s}"
        f"{'After wrong %':>18s}"
        f"{'Delta':>12s}"
    )

    for i, class_name in enumerate(
        CLASSES
    ):
        before_wrong = (
            100.0
            - source_rates[i, i].item()
        )

        after_wrong = (
            100.0
            - mcd_rates[i, i].item()
        )

        delta = (
            after_wrong
            - before_wrong
        )

        print(
            f"{class_name:12s}"
            f"{before_wrong:17.2f}%"
            f"{after_wrong:17.2f}%"
            f"{delta:+11.2f} pp"
        )

    print()
    print("=" * 90)
    print("LARGEST FLOW INCREASES")
    print("=" * 90)

    increases = strongest_changes(
        source_rates,
        mcd_rates,
    )

    for delta, count, true_name, pred_name in increases[:20]:
        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{delta:+7.2f} pp"
        )

    print()
    print("=" * 90)
    print("LARGEST FLOW REDUCTIONS")
    print("=" * 90)

    for delta, count, true_name, pred_name in increases[-20:][::-1]:
        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{delta:+7.2f} pp"
        )

    print()
    print("=" * 90)
    print("MCD ATTRACTOR ANALYSIS")
    print("=" * 90)

    (
        true_share,
        predicted_share,
        ratio,
    ) = attractor_scores(
        mcd_result["predictions"],
        labels,
    )

    for i, class_name in enumerate(
        CLASSES
    ):
        print(
            f"{class_name:12s} | "
            f"true "
            f"{true_share[i].item() * 100:6.2f}% | "
            f"pred "
            f"{predicted_share[i].item() * 100:6.2f}% | "
            f"ratio "
            f"{ratio[i].item():6.2f}x"
        )

    print()
    print("=" * 90)
    print("MCD CONDITIONAL UNCERTAINTY")
    print("=" * 90)

    conditional = conditional_margin_stats(
        labels,
        mcd_result["predictions"],
        mcd_result["margin"],
        mcd_result["disagreement"],
    )

    print(
        f"{'Class':12s}"
        f"{'Correct margin':>18s}"
        f"{'Wrong margin':>18s}"
        f"{'Correct disc':>18s}"
        f"{'Wrong disc':>18s}"
    )

    for item in conditional:
        print(
            f"{item['class']:12s}"
            f"{item['correct_margin']:17.5f}"
            f"{item['wrong_margin']:17.5f}"
            f"{item['correct_disagreement']:17.5f}"
            f"{item['wrong_disagreement']:17.5f}"
        )

    print()
    print("=" * 90)
    print("KEY PAIRS")
    print("=" * 90)

    key_pairs = [
        ("truck", "car"),
        ("truck", "train"),
        ("truck", "bus"),
        ("skateboard", "person"),
        ("skateboard", "knife"),
        ("skateboard", "motorcycle"),
        ("bicycle", "motorcycle"),
        ("person", "knife"),
        ("person", "horse"),
        ("bus", "car"),
        ("car", "person"),
    ]

    name_to_idx = {
        name: i
        for i, name in enumerate(CLASSES)
    }

    for true_name, pred_name in key_pairs:
        i = name_to_idx[true_name]
        j = name_to_idx[pred_name]

        before = source_rates[i, j].item()
        after = mcd_rates[i, j].item()
        delta = after - before

        print(
            f"{true_name:12s} → "
            f"{pred_name:12s} | "
            f"{before:6.2f}% → "
            f"{after:6.2f}% | "
            f"{delta:+7.2f} pp"
        )

    report = {
        "experiment":
            "visda_directional_flow_diagnostic",
        "seed":
            42,
        "target_labels_used_only_for_diagnostics":
            True,
        "source_only_mean_confidence":
            float(
                source_result["confidence"].mean().item()
            ),
        "mcd_mean_confidence":
            float(
                mcd_result["confidence"].mean().item()
            ),
        "source_only_mean_margin":
            float(
                source_result["margin"].mean().item()
            ),
        "mcd_mean_margin":
            float(
                mcd_result["margin"].mean().item()
            ),
        "mcd_mean_disagreement":
            float(
                mcd_result["disagreement"].mean().item()
            ),
        "source_flow_matrix":
            source_before.tolist(),
        "mcd_flow_matrix":
            mcd_after.tolist(),
        "source_flow_rates":
            source_rates.tolist(),
        "mcd_flow_rates":
            mcd_rates.tolist(),
        "flow_delta_rates":
            delta_rates.tolist(),
        "attractor_true_share":
            true_share.tolist(),
        "attractor_predicted_share":
            predicted_share.tolist(),
        "attractor_ratio":
            ratio.tolist(),
        "conditional_uncertainty":
            conditional,
        "key_pairs": [
            {
                "true_class": true_name,
                "predicted_class": pred_name,
                "before_percent":
                    source_rates[
                        name_to_idx[true_name],
                        name_to_idx[pred_name],
                    ].item(),
                "after_percent":
                    mcd_rates[
                        name_to_idx[true_name],
                        name_to_idx[pred_name],
                    ].item(),
                "delta_percent":
                    delta_rates[
                        name_to_idx[true_name],
                        name_to_idx[pred_name],
                    ].item(),
            }
            for true_name, pred_name in key_pairs
        ],
    }

    report_path = (
        OUTPUT_DIR
        / "directional_flow_seed42.json"
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