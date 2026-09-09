import json
from pathlib import Path

import numpy as np
import torch


SEED = 42

NUM_CLASSES = 12

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10
SKATEBOARD_ID = 9
PERSON_ID = 7

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

TARGET_CACHE = Path(
    "checkpoints/visda_feature_cache/target"
)

RPC_CHECKPOINT = Path(
    "checkpoints/visda_rpc_causal_experiment/"
    "rpc_seed42.pt"
)

GRAPH_OUTPUTS = Path(
    "checkpoints/visda_graph_semantic_diffusion/"
    "graph_semantic_outputs_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_truck_semantic_rank_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "truck_semantic_rank_audit_seed42.json"
)


def safe_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


def load_target_labels():
    files = sorted(
        TARGET_CACHE.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No target cache chunks found in {TARGET_CACHE}"
        )

    labels = []

    for path in files:
        payload = safe_load(path)

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
            )

        labels.append(
            payload["labels"].long()
        )

    return torch.cat(
        labels,
        dim=0
    )


def normalize_distribution(
    values
):
    values = values.float()

    if values.ndim != 2:
        raise RuntimeError(
            f"Expected N x C distribution, got {tuple(values.shape)}"
        )

    if values.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, got {values.shape[1]}"
        )

    values = values.clamp_min(
        0.0
    )

    row_sum = values.sum(
        dim=1,
        keepdim=True
    )

    return values / row_sum.clamp_min(
        1e-12
    )


def ranks_from_scores(
    scores,
    true_class
):
    order = torch.argsort(
        scores,
        dim=1,
        descending=True
    )

    true_positions = (
        order
        == true_class.view(
            -1,
            1
        )
    ).nonzero(
        as_tuple=False
    )

    ranks = torch.empty(
        len(scores),
        dtype=torch.long
    )

    ranks[
        true_positions[:, 0]
    ] = (
        true_positions[:, 1]
        + 1
    )

    return ranks


def rank_summary(
    ranks
):
    result = {}

    for k in (
        1,
        2,
        3,
        4,
        5
    ):
        result[
            f"rank_le_{k}"
        ] = float(
            (
                ranks <= k
            )
            .float()
            .mean()
            .item()
        )

    result["mean_rank"] = float(
        ranks.float().mean().item()
    )

    result["median_rank"] = float(
        ranks.float().median().item()
    )

    result["q25_rank"] = float(
        torch.quantile(
            ranks.float(),
            0.25
        ).item()
    )

    result["q75_rank"] = float(
        torch.quantile(
            ranks.float(),
            0.75
        ).item()
    )

    return result


def class_confusion_from_top_rank(
    scores,
    true_labels,
    focus_true_class
):
    mask = (
        true_labels
        == focus_true_class
    )

    if not mask.any():
        return {}

    predictions = scores.argmax(
        dim=1
    )

    selected = predictions[
        mask
    ]

    counts = torch.bincount(
        selected,
        minlength=NUM_CLASSES
    )

    total = int(
        mask.sum().item()
    )

    result = {}

    for class_id in range(
        NUM_CLASSES
    ):
        count = int(
            counts[class_id].item()
        )

        if count == 0:
            continue

        result[
            CLASSES[class_id]
        ] = {
            "count":
                count,
            "fraction":
                float(
                    count / total
                )
        }

    return result


def focus_score_statistics(
    scores,
    labels,
    focus_class,
    competitors
):
    mask = (
        labels
        == focus_class
    )

    if not mask.any():
        raise RuntimeError(
            f"No samples for {CLASSES[focus_class]}"
        )

    selected = scores[
        mask
    ]

    focus_scores = selected[
        :,
        focus_class
    ]

    result = {
        "count":
            int(
                mask.sum().item()
            ),
        "focus_mean":
            float(
                focus_scores.mean().item()
            ),
        "focus_median":
            float(
                focus_scores.median().item()
            ),
    }

    for competitor in competitors:
        values = selected[
            :,
            competitor
        ]

        margin = (
            focus_scores
            - values
        )

        result[
            f"{CLASSES[focus_class]}_vs_{CLASSES[competitor]}"
        ] = {
            "competitor_mean":
                float(
                    values.mean().item()
                ),
            "margin_mean":
                float(
                    margin.mean().item()
                ),
            "margin_median":
                float(
                    margin.median().item()
                ),
            "positive_margin_fraction":
                float(
                    (
                        margin > 0
                    )
                    .float()
                    .mean()
                    .item()
                )
        }

    return result


def binary_auc(
    scores,
    labels,
    positive_class
):
    y = (
        labels
        == positive_class
    ).long()

    positive = scores[
        y == 1
    ]

    negative = scores[
        y == 0
    ]

    if len(positive) == 0 or len(negative) == 0:
        return None

    combined = torch.cat(
        [
            positive,
            negative
        ]
    )

    order = torch.argsort(
        combined,
        descending=True
    )

    sorted_labels = y.new_zeros(
        len(combined)
    )

    sorted_labels[
        :len(positive)
    ] = 1

    sorted_labels = sorted_labels[
        order
    ]

    positive_count = float(
        len(positive)
    )

    negative_count = float(
        len(negative)
    )

    rank_sum = (
        torch.nonzero(
            sorted_labels == 1,
            as_tuple=False
        )
        .flatten()
        .float()
        + 1.0
    ).sum().item()

    auc = (
        rank_sum
        - positive_count
        * (positive_count + 1.0)
        / 2.0
    ) / (
        positive_count
        * negative_count
    )

    return float(
        auc
    )


def main():
    print("=" * 90)
    print(
        "VISDA-2017 TRUCK SEMANTIC RANK AUDIT"
    )
    print("=" * 90)

    print(
        f"seed={SEED}"
    )

    print(
        f"rpc_checkpoint={RPC_CHECKPOINT}"
    )

    print(
        f"graph_outputs={GRAPH_OUTPUTS}"
    )

    print()

    print(
        "Loading target labels..."
    )

    target_labels = load_target_labels()

    print(
        f"Target samples: "
        f"{len(target_labels)}"
    )

    print()

    print(
        "Loading graph semantic outputs..."
    )

    graph_payload = safe_load(
        GRAPH_OUTPUTS
    )

    required_graph_keys = [
        "anchor_probabilities",
        "diffused_probabilities",
    ]

    for key in required_graph_keys:
        if key not in graph_payload:
            raise RuntimeError(
                f"Graph output missing {key}"
            )

    anchor_scores = normalize_distribution(
        graph_payload[
            "anchor_probabilities"
        ]
    )

    diffusion_scores = normalize_distribution(
        graph_payload[
            "diffused_probabilities"
        ]
    )

    if len(anchor_scores) != len(
        target_labels
    ):
        raise RuntimeError(
            "Anchor output and target labels have different lengths"
        )

    if len(diffusion_scores) != len(
        target_labels
    ):
        raise RuntimeError(
            "Diffusion output and target labels have different lengths"
        )

    print(
        f"Anchor scores: "
        f"{tuple(anchor_scores.shape)}"
    )

    print(
        f"Diffusion scores: "
        f"{tuple(diffusion_scores.shape)}"
    )

    print()

    print(
        "Loading RPC checkpoint..."
    )

    rpc_payload = safe_load(
        RPC_CHECKPOINT
    )

    if "final_metrics" in rpc_payload:
        checkpoint_metrics = (
            rpc_payload[
                "final_metrics"
            ]
        )

        print(
            "Checkpoint final mean-class: "
            f"{checkpoint_metrics.get('mean_class_accuracy', 'unknown')}"
        )

    print()

    print(
        "=========================================================="
    )

    print(
        "TRUE TRUCK RANK ANALYSIS"
    )

    print(
        "=========================================================="
    )

    truck_mask = (
        target_labels
        == TRUCK_ID
    )

    truck_labels = target_labels[
        truck_mask
    ]

    truck_anchor = anchor_scores[
        truck_mask
    ]

    truck_diffusion = diffusion_scores[
        truck_mask
    ]

    truck_anchor_ranks = ranks_from_scores(
        truck_anchor,
        truck_labels
    )

    truck_diffusion_ranks = ranks_from_scores(
        truck_diffusion,
        truck_labels
    )

    anchor_rank_summary = rank_summary(
        truck_anchor_ranks
    )

    diffusion_rank_summary = rank_summary(
        truck_diffusion_ranks
    )

    print()
    print(
        "SOURCE MULTI-PROTOTYPE ANCHOR"
    )

    for key, value in anchor_rank_summary.items():
        if key == "mean_rank":
            print(
                f"{key:20s}: {value:.4f}"
            )
        else:
            print(
                f"{key:20s}: {100.0 * value:.2f}%"
            )

    print()
    print(
        "TARGET GRAPH DIFFUSION"
    )

    for key, value in diffusion_rank_summary.items():
        if key == "mean_rank":
            print(
                f"{key:20s}: {value:.4f}"
            )
        else:
            print(
                f"{key:20s}: {100.0 * value:.2f}%"
            )

    print()
    print(
        "TRUCK TOP-1 CONFUSION"
    )

    anchor_confusion = (
        class_confusion_from_top_rank(
            anchor_scores,
            target_labels,
            TRUCK_ID
        )
    )

    diffusion_confusion = (
        class_confusion_from_top_rank(
            diffusion_scores,
            target_labels,
            TRUCK_ID
        )
    )

    print()
    print(
        "Anchor:"
    )

    for name, row in sorted(
        anchor_confusion.items(),
        key=lambda item: item[1]["count"],
        reverse=True
    ):
        print(
            f"{name:12s} | "
            f"n={row['count']:5d} | "
            f"{100.0 * row['fraction']:7.2f}%"
        )

    print()
    print(
        "Diffusion:"
    )

    for name, row in sorted(
        diffusion_confusion.items(),
        key=lambda item: item[1]["count"],
        reverse=True
    ):
        print(
            f"{name:12s} | "
            f"n={row['count']:5d} | "
            f"{100.0 * row['fraction']:7.2f}%"
        )

    print()
    print(
        "TRUCK VS MAIN COMPETITORS"
    )

    competitors = [
        CAR_ID,
        BUS_ID,
        TRAIN_ID,
        SKATEBOARD_ID,
        PERSON_ID,
    ]

    anchor_focus = focus_score_statistics(
        anchor_scores,
        target_labels,
        TRUCK_ID,
        competitors
    )

    diffusion_focus = focus_score_statistics(
        diffusion_scores,
        target_labels,
        TRUCK_ID,
        competitors
    )

    print()
    print(
        "Anchor"
    )

    print(
        f"truck mean: "
        f"{anchor_focus['focus_mean']:.6f}"
    )

    print(
        f"truck median: "
        f"{anchor_focus['focus_median']:.6f}"
    )

    for competitor in competitors:
        key = (
            f"truck_vs_{CLASSES[competitor]}"
        )

        row = anchor_focus[key]

        print(
            f"truck vs "
            f"{CLASSES[competitor]:10s} | "
            f"competitor={row['competitor_mean']:.6f} | "
            f"margin={row['margin_mean']:.6f} | "
            f"positive={100.0 * row['positive_margin_fraction']:.2f}%"
        )

    print()
    print(
        "Diffusion"
    )

    print(
        f"truck mean: "
        f"{diffusion_focus['focus_mean']:.6f}"
    )

    print(
        f"truck median: "
        f"{diffusion_focus['focus_median']:.6f}"
    )

    for competitor in competitors:
        key = (
            f"truck_vs_{CLASSES[competitor]}"
        )

        row = diffusion_focus[key]

        print(
            f"truck vs "
            f"{CLASSES[competitor]:10s} | "
            f"competitor={row['competitor_mean']:.6f} | "
            f"margin={row['margin_mean']:.6f} | "
            f"positive={100.0 * row['positive_margin_fraction']:.2f}%"
        )

    print()
    print(
        "TRUCK SCORE DISCRIMINATION"
    )

    anchor_truck_auc = binary_auc(
        anchor_scores[:, TRUCK_ID],
        target_labels,
        TRUCK_ID
    )

    diffusion_truck_auc = binary_auc(
        diffusion_scores[:, TRUCK_ID],
        target_labels,
        TRUCK_ID
    )

    print(
        f"Anchor truck-score AUROC: "
        f"{anchor_truck_auc}"
    )

    print(
        f"Diffused truck-score AUROC: "
        f"{diffusion_truck_auc}"
    )

    print()
    print(
        "PER-RANK TRUCK BREAKDOWN"
    )

    for rank_limit in (
        1,
        2,
        3,
        4,
        5
    ):
        anchor_fraction = (
            (
                truck_anchor_ranks
                <= rank_limit
            )
            .float()
            .mean()
            .item()
        )

        diffusion_fraction = (
            (
                truck_diffusion_ranks
                <= rank_limit
            )
            .float()
            .mean()
            .item()
        )

        print(
            f"top-{rank_limit} | "
            f"anchor={100.0 * anchor_fraction:.2f}% | "
            f"diffusion={100.0 * diffusion_fraction:.2f}%"
        )

    print()
    print(
        "GLOBAL SEMANTIC PERFORMANCE"
    )

    anchor_predictions = anchor_scores.argmax(
        dim=1
    )

    diffusion_predictions = diffusion_scores.argmax(
        dim=1
    )

    anchor_correct = (
        anchor_predictions
        == target_labels
    )

    diffusion_correct = (
        diffusion_predictions
        == target_labels
    )

    anchor_class_accuracy = []

    diffusion_class_accuracy = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            target_labels
            == class_id
        )

        anchor_class_accuracy.append(
            100.0
            * anchor_correct[mask]
            .float()
            .mean()
            .item()
        )

        diffusion_class_accuracy.append(
            100.0
            * diffusion_correct[mask]
            .float()
            .mean()
            .item()
        )

    print(
        f"Anchor overall: "
        f"{100.0 * anchor_correct.float().mean().item():.2f}%"
    )

    print(
        f"Diffusion overall: "
        f"{100.0 * diffusion_correct.float().mean().item():.2f}%"
    )

    print(
        f"Anchor MCA: "
        f"{np.mean(anchor_class_accuracy):.2f}%"
    )

    print(
        f"Diffusion MCA: "
        f"{np.mean(diffusion_class_accuracy):.2f}%"
    )

    print()
    print(
        "ALL TRUE TRUCK SAMPLES: SCORE RANK DISTRIBUTION"
    )

    anchor_rank_counts = torch.bincount(
        truck_anchor_ranks,
        minlength=NUM_CLASSES + 1
    )[1:]

    diffusion_rank_counts = torch.bincount(
        truck_diffusion_ranks,
        minlength=NUM_CLASSES + 1
    )[1:]

    max_print_rank = 12

    for rank in range(
        1,
        max_print_rank + 1
    ):
        anchor_count = int(
            anchor_rank_counts[
                rank - 1
            ].item()
        )

        diffusion_count = int(
            diffusion_rank_counts[
                rank - 1
            ].item()
        )

        total = len(
            truck_labels
        )

        print(
            f"rank={rank:2d} | "
            f"anchor={anchor_count:5d} "
            f"({100.0 * anchor_count / total:6.2f}%) | "
            f"diffusion={diffusion_count:5d} "
            f"({100.0 * diffusion_count / total:6.2f}%)"
        )

    result = {
        "experiment":
            "visda_truck_semantic_rank_audit",
        "seed":
            SEED,
        "rpc_checkpoint":
            str(RPC_CHECKPOINT),
        "graph_outputs":
            str(GRAPH_OUTPUTS),
        "target_samples":
            len(target_labels),
        "truck_samples":
            int(
                truck_mask.sum().item()
            ),
        "anchor_rank_summary":
            anchor_rank_summary,
        "diffusion_rank_summary":
            diffusion_rank_summary,
        "anchor_truck_confusion":
            anchor_confusion,
        "diffusion_truck_confusion":
            diffusion_confusion,
        "anchor_truck_focus_statistics":
            anchor_focus,
        "diffusion_truck_focus_statistics":
            diffusion_focus,
        "anchor_truck_score_auc":
            anchor_truck_auc,
        "diffusion_truck_score_auc":
            diffusion_truck_auc,
        "anchor_per_class_accuracy":
            {
                CLASSES[i]:
                    anchor_class_accuracy[i]
                for i in range(
                    NUM_CLASSES
                )
            },
        "diffusion_per_class_accuracy":
            {
                CLASSES[i]:
                    diffusion_class_accuracy[i]
                for i in range(
                    NUM_CLASSES
                )
            }
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2
        )

    print()
    print("=" * 90)
    print(
        "TRUCK SEMANTIC RANK AUDIT COMPLETE"
    )
    print("=" * 90)

    print(
        f"Anchor truck top-1: "
        f"{100.0 * anchor_rank_summary['rank_le_1']:.2f}%"
    )

    print(
        f"Diffusion truck top-1: "
        f"{100.0 * diffusion_rank_summary['rank_le_1']:.2f}%"
    )

    print(
        f"Anchor truck mean rank: "
        f"{anchor_rank_summary['mean_rank']:.4f}"
    )

    print(
        f"Diffusion truck mean rank: "
        f"{diffusion_rank_summary['mean_rank']:.4f}"
    )

    print(
        f"Anchor truck AUROC: "
        f"{anchor_truck_auc}"
    )

    print(
        f"Diffusion truck AUROC: "
        f"{diffusion_truck_auc}"
    )

    print(
        f"Saved: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()