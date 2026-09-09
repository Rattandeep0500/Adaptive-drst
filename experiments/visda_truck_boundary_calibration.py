import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


SEED = 42

NUM_CLASSES = 12

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10

COMPETITOR_IDS = [
    CAR_ID,
    BUS_ID,
    TRAIN_ID,
]

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

GRAPH_OUTPUTS = Path(
    "checkpoints/visda_graph_semantic_diffusion/"
    "graph_semantic_outputs_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_truck_boundary_calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "truck_boundary_calibration_seed42.json"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
            f"No target cache files found in {TARGET_CACHE}"
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


def normalize_scores(scores):
    scores = scores.float()

    if scores.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D scores, got {tuple(scores.shape)}"
        )

    if scores.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, got {scores.shape[1]}"
        )

    scores = scores.clamp_min(
        1e-12
    )

    return scores / scores.sum(
        dim=1,
        keepdim=True
    )


def binary_metrics(
    scores,
    labels,
    positive_class
):
    y = (
        labels
        == positive_class
    ).long()

    if y.unique().numel() < 2:
        raise RuntimeError(
            "Binary evaluation requires both positive and negative samples"
        )

    predictions = (
        scores
        >= 0.5
    ).long()

    accuracy = (
        predictions
        == y
    ).float().mean().item()

    auc = roc_auc_score(
        y.numpy(),
        scores.numpy()
    )

    tp = int(
        (
            (predictions == 1)
            & (y == 1)
        ).sum().item()
    )

    tn = int(
        (
            (predictions == 0)
            & (y == 0)
        ).sum().item()
    )

    fp = int(
        (
            (predictions == 1)
            & (y == 0)
        ).sum().item()
    )

    fn = int(
        (
            (predictions == 0)
            & (y == 1)
        ).sum().item()
    )

    precision = (
        tp / max(
            tp + fp,
            1
        )
    )

    recall = (
        tp / max(
            tp + fn,
            1
        )
    )

    return {
        "accuracy":
            float(
                100.0 * accuracy
            ),
        "auc":
            float(
                auc
            ),
        "precision":
            float(
                100.0 * precision
            ),
        "recall":
            float(
                100.0 * recall
            ),
        "tp":
            tp,
        "tn":
            tn,
        "fp":
            fp,
        "fn":
            fn,
    }


def multiclass_accuracy(
    predictions,
    labels
):
    correct = (
        predictions
        == labels
    )

    overall = (
        100.0
        * correct.float().mean().item()
    )

    per_class = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            labels
            == class_id
        )

        if not mask.any():
            per_class.append(
                0.0
            )
        else:
            per_class.append(
                100.0
                * correct[
                    mask
                ]
                .float()
                .mean()
                .item()
            )

    return {
        "overall":
            overall,
        "mean_class":
            float(
                np.mean(
                    per_class
                )
            ),
        "per_class":
            per_class,
    }


def top1_predictions(
    scores
):
    return scores.argmax(
        dim=1
    )


def build_adjusted_scores(
    scores,
    params
):
    logits = (
        scores.clamp_min(
            1e-12
        ).log()
    )

    truck_adjusted = (
        logits[
            :,
            TRUCK_ID
        ]
        + params["bias"]
        + params["car"] * logits[
            :,
            CAR_ID
        ]
        + params["bus"] * logits[
            :,
            BUS_ID
        ]
        + params["train"] * logits[
            :,
            TRAIN_ID
        ]
    )

    other_indices = [
        i
        for i in range(
            NUM_CLASSES
        )
        if i != TRUCK_ID
    ]

    adjusted_logits = logits.clone()

    adjusted_logits[
        :,
        TRUCK_ID
    ] = truck_adjusted

    adjusted_scores = torch.softmax(
        adjusted_logits,
        dim=1
    )

    return adjusted_scores


def build_log_odds_features(
    scores
):
    eps = 1e-12

    log_scores = (
        scores
        .clamp_min(
            eps
        )
        .log()
    )

    truck_score = log_scores[
        :,
        TRUCK_ID
    ]

    features = []

    features.append(
        (
            truck_score
            - log_scores[
                :,
                CAR_ID
            ]
        ).numpy()
    )

    features.append(
        (
            truck_score
            - log_scores[
                :,
                BUS_ID
            ]
        ).numpy()
    )

    features.append(
        (
            truck_score
            - log_scores[
                :,
                TRAIN_ID
            ]
        ).numpy()
    )

    features.append(
        truck_score.numpy()
    )

    return np.stack(
        features,
        axis=1
    )


def fit_pairwise_logistic(
    scores,
    labels,
    train_mask
):
    x = build_log_odds_features(
        scores
    )

    y = (
        labels
        == TRUCK_ID
    ).long().numpy()

    x_train = x[
        train_mask.numpy()
    ]

    y_train = y[
        train_mask.numpy()
    ]

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=2000,
        random_state=SEED,
        class_weight=None
    )

    model.fit(
        x_train,
        y_train
    )

    return model


def logistic_scores(
    model,
    scores
):
    x = build_log_odds_features(
        scores
    )

    probability = model.predict_proba(
        x
    )[:, 1]

    return torch.from_numpy(
        probability
    ).float()


def evaluate_binary_score(
    name,
    score,
    labels,
    mask
):
    selected_score = score[
        mask
    ]

    selected_labels = labels[
        mask
    ]

    metrics = binary_metrics(
        selected_score,
        selected_labels,
        TRUCK_ID
    )

    print()
    print(name)

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.2f}%"
    )

    print(
        f"AUROC: "
        f"{metrics['auc']:.6f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.2f}%"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.2f}%"
    )

    return metrics


def multiclass_from_truck_probability(
    original_scores,
    truck_probability
):
    result = original_scores.clone()

    original_truck = result[
        :,
        TRUCK_ID
    ]

    adjusted_truck = truck_probability.clamp(
        1e-6,
        1.0 - 1e-6
    )

    non_truck_mass = (
        1.0
        - original_truck
    ).clamp_min(
        1e-12
    )

    scale = (
        1.0
        - adjusted_truck
    ) / non_truck_mass

    non_truck_ids = [
        i
        for i in range(
            NUM_CLASSES
        )
        if i != TRUCK_ID
    ]

    result[
        :,
        non_truck_ids
    ] *= scale.unsqueeze(
        1
    )

    result[
        :,
        TRUCK_ID
    ] = adjusted_truck

    return result


def evaluate_candidate_boundary(
    name,
    truck_probability,
    labels,
    evaluation_mask,
    original_scores
):
    selected_probability = truck_probability[
        evaluation_mask
    ]

    selected_labels = labels[
        evaluation_mask
    ]

    binary = binary_metrics(
        selected_probability,
        selected_labels,
        TRUCK_ID
    )

    multiclass_scores = multiclass_from_truck_probability(
        original_scores[
            evaluation_mask
        ],
        selected_probability
    )

    predictions = top1_predictions(
        multiclass_scores
    )

    multi = multiclass_accuracy(
        predictions,
        selected_labels
    )

    print()
    print(name)

    print(
        f"Truck binary AUROC: "
        f"{binary['auc']:.6f}"
    )

    print(
        f"Truck binary accuracy: "
        f"{binary['accuracy']:.2f}%"
    )

    print(
        f"Truck recall: "
        f"{binary['recall']:.2f}%"
    )

    print(
        f"Multiclass overall: "
        f"{multi['overall']:.2f}%"
    )

    print(
        f"Multiclass MCA: "
        f"{multi['mean_class']:.2f}%"
    )

    print(
        f"Multiclass truck: "
        f"{multi['per_class'][TRUCK_ID]:.2f}%"
    )

    print(
        f"Multiclass car: "
        f"{multi['per_class'][CAR_ID]:.2f}%"
    )

    print(
        f"Multiclass bus: "
        f"{multi['per_class'][BUS_ID]:.2f}%"
    )

    print(
        f"Multiclass train: "
        f"{multi['per_class'][TRAIN_ID]:.2f}%"
    )

    return {
        "binary":
            binary,
        "multiclass":
            multi
    }


def threshold_search(
    scores,
    labels,
    calibration_mask,
    evaluation_mask
):
    selected_scores = scores[
        calibration_mask
    ]

    selected_labels = labels[
        calibration_mask
    ]

    truck_score = selected_scores[
        :,
        TRUCK_ID
    ]

    car_score = selected_scores[
        :,
        CAR_ID
    ]

    bus_score = selected_scores[
        :,
        BUS_ID
    ]

    train_score = selected_scores[
        :,
        TRAIN_ID
    ]

    candidates = []

    for a in np.linspace(
        -1.0,
        2.0,
        13
    ):
        for b in np.linspace(
            -1.0,
            2.0,
            13
        ):
            for c in np.linspace(
                -1.0,
                2.0,
                13
            ):
                score = (
                    truck_score
                    - a * car_score
                    - b * bus_score
                    - c * train_score
                )

                auc = roc_auc_score(
                    (
                        selected_labels
                        == TRUCK_ID
                    ).numpy(),
                    score.numpy()
                )

                candidates.append(
                    (
                        auc,
                        float(a),
                        float(b),
                        float(c)
                    )
                )

    candidates.sort(
        reverse=True
    )

    best = candidates[
        0
    ]

    a = best[1]
    b = best[2]
    c = best[3]

    evaluation_score = (
        scores[
            evaluation_mask,
            TRUCK_ID
        ]
        - a * scores[
            evaluation_mask,
            CAR_ID
        ]
        - b * scores[
            evaluation_mask,
            BUS_ID
        ]
        - c * scores[
            evaluation_mask,
            TRAIN_ID
        ]
    )

    evaluation_labels = labels[
        evaluation_mask
    ]

    auc = roc_auc_score(
        (
            evaluation_labels
            == TRUCK_ID
        ).numpy(),
        evaluation_score.numpy()
    )

    return {
        "best_auc_calibration":
            float(
                best[0]
            ),
        "car_weight":
            a,
        "bus_weight":
            b,
        "train_weight":
            c,
        "evaluation_auc":
            float(
                auc
            )
    }


def split_mask(
    labels
):
    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    for class_id in range(
        NUM_CLASSES
    ):
        indices = torch.nonzero(
            labels == class_id,
            as_tuple=False
        ).flatten()

        permutation = torch.randperm(
            len(indices),
            generator=generator
        )

        split = int(
            0.5
            * len(indices)
        )

        calibration_indices = (
            indices[
                permutation[
                    :split
                ]
            ]
        )

        mask[
            calibration_indices
        ] = True

    return mask


def main():
    set_seed(
        SEED
    )

    print("=" * 90)
    print(
        "VISDA-2017 TRUCK-VS-SINK BOUNDARY CALIBRATION"
    )
    print("=" * 90)

    print(
        f"seed={SEED}"
    )

    print(
        f"graph_outputs={GRAPH_OUTPUTS}"
    )

    print()

    labels = load_target_labels()

    graph_payload = safe_load(
        GRAPH_OUTPUTS
    )

    anchor_scores = normalize_scores(
        graph_payload[
            "anchor_probabilities"
        ]
    )

    diffusion_scores = normalize_scores(
        graph_payload[
            "diffused_probabilities"
        ]
    )

    print(
        f"Target samples: "
        f"{len(labels)}"
    )

    print(
        f"Anchor scores: "
        f"{tuple(anchor_scores.shape)}"
    )

    print(
        f"Diffusion scores: "
        f"{tuple(diffusion_scores.shape)}"
    )

    calibration_mask = split_mask(
        labels
    )

    evaluation_mask = ~calibration_mask

    print()
    print(
        "Calibration/evaluation split"
    )

    print(
        f"Calibration samples: "
        f"{int(calibration_mask.sum().item())}"
    )

    print(
        f"Evaluation samples: "
        f"{int(evaluation_mask.sum().item())}"
    )

    anchor_predictions = top1_predictions(
        anchor_scores
    )

    diffusion_predictions = top1_predictions(
        diffusion_scores
    )

    anchor_global = multiclass_accuracy(
        anchor_predictions,
        labels
    )

    diffusion_global = multiclass_accuracy(
        diffusion_predictions,
        labels
    )

    print()
    print(
        "BASELINE REFERENCE"
    )

    print(
        f"Anchor overall: "
        f"{anchor_global['overall']:.2f}%"
    )

    print(
        f"Anchor MCA: "
        f"{anchor_global['mean_class']:.2f}%"
    )

    print(
        f"Anchor truck: "
        f"{anchor_global['per_class'][TRUCK_ID]:.2f}%"
    )

    print(
        f"Diffusion overall: "
        f"{diffusion_global['overall']:.2f}%"
    )

    print(
        f"Diffusion MCA: "
        f"{diffusion_global['mean_class']:.2f}%"
    )

    print(
        f"Diffusion truck: "
        f"{diffusion_global['per_class'][TRUCK_ID]:.2f}%"
    )

    print()
    print(
        "RAW TRUCK SCORE"
    )

    raw_anchor_truck = anchor_scores[
        :,
        TRUCK_ID
    ]

    raw_diffusion_truck = diffusion_scores[
        :,
        TRUCK_ID
    ]

    anchor_raw_metrics = evaluate_binary_score(
        "Anchor truck probability",
        raw_anchor_truck,
        labels,
        evaluation_mask
    )

    diffusion_raw_metrics = evaluate_binary_score(
        "Diffusion truck probability",
        raw_diffusion_truck,
        labels,
        evaluation_mask
    )

    print()
    print(
        "PAIRWISE LOGISTIC CALIBRATION"
    )

    anchor_model = fit_pairwise_logistic(
        anchor_scores,
        labels,
        calibration_mask
    )

    diffusion_model = fit_pairwise_logistic(
        diffusion_scores,
        labels,
        calibration_mask
    )

    anchor_calibrated_probability = (
        logistic_scores(
            anchor_model,
            anchor_scores
        )
    )

    diffusion_calibrated_probability = (
        logistic_scores(
            diffusion_model,
            diffusion_scores
        )
    )

    anchor_calibrated = evaluate_candidate_boundary(
        "Anchor pairwise calibration",
        anchor_calibrated_probability,
        labels,
        evaluation_mask,
        anchor_scores
    )

    diffusion_calibrated = evaluate_candidate_boundary(
        "Diffusion pairwise calibration",
        diffusion_calibrated_probability,
        labels,
        evaluation_mask,
        diffusion_scores
    )

    print()
    print(
        "SIMPLE TRUCK SCORE - COMPETITOR SCORE CALIBRATION"
    )

    anchor_threshold = threshold_search(
        anchor_scores,
        labels,
        calibration_mask,
        evaluation_mask
    )

    diffusion_threshold = threshold_search(
        diffusion_scores,
        labels,
        calibration_mask,
        evaluation_mask
    )

    print()
    print(
        "Anchor:"
    )

    print(
        f"calibration AUC="
        f"{anchor_threshold['best_auc_calibration']:.6f}"
    )

    print(
        f"weights "
        f"car={anchor_threshold['car_weight']:.3f} "
        f"bus={anchor_threshold['bus_weight']:.3f} "
        f"train={anchor_threshold['train_weight']:.3f}"
    )

    print(
        f"evaluation AUC="
        f"{anchor_threshold['evaluation_auc']:.6f}"
    )

    print()
    print(
        "Diffusion:"
    )

    print(
        f"calibration AUC="
        f"{diffusion_threshold['best_auc_calibration']:.6f}"
    )

    print(
        f"weights "
        f"car={diffusion_threshold['car_weight']:.3f} "
        f"bus={diffusion_threshold['bus_weight']:.3f} "
        f"train={diffusion_threshold['train_weight']:.3f}"
    )

    print(
        f"evaluation AUC="
        f"{diffusion_threshold['evaluation_auc']:.6f}"
    )

    print()
    print(
        "TRUCK VS COMPETITOR RAW MARGINS"
    )

    focus_rows = {}

    for name, scores in (
        (
            "anchor",
            anchor_scores
        ),
        (
            "diffusion",
            diffusion_scores
        )
    ):
        truck = scores[
            :,
            TRUCK_ID
        ]

        focus_rows[
            name
        ] = {}

        for competitor_id in (
            CAR_ID,
            BUS_ID,
            TRAIN_ID
        ):
            margin = (
                truck
                - scores[
                    :,
                    competitor_id
                ]
            )

            calibration_margin = margin[
                calibration_mask
            ]

            evaluation_margin = margin[
                evaluation_mask
            ]

            focus_rows[
                name
            ][
                CLASSES[competitor_id]
            ] = {
                "calibration_mean":
                    float(
                        calibration_margin.mean().item()
                    ),
                "evaluation_mean":
                    float(
                        evaluation_margin.mean().item()
                    ),
                "evaluation_positive_fraction":
                    float(
                        (
                            evaluation_margin
                            > 0
                        )
                        .float()
                        .mean()
                        .item()
                    )
            }

            print(
                f"{name:10s} truck-{CLASSES[competitor_id]:5s} | "
                f"cal_mean="
                f"{calibration_margin.mean().item():.6f} | "
                f"eval_mean="
                f"{evaluation_margin.mean().item():.6f} | "
                f"eval_positive="
                f"{100.0 * (
                    (evaluation_margin > 0)
                    .float()
                    .mean()
                    .item()
                ):.2f}%"
            )

    result = {
        "experiment":
            "visda_truck_boundary_calibration",
        "seed":
            SEED,
        "graph_outputs":
            str(GRAPH_OUTPUTS),
        "target_samples":
            len(labels),
        "calibration_samples":
            int(
                calibration_mask.sum().item()
            ),
        "evaluation_samples":
            int(
                evaluation_mask.sum().item()
            ),
        "baseline_anchor":
            anchor_global,
        "baseline_diffusion":
            diffusion_global,
        "anchor_raw_truck":
            anchor_raw_metrics,
        "diffusion_raw_truck":
            diffusion_raw_metrics,
        "anchor_pairwise_calibration":
            anchor_calibrated,
        "diffusion_pairwise_calibration":
            diffusion_calibrated,
        "anchor_threshold_calibration":
            anchor_threshold,
        "diffusion_threshold_calibration":
            diffusion_threshold,
        "truck_competitor_margins":
            focus_rows
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
        "TRUCK BOUNDARY CALIBRATION COMPLETE"
    )
    print("=" * 90)

    print(
        f"Saved: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()