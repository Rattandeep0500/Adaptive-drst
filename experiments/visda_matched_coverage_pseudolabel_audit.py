import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


SEED = 42
NUM_CLASSES = 12
TRUCK = 11

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

COVERAGES = [
    0.10,
    0.20,
    0.30,
    0.50,
    0.70,
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--support-output",
        type=Path,
        default=Path(
            "checkpoints/visda_target_class_conditional_support/"
            "target_class_conditional_support_seed42.npz"
        ),
    )

    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path(
            "checkpoints/visda_feature_cache/target"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "checkpoints/visda_matched_coverage_pseudolabel_audit"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_support_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Support artifact not found: {path}"
        )

    payload = np.load(
        path,
        allow_pickle=False,
    )

    required = [
        "support_probabilities",
        "support_top1",
        "support_top1_score",
        "support_margin",
        "support_entropy",
        "support_reliability",
        "teacher_probabilities",
        "teacher_prediction",
        "teacher_confidence",
        "teacher_support_agreement",
    ]

    missing = [
        key
        for key in required
        if key not in payload.files
    ]

    if missing:
        raise RuntimeError(
            "Support artifact is missing required arrays: "
            + ", ".join(missing)
        )

    arrays = {
        key: payload[key]
        for key in required
    }

    return arrays


def load_target_labels(cache_dir):
    files = sorted(
        cache_dir.glob(
            "chunk_*.pt"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No target cache chunks found in {cache_dir}"
        )

    labels = []

    for path in files:
        payload = safe_torch_load(
            path
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                f"Unexpected cache object in {path}"
            )

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
            )

        labels.append(
            payload[
                "labels"
            ]
            .long()
            .cpu()
        )

    result = torch.cat(
        labels,
        dim=0,
    ).numpy().astype(
        np.int64
    )

    return result


def validate_arrays(arrays):
    n = len(
        arrays[
            "teacher_prediction"
        ]
    )

    for key, value in arrays.items():
        value = np.asarray(
            value
        )

        if len(value) != n:
            raise RuntimeError(
                f"{key} has length {len(value)} "
                f"but expected {n}"
            )

    teacher_prediction = arrays[
        "teacher_prediction"
    ]

    teacher_confidence = arrays[
        "teacher_confidence"
    ]

    support_top1 = arrays[
        "support_top1"
    ]

    support_top1_score = arrays[
        "support_top1_score"
    ]

    support_margin = arrays[
        "support_margin"
    ]

    support_reliability = arrays[
        "support_reliability"
    ]

    agreement = arrays[
        "teacher_support_agreement"
    ]

    if teacher_prediction.ndim != 1:
        raise RuntimeError(
            "teacher_prediction must be 1D"
        )

    if support_top1.ndim != 1:
        raise RuntimeError(
            "support_top1 must be 1D"
        )

    if np.any(
        teacher_prediction < 0
    ) or np.any(
        teacher_prediction >= NUM_CLASSES
    ):
        raise RuntimeError(
            "teacher_prediction contains invalid class IDs"
        )

    if np.any(
        support_top1 < 0
    ) or np.any(
        support_top1 >= NUM_CLASSES
    ):
        raise RuntimeError(
            "support_top1 contains invalid class IDs"
        )

    finite_arrays = [
        teacher_confidence,
        support_top1_score,
        support_margin,
        support_reliability,
    ]

    for value in finite_arrays:
        if not np.all(
            np.isfinite(value)
        ):
            raise RuntimeError(
                "A reliability array contains non-finite values"
            )

    expected_agreement = (
        teacher_prediction
        == support_top1
    ).astype(
        np.float64
    )

    stored_agreement = (
        np.asarray(
            agreement
        )
        .astype(
            np.float64
        )
    )

    if not np.array_equal(
        expected_agreement,
        stored_agreement,
    ):
        raise RuntimeError(
            "teacher_support_agreement does not match "
            "teacher_prediction == support_top1"
        )

    return n


def rank_percentiles(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n <= 1:
        return np.ones(
            n,
            dtype=np.float64,
        )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    ranks[
        order
    ] = np.arange(
        n,
        dtype=np.float64,
    )

    return ranks / (
        n - 1
    )


def average_precision(
    scores,
    labels,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    positives = int(
        np.sum(
            labels == 1
        )
    )

    if positives == 0:
        return 0.0

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_labels = labels[
        order
    ]

    tp = 0
    ap = 0.0

    for rank, label in enumerate(
        sorted_labels,
        start=1,
    ):
        if label == 1:
            tp += 1
            ap += (
                tp / rank
            )

    return float(
        ap / positives
    )


def auc_binary(
    scores,
    labels,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    positive_count = int(
        np.sum(
            labels == 1
        )
    )

    negative_count = int(
        np.sum(
            labels == 0
        )
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        return None

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    sorted_scores = scores[
        order
    ]

    sorted_labels = labels[
        order
    ]

    ranks = np.arange(
        1,
        len(scores) + 1,
        dtype=np.float64,
    )

    positive_ranks = ranks[
        sorted_labels == 1
    ]

    rank_sum = float(
        np.sum(
            positive_ranks
        )
    )

    auc = (
        rank_sum
        - positive_count
        * (positive_count + 1)
        / 2.0
    ) / (
        positive_count
        * negative_count
    )

    return float(auc)


def top_fraction_rows(
    score,
    correctness,
):
    score = np.asarray(
        score,
        dtype=np.float64,
    )

    correctness = np.asarray(
        correctness,
        dtype=np.int64,
    )

    order = np.argsort(
        -score,
        kind="mergesort",
    )

    rows = []

    for fraction in COVERAGES:
        count = max(
            1,
            int(
                np.ceil(
                    fraction
                    * len(order)
                )
            ),
        )

        selected = order[
            :count
        ]

        accuracy = float(
            correctness[
                selected
            ].mean()
        )

        rows.append(
            {
                "coverage": float(
                    fraction
                ),
                "selected": int(
                    count
                ),
                "correct": int(
                    correctness[
                        selected
                    ].sum()
                ),
                "precision": accuracy,
            }
        )

    return rows


def build_methods(
    teacher_confidence,
    support_on_teacher,
    support_margin,
    support_reliability,
    agreement,
):
    teacher_rank = rank_percentiles(
        teacher_confidence
    )

    support_rank = rank_percentiles(
        support_on_teacher
    )

    margin_rank = rank_percentiles(
        support_margin
    )

    reliability_rank = rank_percentiles(
        support_reliability
    )

    methods = {}

    methods[
        "teacher_confidence"
    ] = teacher_confidence.copy()

    methods[
        "independent_support"
    ] = support_on_teacher.copy()

    methods[
        "support_margin"
    ] = support_margin.copy()

    methods[
        "agreement_first"
    ] = (
        0.70 * agreement
        + 0.30 * teacher_rank
    )

    methods[
        "confidence_support_agreement"
    ] = (
        0.45 * teacher_rank
        + 0.30 * support_rank
        + 0.25 * agreement
    )

    methods[
        "confidence_support_margin"
    ] = (
        0.40 * teacher_rank
        + 0.30 * reliability_rank
        + 0.30 * margin_rank
    )

    return methods


def evaluate_method(
    name,
    score,
    teacher_prediction,
    target_labels,
):
    correctness = (
        teacher_prediction
        == target_labels
    ).astype(
        np.int64
    )

    return {
        "name": name,
        "correctness_auc": auc_binary(
            score,
            correctness,
        ),
        "correctness_ap": average_precision(
            score,
            correctness,
        ),
        "matched_coverage": top_fraction_rows(
            score,
            correctness,
        ),
    }


def evaluate_disagreement(
    teacher_prediction,
    support_top1,
    teacher_confidence,
    support_on_teacher,
    target_labels,
):
    agreement = (
        teacher_prediction
        == support_top1
    )

    teacher_correct = (
        teacher_prediction
        == target_labels
    )

    results = []

    for state_name, mask in [
        (
            "agreement",
            agreement,
        ),
        (
            "disagreement",
            ~agreement,
        ),
    ]:
        count = int(
            mask.sum()
        )

        if count == 0:
            accuracy = 0.0
            confidence = 0.0
            support = 0.0
        else:
            accuracy = float(
                teacher_correct[
                    mask
                ].mean()
            )

            confidence = float(
                teacher_confidence[
                    mask
                ].mean()
            )

            support = float(
                support_on_teacher[
                    mask
                ].mean()
            )

        results.append(
            {
                "state": state_name,
                "count": count,
                "fraction": float(
                    count
                    / len(
                        teacher_prediction
                    )
                ),
                "teacher_accuracy": accuracy,
                "mean_teacher_confidence": confidence,
                "mean_support_on_teacher": support,
            }
        )

    return results


def truck_analysis(
    teacher_prediction,
    teacher_confidence,
    support_probabilities,
    support_top1,
    support_reliability,
    target_labels,
):
    truck_mask = (
        target_labels
        == TRUCK
    )

    result = {
        "true_truck_samples": int(
            truck_mask.sum()
        ),
    }

    if not truck_mask.any():
        return result

    truck_support_probability = (
        support_probabilities[
            truck_mask,
            TRUCK,
        ]
    )

    truck_teacher_prediction = (
        teacher_prediction[
            truck_mask
        ]
    )

    truck_support_prediction = (
        support_top1[
            truck_mask
        ]
    )

    truck_teacher_confidence = (
        teacher_confidence[
            truck_mask
        ]
    )

    truck_support_reliability = (
        support_reliability[
            truck_mask
        ]
    )

    result.update(
        {
            "teacher_truck_accuracy": float(
                np.mean(
                    truck_teacher_prediction
                    == TRUCK
                )
            ),
            "support_truck_top1_accuracy": float(
                np.mean(
                    truck_support_prediction
                    == TRUCK
                )
            ),
            "teacher_confidence_mean": float(
                truck_teacher_confidence.mean()
            ),
            "support_truck_probability_mean": float(
                truck_support_probability.mean()
            ),
            "support_reliability_mean": float(
                truck_support_reliability.mean()
            ),
            "truck_support_auc": auc_binary(
                support_probabilities[
                    :,
                    TRUCK,
                ],
                (
                    target_labels
                    == TRUCK
                ).astype(
                    np.int64
                ),
            ),
        }
    )

    return result


def per_class_analysis(
    teacher_prediction,
    support_top1,
    target_labels,
):
    result = {}

    for class_id, class_name in enumerate(
        CLASSES
    ):
        mask = (
            target_labels
            == class_id
        )

        teacher_accuracy = float(
            np.mean(
                teacher_prediction[
                    mask
                ]
                == class_id
            )
        )

        support_accuracy = float(
            np.mean(
                support_top1[
                    mask
                ]
                == class_id
            )
        )

        agreement_correct = float(
            np.mean(
                (
                    (
                        teacher_prediction[
                            mask
                        ]
                        == class_id
                    )
                    & (
                        support_top1[
                            mask
                        ]
                        == class_id
                    )
                )
            )
        )

        result[
            class_name
        ] = {
            "samples": int(
                mask.sum()
            ),
            "teacher_accuracy": teacher_accuracy,
            "support_accuracy": support_accuracy,
            "agreement_correct": agreement_correct,
        }

    return result


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "VISDA-2017 MATCHED-COVERAGE PSEUDO-LABEL ADMISSION AUDIT"
    )
    print("=" * 100)
    print(
        "device=cpu"
    )
    print(
        f"seed={args.seed}"
    )

    print(
        "\nLoading frozen independent-support artifact..."
    )

    arrays = load_support_artifact(
        args.support_output
    )

    n_target = validate_arrays(
        arrays
    )

    print(
        f"target_samples={n_target}"
    )

    support_probabilities = np.asarray(
        arrays[
            "support_probabilities"
        ],
        dtype=np.float64,
    )

    if support_probabilities.shape != (
        n_target,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            "support_probabilities must have shape "
            f"({n_target}, {NUM_CLASSES})"
        )

    teacher_probabilities = np.asarray(
        arrays[
            "teacher_probabilities"
        ],
        dtype=np.float64,
    )

    if teacher_probabilities.shape != (
        n_target,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            "teacher_probabilities must have shape "
            f"({n_target}, {NUM_CLASSES})"
        )

    teacher_prediction = np.asarray(
        arrays[
            "teacher_prediction"
        ],
        dtype=np.int64,
    )

    teacher_confidence = np.asarray(
        arrays[
            "teacher_confidence"
        ],
        dtype=np.float64,
    )

    support_top1 = np.asarray(
        arrays[
            "support_top1"
        ],
        dtype=np.int64,
    )

    support_top1_score = np.asarray(
        arrays[
            "support_top1_score"
        ],
        dtype=np.float64,
    )

    support_margin = np.asarray(
        arrays[
            "support_margin"
        ],
        dtype=np.float64,
    )

    support_reliability = np.asarray(
        arrays[
            "support_reliability"
        ],
        dtype=np.float64,
    )

    agreement = (
        teacher_prediction
        == support_top1
    ).astype(
        np.float64
    )

    support_on_teacher = (
        support_probabilities[
            np.arange(
                n_target
            ),
            teacher_prediction,
        ]
    )

    if not np.allclose(
        agreement,
        np.asarray(
            arrays[
                "teacher_support_agreement"
            ],
            dtype=np.float64,
        ),
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Stored agreement does not match reconstructed agreement"
        )

    print(
        "Frozen teacher and support signals verified."
    )

    print(
        "\nBuilding matched-coverage selector scores..."
    )

    methods = build_methods(
        teacher_confidence,
        support_on_teacher,
        support_margin,
        support_reliability,
        agreement,
    )

    print(
        "All selector scores are now frozen."
    )

    print(
        "No target labels have been loaded."
    )

    print(
        "\nLoading target labels for evaluation only..."
    )

    target_labels = load_target_labels(
        args.target_cache
    )

    if len(target_labels) != n_target:
        raise RuntimeError(
            f"Target label count {len(target_labels)} "
            f"does not match support artifact count {n_target}"
        )

    teacher_correct = (
        teacher_prediction
        == target_labels
    ).astype(
        np.int64
    )

    support_correct = (
        support_top1
        == target_labels
    ).astype(
        np.int64
    )

    print(
        "\nBASELINE TEACHER"
    )

    print(
        f"overall_accuracy={teacher_correct.mean() * 100:.2f}%"
    )

    print(
        f"mean_teacher_confidence={teacher_confidence.mean():.6f}"
    )

    print(
        f"support_top1_accuracy={support_correct.mean() * 100:.2f}%"
    )

    print(
        f"support_top1_mean_probability={support_top1_score.mean():.6f}"
    )

    print(
        f"support_on_teacher_mean={support_on_teacher.mean():.6f}"
    )

    print(
        f"support_margin_mean={support_margin.mean():.6f}"
    )

    print(
        f"support_reliability_mean={support_reliability.mean():.6f}"
    )

    print(
        f"teacher_support_agreement={agreement.mean() * 100:.2f}%"
    )

    print(
        "\nMATCHED-COVERAGE RESULTS"
    )

    evaluations = {}

    for name, score in methods.items():
        result = evaluate_method(
            name,
            score,
            teacher_prediction,
            target_labels,
        )

        evaluations[
            name
        ] = result

        print(
            f"\n{name}"
        )

        print(
            f"AUC_correctness={result['correctness_auc']}"
        )

        print(
            f"AP_correctness={result['correctness_ap']}"
        )

        for row in result[
            "matched_coverage"
        ]:
            print(
                f"coverage={row['coverage'] * 100:5.1f}% "
                f"selected={row['selected']:6d} "
                f"precision={row['precision'] * 100:6.2f}%"
            )

    print(
        "\nBEST METHOD AT EACH MATCHED COVERAGE"
    )

    best_by_coverage = {}

    for coverage in COVERAGES:
        candidates = []

        for name, result in evaluations.items():
            for row in result[
                "matched_coverage"
            ]:
                if abs(
                    row["coverage"]
                    - coverage
                ) < 1e-12:
                    candidates.append(
                        (
                            row["precision"],
                            name,
                        )
                    )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        best_precision, best_name = (
            candidates[0]
        )

        best_by_coverage[
            str(coverage)
        ] = {
            "method": best_name,
            "precision": float(
                best_precision
            ),
        }

        print(
            f"coverage={coverage * 100:5.1f}% "
            f"best={best_name} "
            f"precision={best_precision * 100:.2f}%"
        )

    print(
        "\nAGREEMENT / DISAGREEMENT ANALYSIS"
    )

    disagreement_results = evaluate_disagreement(
        teacher_prediction,
        support_top1,
        teacher_confidence,
        support_on_teacher,
        target_labels,
    )

    for row in disagreement_results:
        print(
            f"{row['state']:12s} "
            f"count={row['count']:6d} "
            f"fraction={row['fraction'] * 100:6.2f}% "
            f"teacher_accuracy={row['teacher_accuracy'] * 100:6.2f}% "
            f"confidence={row['mean_teacher_confidence']:.6f} "
            f"support={row['mean_support_on_teacher']:.6f}"
        )

    print(
        "\nTRUCK ANALYSIS"
    )

    truck_result = truck_analysis(
        teacher_prediction,
        teacher_confidence,
        support_probabilities,
        support_top1,
        support_reliability,
        target_labels,
    )

    print(
        f"true_truck_samples={truck_result['true_truck_samples']}"
    )

    for key, value in truck_result.items():
        if key == "true_truck_samples":
            continue

        print(
            f"{key}={value}"
        )

    print(
        "\nPER-CLASS COMPARISON"
    )

    per_class = per_class_analysis(
        teacher_prediction,
        support_top1,
        target_labels,
    )

    for class_name, row in per_class.items():
        print(
            f"{class_name:12s} "
            f"teacher={row['teacher_accuracy'] * 100:6.2f}% "
            f"support={row['support_accuracy'] * 100:6.2f}% "
            f"agreement_correct={row['agreement_correct'] * 100:6.2f}%"
        )

    print(
        "\nSCIENTIFIC COMPARISON"
    )

    teacher_auc = evaluations[
        "teacher_confidence"
    ][
        "correctness_auc"
    ]

    support_auc = evaluations[
        "independent_support"
    ][
        "correctness_auc"
    ]

    margin_auc = evaluations[
        "support_margin"
    ][
        "correctness_auc"
    ]

    teacher_ap = evaluations[
        "teacher_confidence"
    ][
        "correctness_ap"
    ]

    support_ap = evaluations[
        "independent_support"
    ][
        "correctness_ap"
    ]

    margin_ap = evaluations[
        "support_margin"
    ][
        "correctness_ap"
    ]

    auc_gain = (
        support_auc - teacher_auc
        if (
            support_auc is not None
            and teacher_auc is not None
        )
        else None
    )

    margin_auc_gain = (
        margin_auc - teacher_auc
        if (
            margin_auc is not None
            and teacher_auc is not None
        )
        else None
    )

    ap_gain = (
        support_ap - teacher_ap
    )

    margin_ap_gain = (
        margin_ap - teacher_ap
    )

    print(
        f"teacher_confidence_AUC={teacher_auc}"
    )

    print(
        f"independent_support_AUC={support_auc}"
    )

    print(
        f"support_margin_AUC={margin_auc}"
    )

    print(
        f"support_AUC_gain={auc_gain}"
    )

    print(
        f"support_margin_AUC_gain={margin_auc_gain}"
    )

    print(
        f"teacher_confidence_AP={teacher_ap}"
    )

    print(
        f"independent_support_AP={support_ap}"
    )

    print(
        f"support_margin_AP={margin_ap}"
    )

    print(
        f"support_AP_gain={ap_gain}"
    )

    print(
        f"support_margin_AP_gain={margin_ap_gain}"
    )

    if (
        auc_gain is not None
        and auc_gain > 0
        and margin_auc_gain is not None
        and margin_auc_gain > 0
    ):
        decision = (
            "independent_support_adds_reliability_information"
        )
    elif (
        auc_gain is not None
        and auc_gain > 0
    ):
        decision = (
            "independent_support_adds_partial_reliability_information"
        )
    else:
        decision = (
            "independent_support_does_not_improve_correctness_ranking"
        )

    print(
        f"scientific_decision={decision}"
    )

    output_npz = (
        args.output_dir
        / f"matched_coverage_support_scores_seed{args.seed}.npz"
    )

    save_arrays = {
        "teacher_prediction": teacher_prediction,
        "teacher_confidence": teacher_confidence,
        "support_on_teacher": support_on_teacher,
        "support_margin": support_margin,
        "support_reliability": support_reliability,
        "agreement": agreement,
    }

    for name, score in methods.items():
        save_arrays[
            name
        ] = score

    np.savez_compressed(
        output_npz,
        **save_arrays,
    )

    output_json = (
        args.output_dir
        / f"matched_coverage_support_audit_seed{args.seed}.json"
    )

    summary = {
        "experiment": (
            "visda_matched_coverage_pseudolabel_admission_audit"
        ),
        "seed": int(
            args.seed
        ),
        "device": "cpu",
        "target_samples": int(
            n_target
        ),
        "constraints": {
            "model_training": False,
            "rpc_frozen": True,
            "support_artifact_frozen": True,
            "target_labels_used_for_selector_construction": False,
            "target_labels_used_for_score_construction": False,
            "target_labels_loaded_only_after_scores_frozen": True,
        },
        "inputs": {
            "support_output": str(
                args.support_output
            ),
            "target_cache": str(
                args.target_cache
            ),
        },
        "baseline": {
            "teacher_accuracy": float(
                teacher_correct.mean()
            ),
            "mean_teacher_confidence": float(
                teacher_confidence.mean()
            ),
            "support_top1_accuracy": float(
                support_correct.mean()
            ),
            "support_top1_mean_probability": float(
                support_top1_score.mean()
            ),
            "support_on_teacher_mean": float(
                support_on_teacher.mean()
            ),
            "support_margin_mean": float(
                support_margin.mean()
            ),
            "support_reliability_mean": float(
                support_reliability.mean()
            ),
            "teacher_support_agreement": float(
                agreement.mean()
            ),
        },
        "methods": evaluations,
        "best_by_coverage": best_by_coverage,
        "agreement_disagreement": disagreement_results,
        "truck_analysis": truck_result,
        "per_class": per_class,
        "scientific_comparison": {
            "teacher_confidence_AUC": teacher_auc,
            "independent_support_AUC": support_auc,
            "support_margin_AUC": margin_auc,
            "support_AUC_gain": auc_gain,
            "support_margin_AUC_gain": margin_auc_gain,
            "teacher_confidence_AP": teacher_ap,
            "independent_support_AP": support_ap,
            "support_margin_AP": margin_ap,
            "support_AP_gain": ap_gain,
            "support_margin_AP_gain": margin_ap_gain,
            "decision": decision,
        },
        "outputs": {
            "npz": str(
                output_npz
            ),
            "json": str(
                output_json
            ),
        },
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "\nOUTPUTS"
    )

    print(
        f"npz={output_npz}"
    )

    print(
        f"json={output_json}"
    )

    print(
        "\nMATCHED-COVERAGE PSEUDO-LABEL ADMISSION AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()