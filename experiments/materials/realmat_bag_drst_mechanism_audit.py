from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
FEATURE_FILE = DATA_ROOT / "features" / "realmat_bag_structure_features.parquet"

OUTPUT_DIR = DATA_ROOT / "features"
RESULT_FILE = OUTPUT_DIR / "realmat_bag_drst_mechanism_audit.json"

THRESHOLD = 1.5
RANDOM_STATE = 42

PSEUDO_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
]

FEATURE_COLUMNS = [
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "cell_volume",
    "density",
    "n_elements",
    "total_atoms",
    "mean_atomic_number",
    "min_atomic_number",
    "max_atomic_number",
    "mean_atomic_mass",
    "min_atomic_mass",
    "max_atomic_mass",
    "composition_entropy",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def prepare_labels(df):
    return (
        df["bandgap"].to_numpy(dtype=np.float64) >= THRESHOLD
    ).astype(np.int64)


def density_ratios(x_source, x_target):
    x = np.vstack(
        [
            x_source,
            x_target,
        ]
    )

    y = np.concatenate(
        [
            np.zeros(len(x_source), dtype=np.int64),
            np.ones(len(x_target), dtype=np.int64),
        ]
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(x, y)

    source_target_probability = model.predict_proba(x_source)[:, 1]
    target_target_probability = model.predict_proba(x_target)[:, 1]

    eps = 1e-8

    source_probability = np.clip(
        1.0 - source_target_probability,
        eps,
        1.0 - eps,
    )

    target_probability = np.clip(
        source_target_probability,
        eps,
        1.0 - eps,
    )

    target_prior = len(x_target) / (
        len(x_source) + len(x_target)
    )

    source_prior = len(x_source) / (
        len(x_source) + len(x_target)
    )

    source_to_target = (
        target_probability
        / source_probability
    ) * (
        source_prior
        / target_prior
    )

    target_source_probability = np.clip(
        1.0 - target_target_probability,
        eps,
        1.0 - eps,
    )

    target_probability_2 = np.clip(
        target_target_probability,
        eps,
        1.0 - eps,
    )

    target_to_source = (
        target_source_probability
        / target_probability_2
    ) * (
        target_prior
        / source_prior
    )

    return (
        source_to_target,
        target_to_source,
        model,
    )


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def main():
    set_seed(RANDOM_STATE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("REALMAT-BAG DRST MECHANISM AUDIT")
    print("=" * 80)

    df = pd.read_parquet(
        FEATURE_FILE
    )

    source = df[
        df["domain"] == "computational"
    ].copy()

    target_train = df[
        df["domain"] == "experimental_train"
    ].copy()

    target_test = df[
        df["domain"] == "experimental_test"
    ].copy()

    x_source_raw = source[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    x_target_raw = target_train[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    x_test_raw = target_test[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    y_source = prepare_labels(source)
    y_target_train = prepare_labels(target_train)
    y_target_test = prepare_labels(target_test)

    scaler = StandardScaler()

    x_source = scaler.fit_transform(
        x_source_raw
    )

    x_target = scaler.transform(
        x_target_raw
    )

    x_test = scaler.transform(
        x_test_raw
    )

    print()
    print("=" * 80)
    print("DOMAIN DISCRIMINATOR")
    print("=" * 80)

    source_to_target, target_to_source, domain_model = density_ratios(
        x_source,
        x_target,
    )

    domain_source_probability = domain_model.predict_proba(
        x_source
    )[:, 1]

    domain_target_probability = domain_model.predict_proba(
        x_target
    )[:, 1]

    domain_labels = np.concatenate(
        [
            np.zeros(
                len(x_source),
                dtype=np.int64,
            ),
            np.ones(
                len(x_target),
                dtype=np.int64,
            ),
        ]
    )

    domain_predictions = np.concatenate(
        [
            (domain_source_probability >= 0.5).astype(np.int64),
            (domain_target_probability >= 0.5).astype(np.int64),
        ]
    )

    domain_metrics = classification_metrics(
        domain_labels,
        domain_predictions,
    )

    domain_auc = roc_auc_score(
        domain_labels,
        np.concatenate(
            [
                domain_source_probability,
                domain_target_probability,
            ]
        ),
    )

    domain_ap = average_precision_score(
        domain_labels,
        np.concatenate(
            [
                domain_source_probability,
                domain_target_probability,
            ]
        ),
    )

    print(
        f"DOMAIN ACCURACY: "
        f"{domain_metrics['accuracy']:.6f}"
    )

    print(
        f"DOMAIN BALANCED ACCURACY: "
        f"{domain_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"DOMAIN ROC AUC: "
        f"{domain_auc:.6f}"
    )

    print(
        f"DOMAIN AP: "
        f"{domain_ap:.6f}"
    )

    print()
    print("SOURCE-TO-TARGET RATIO")

    print(
        f"MEAN: {np.mean(source_to_target):.6f}"
    )

    print(
        f"STD: {np.std(source_to_target):.6f}"
    )

    print(
        f"MIN: {np.min(source_to_target):.6f}"
    )

    print(
        f"MAX: {np.max(source_to_target):.6f}"
    )

    print()
    print("TARGET-TO-SOURCE SUPPORT RATIO")

    print(
        f"MEAN: {np.mean(target_to_source):.6f}"
    )

    print(
        f"STD: {np.std(target_to_source):.6f}"
    )

    print(
        f"MIN: {np.min(target_to_source):.6f}"
    )

    print(
        f"MAX: {np.max(target_to_source):.6f}"
    )

    print()
    print("=" * 80)
    print("PRE-ADAPTATION TARGET CLASSIFIER")
    print("=" * 80)

    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    classifier.fit(
        x_source,
        y_source,
    )

    target_probability = classifier.predict_proba(
        x_target
    )

    test_probability = classifier.predict_proba(
        x_test
    )

    target_prediction = np.argmax(
        target_probability,
        axis=1,
    )

    test_prediction = np.argmax(
        test_probability,
        axis=1,
    )

    train_metrics = classification_metrics(
        y_target_train,
        target_prediction,
    )

    test_metrics = classification_metrics(
        y_target_test,
        test_prediction,
    )

    print("EXPERIMENTAL TRAIN")

    for key, value in train_metrics.items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    print()
    print("EXPERIMENTAL TEST")

    for key, value in test_metrics.items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    print()
    print("=" * 80)
    print("PSEUDO-LABEL COVERAGE")
    print("=" * 80)

    target_confidence = np.max(
        target_probability,
        axis=1,
    )

    target_correct = (
        target_prediction
        == y_target_train
    ).astype(np.int64)

    confidence_auc = roc_auc_score(
        target_correct,
        target_confidence,
    )

    confidence_ap = average_precision_score(
        target_correct,
        target_confidence,
    )

    print(
        f"CONFIDENCE CORRECTNESS ROC AUC: "
        f"{confidence_auc:.6f}"
    )

    print(
        f"CONFIDENCE CORRECTNESS AP: "
        f"{confidence_ap:.6f}"
    )

    pseudo_rows = []

    for threshold in PSEUDO_THRESHOLDS:
        mask = (
            target_confidence
            >= threshold
        )

        count = int(
            np.sum(mask)
        )

        coverage = (
            count
            / len(target_confidence)
        )

        if count > 0:
            precision = float(
                np.mean(
                    target_correct[mask]
                )
            )
        else:
            precision = 0.0

        pseudo_rows.append(
            {
                "threshold": threshold,
                "count": count,
                "coverage": float(
                    coverage
                ),
                "pseudo_label_precision": precision,
            }
        )

        print(
            f"THRESHOLD={threshold:.2f} "
            f"COUNT={count} "
            f"COVERAGE={coverage:.6f} "
            f"PRECISION={precision:.6f}"
        )

    print()
    print("=" * 80)
    print("SUPPORT / CORRECTNESS RELATIONSHIP")
    print("=" * 80)

    support_score = 1.0 / np.maximum(
        target_to_source,
        1e-8,
    )

    support_auc = roc_auc_score(
        target_correct,
        support_score,
    )

    support_ap = average_precision_score(
        target_correct,
        support_score,
    )

    print(
        f"SUPPORT SCORE ROC AUC: "
        f"{support_auc:.6f}"
    )

    print(
        f"SUPPORT SCORE AP: "
        f"{support_ap:.6f}"
    )

    print()
    print("=" * 80)
    print("CONFIDENCE + SUPPORT")
    print("=" * 80)

    normalized_support = (
        support_score
        / np.mean(support_score)
    )

    combined_score = (
        target_confidence
        * normalized_support
    )

    combined_auc = roc_auc_score(
        target_correct,
        combined_score,
    )

    combined_ap = average_precision_score(
        target_correct,
        combined_score,
    )

    print(
        f"COMBINED ROC AUC: "
        f"{combined_auc:.6f}"
    )

    print(
        f"COMBINED AP: "
        f"{combined_ap:.6f}"
    )

    combined_rows = []

    for threshold in PSEUDO_THRESHOLDS:
        percentile_mask = (
            combined_score
            >= np.quantile(
                combined_score,
                threshold,
            )
        )

        count = int(
            np.sum(percentile_mask)
        )

        precision = float(
            np.mean(
                target_correct[
                    percentile_mask
                ]
            )
        )

        combined_rows.append(
            {
                "quantile": threshold,
                "count": count,
                "precision": precision,
            }
        )

    print()
    print("=" * 80)
    print("SOURCE WEIGHT QUALITY")
    print("=" * 80)

    source_domain_correct = (
        domain_source_probability < 0.5
    ).astype(np.int64)

    source_weights = (
        source_to_target
        / np.mean(
            source_to_target
        )
    )

    source_weight_auc = roc_auc_score(
        source_domain_correct,
        -source_weights,
    )

    print(
        f"SOURCE WEIGHT / SOURCE-LIKE ROC AUC: "
        f"{source_weight_auc:.6f}"
    )

    low_weight = np.quantile(
        source_weights,
        0.10,
    )

    high_weight = np.quantile(
        source_weights,
        0.90,
    )

    print(
        f"10TH PERCENTILE WEIGHT: "
        f"{low_weight:.6f}"
    )

    print(
        f"90TH PERCENTILE WEIGHT: "
        f"{high_weight:.6f}"
    )

    print()
    print("=" * 80)
    print("BANDGAP-THRESHOLD SENSITIVITY")
    print("=" * 80)

    threshold_results = []

    for threshold in [
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
    ]:
        source_y = (
            source["bandgap"].to_numpy(
                dtype=np.float64
            ) >= threshold
        ).astype(np.int64)

        target_y = (
            target_train["bandgap"].to_numpy(
                dtype=np.float64
            ) >= threshold
        ).astype(np.int64)

        test_y = (
            target_test["bandgap"].to_numpy(
                dtype=np.float64
            ) >= threshold
        ).astype(np.int64)

        source_counts = np.bincount(
            source_y,
            minlength=2,
        )

        target_counts = np.bincount(
            target_y,
            minlength=2,
        )

        test_counts = np.bincount(
            test_y,
            minlength=2,
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "source_class_0": int(
                    source_counts[0]
                ),
                "source_class_1": int(
                    source_counts[1]
                ),
                "target_train_class_0": int(
                    target_counts[0]
                ),
                "target_train_class_1": int(
                    target_counts[1]
                ),
                "target_test_class_0": int(
                    test_counts[0]
                ),
                "target_test_class_1": int(
                    test_counts[1]
                ),
            }
        )

        print(
            f"THRESHOLD={threshold:.2f} "
            f"SOURCE=({source_counts[0]},{source_counts[1]}) "
            f"TARGET_TRAIN=({target_counts[0]},{target_counts[1]}) "
            f"TARGET_TEST=({test_counts[0]},{test_counts[1]})"
        )

    results = {
        "experiment": "realmat_bag_drst_mechanism_audit",
        "threshold_ev": THRESHOLD,
        "random_state": RANDOM_STATE,
        "sample_counts": {
            "source": len(source),
            "target_train": len(target_train),
            "target_test": len(target_test),
        },
        "domain_discriminator": {
            "accuracy": domain_metrics["accuracy"],
            "balanced_accuracy": domain_metrics[
                "balanced_accuracy"
            ],
            "roc_auc": float(domain_auc),
            "average_precision": float(domain_ap),
        },
        "source_target_ratio": {
            "mean": float(
                np.mean(source_to_target)
            ),
            "std": float(
                np.std(source_to_target)
            ),
            "min": float(
                np.min(source_to_target)
            ),
            "max": float(
                np.max(source_to_target)
            ),
        },
        "target_support_ratio": {
            "mean": float(
                np.mean(target_to_source)
            ),
            "std": float(
                np.std(target_to_source)
            ),
            "min": float(
                np.min(target_to_source)
            ),
            "max": float(
                np.max(target_to_source)
            ),
        },
        "pre_adaptation_target_classifier": {
            "experimental_train": train_metrics,
            "experimental_test": test_metrics,
            "confidence_correctness_auc": float(
                confidence_auc
            ),
            "confidence_correctness_ap": float(
                confidence_ap
            ),
        },
        "pseudo_label_thresholds": pseudo_rows,
        "support_quality": {
            "roc_auc": float(
                support_auc
            ),
            "average_precision": float(
                support_ap
            ),
        },
        "combined_confidence_support": {
            "roc_auc": float(
                combined_auc
            ),
            "average_precision": float(
                combined_ap
            ),
        },
        "combined_quantiles": combined_rows,
        "source_weight_quality": {
            "roc_auc": float(
                source_weight_auc
            ),
            "weight_p10": float(
                low_weight
            ),
            "weight_p90": float(
                high_weight
            ),
        },
        "threshold_sensitivity": threshold_results,
    }

    with RESULT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("MECHANISM AUDIT COMPLETE")
    print("=" * 80)
    print(f"RESULTS: {RESULT_FILE}")


if __name__ == "__main__":
    main()