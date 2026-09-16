from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
FEATURE_FILE = DATA_ROOT / "features" / "realmat_bag_structure_features.parquet"
OUTPUT_DIR = DATA_ROOT / "features"

THRESHOLD = 1.5

RANDOM_STATE = 42

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


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def prepare(df):
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = (df["bandgap"].to_numpy(dtype=np.float64) >= THRESHOLD).astype(np.int64)
    return x, y


def train_model(x, y):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(x, y)
    return model


def main():
    print("=" * 80)
    print("REALMAT-BAG BASELINE EXPERIMENT")
    print("=" * 80)

    df = pd.read_parquet(FEATURE_FILE)

    source = df[df["domain"] == "computational"].copy()
    target_train = df[df["domain"] == "experimental_train"].copy()
    target_test = df[df["domain"] == "experimental_test"].copy()

    print(f"COMPUTATIONAL SAMPLES: {len(source)}")
    print(f"EXPERIMENTAL TRAIN SAMPLES: {len(target_train)}")
    print(f"EXPERIMENTAL TEST SAMPLES: {len(target_test)}")
    print(f"CLASS THRESHOLD: {THRESHOLD:.3f} eV")

    x_source, y_source = prepare(source)
    x_target_train, y_target_train = prepare(target_train)
    x_target_test, y_target_test = prepare(target_test)

    print()
    print("=" * 80)
    print("CLASS DISTRIBUTION")
    print("=" * 80)

    for name, y in [
        ("COMPUTATIONAL", y_source),
        ("EXPERIMENTAL TRAIN", y_target_train),
        ("EXPERIMENTAL TEST", y_target_test),
    ]:
        print(name)
        print(f"CLASS 0: {int((y == 0).sum())}")
        print(f"CLASS 1: {int((y == 1).sum())}")

    print()
    print("=" * 80)
    print("SOURCE-ONLY MODEL")
    print("=" * 80)

    source_model = train_model(x_source, y_source)

    source_pred_train = source_model.predict(x_source)
    target_pred_train = source_model.predict(x_target_train)
    target_pred_test = source_model.predict(x_target_test)

    source_metrics = metrics(y_source, source_pred_train)
    target_train_metrics = metrics(y_target_train, target_pred_train)
    target_test_metrics = metrics(y_target_test, target_pred_test)

    print("SOURCE TRAIN")
    for key, value in source_metrics.items():
        print(f"{key.upper()}: {value:.6f}")

    print()
    print("EXPERIMENTAL TRAIN")
    for key, value in target_train_metrics.items():
        print(f"{key.upper()}: {value:.6f}")

    print()
    print("EXPERIMENTAL TEST")
    for key, value in target_test_metrics.items():
        print(f"{key.upper()}: {value:.6f}")

    print()
    print("EXPERIMENTAL TEST CONFUSION MATRIX")
    print(confusion_matrix(y_target_test, target_pred_test))

    print()
    print("=" * 80)
    print("TARGET-SUPERVISED REFERENCE")
    print("=" * 80)

    target_model = train_model(x_target_train, y_target_train)

    target_supervised_pred_train = target_model.predict(x_target_train)
    target_supervised_pred_test = target_model.predict(x_target_test)

    target_supervised_train_metrics = metrics(
        y_target_train,
        target_supervised_pred_train,
    )

    target_supervised_test_metrics = metrics(
        y_target_test,
        target_supervised_pred_test,
    )

    print("EXPERIMENTAL TRAIN")
    for key, value in target_supervised_train_metrics.items():
        print(f"{key.upper()}: {value:.6f}")

    print()
    print("EXPERIMENTAL TEST")
    for key, value in target_supervised_test_metrics.items():
        print(f"{key.upper()}: {value:.6f}")

    results = {
        "threshold_ev": THRESHOLD,
        "source_samples": len(source),
        "target_train_samples": len(target_train),
        "target_test_samples": len(target_test),
        "source_only": {
            "source_train": source_metrics,
            "target_train": target_train_metrics,
            "target_test": target_test_metrics,
        },
        "target_supervised": {
            "target_train": target_supervised_train_metrics,
            "target_test": target_supervised_test_metrics,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "realmat_bag_baseline_results.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("BASELINE COMPLETE")
    print("=" * 80)
    print(f"RESULTS: {output_file}")


if __name__ == "__main__":
    main()