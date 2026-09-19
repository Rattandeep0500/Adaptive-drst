from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
FEATURE_FILE = DATA_ROOT / "features" / "realmat_bag_structure_features.parquet"

OUTPUT_DIR = DATA_ROOT / "features"
CHECKPOINT_DIR = ROOT / "checkpoints" / "realmat_bag_drst"

RESULT_FILE = OUTPUT_DIR / "realmat_bag_drst_results.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "drst_seed42.pt"

THRESHOLD = 1.5
RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE_SOURCE = 128
BATCH_SIZE_TARGET = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TARGET_LOSS_WEIGHT = 1.0

PSEUDO_LABEL_THRESHOLD = 0.90

SOURCE_RATIO_MIN = 0.10
SOURCE_RATIO_MAX = 10.0

SUPPORT_RATIO_MIN = 0.10
SUPPORT_RATIO_MAX = 10.0

HIDDEN_1 = 128
HIDDEN_2 = 64
DROPOUT = 0.10

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
    torch.manual_seed(seed)


class MaterialClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(HIDDEN_2, 2)

    def forward(self, x):
        features = self.network(x)
        logits = self.classifier(features)
        return logits


def prepare(df):
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    y = (
        df["bandgap"].to_numpy(dtype=np.float32) >= THRESHOLD
    ).astype(np.int64)

    return x, y


def calculate_metrics(y_true, y_pred):
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
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def class_weights(y):
    counts = np.bincount(
        y,
        minlength=2,
    ).astype(np.float64)

    weights = np.zeros(
        2,
        dtype=np.float32,
    )

    for index in range(2):
        if counts[index] > 0:
            weights[index] = (
                len(y)
                / (
                    2.0
                    * counts[index]
                )
            )
        else:
            weights[index] = 1.0

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def estimate_density_ratios(x_source, x_target):
    x_domain = np.vstack(
        [
            x_source,
            x_target,
        ]
    )

    y_domain = np.concatenate(
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

    domain_model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    domain_model.fit(
        x_domain,
        y_domain,
    )

    target_probability_source = domain_model.predict_proba(
        x_source
    )[:, 1]

    target_probability_target = domain_model.predict_proba(
        x_target
    )[:, 1]

    eps = 1e-6

    source_probability = np.clip(
        1.0 - target_probability_source,
        eps,
        1.0 - eps,
    )

    target_probability = np.clip(
        target_probability_source,
        eps,
        1.0 - eps,
    )

    n_source = len(x_source)
    n_target = len(x_target)

    source_prior = (
        n_source
        / (
            n_source
            + n_target
        )
    )

    target_prior = (
        n_target
        / (
            n_source
            + n_target
        )
    )

    ratio_target_over_source = (
        (
            target_probability
            / source_probability
        )
        * (
            source_prior
            / target_prior
        )
    )

    source_support = 1.0 / np.maximum(
        ratio_target_over_source,
        eps,
    )

    target_source_probability = np.clip(
        1.0 - target_probability_target,
        eps,
        1.0 - eps,
    )

    target_target_probability = np.clip(
        target_probability_target,
        eps,
        1.0 - eps,
    )

    ratio_source_over_target = (
        (
            target_source_probability
            / target_target_probability
        )
        * (
            target_prior
            / source_prior
        )
    )

    return (
        ratio_target_over_source,
        ratio_source_over_target,
        domain_model,
    )


def normalize_positive_weights(weights):
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    mean = np.mean(weights)

    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(
            weights,
            dtype=np.float64,
        )

    return weights / mean


def weighted_cross_entropy(
    logits,
    labels,
    weights,
    criterion,
):
    losses = nn.functional.cross_entropy(
        logits,
        labels,
        weight=criterion.weight,
        reduction="none",
    )

    weights = weights.to(
        losses.device
    )

    return torch.sum(
        losses * weights
    ) / torch.clamp(
        torch.sum(weights),
        min=1e-8,
    )


def evaluate(
    model,
    x,
    y,
):
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(x)

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        ).cpu().numpy()

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "metrics": calculate_metrics(
            y,
            predictions,
        ),
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
        ).tolist(),
    }


def main():
    set_seed(
        RANDOM_STATE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("REALMAT-BAG DRST-MAT")
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

    print(
        f"COMPUTATIONAL SAMPLES: "
        f"{len(source)}"
    )

    print(
        f"EXPERIMENTAL TRAIN SAMPLES: "
        f"{len(target_train)}"
    )

    print(
        f"EXPERIMENTAL TEST SAMPLES: "
        f"{len(target_test)}"
    )

    print(
        f"FEATURES: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"THRESHOLD: "
        f"{THRESHOLD:.3f} eV"
    )

    print(
        f"PSEUDO-LABEL THRESHOLD: "
        f"{PSEUDO_LABEL_THRESHOLD:.3f}"
    )

    x_source, y_source = prepare(
        source
    )

    x_target_train, y_target_train = prepare(
        target_train
    )

    x_target_test, y_target_test = prepare(
        target_test
    )

    scaler = StandardScaler()

    x_source = scaler.fit_transform(
        x_source
    ).astype(np.float32)

    x_target_train = scaler.transform(
        x_target_train
    ).astype(np.float32)

    x_target_test = scaler.transform(
        x_target_test
    ).astype(np.float32)

    print()
    print("=" * 80)
    print("DENSITY-RATIO ESTIMATION")
    print("=" * 80)

    (
        source_target_ratio,
        target_source_ratio,
        domain_model,
    ) = estimate_density_ratios(
        x_source,
        x_target_train,
    )

    source_target_ratio = np.clip(
        source_target_ratio,
        SOURCE_RATIO_MIN,
        SOURCE_RATIO_MAX,
    )

    target_source_ratio = np.clip(
        target_source_ratio,
        SUPPORT_RATIO_MIN,
        SUPPORT_RATIO_MAX,
    )

    source_target_ratio = normalize_positive_weights(
        source_target_ratio
    )

    target_source_ratio = normalize_positive_weights(
        target_source_ratio
    )

    print(
        f"SOURCE WEIGHT MEAN: "
        f"{np.mean(source_target_ratio):.6f}"
    )

    print(
        f"SOURCE WEIGHT MIN: "
        f"{np.min(source_target_ratio):.6f}"
    )

    print(
        f"SOURCE WEIGHT MAX: "
        f"{np.max(source_target_ratio):.6f}"
    )

    print(
        f"TARGET SUPPORT WEIGHT MEAN: "
        f"{np.mean(target_source_ratio):.6f}"
    )

    print(
        f"TARGET SUPPORT WEIGHT MIN: "
        f"{np.min(target_source_ratio):.6f}"
    )

    print(
        f"TARGET SUPPORT WEIGHT MAX: "
        f"{np.max(target_source_ratio):.6f}"
    )

    print()
    print("=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    model = MaterialClassifier(
        input_dim=len(FEATURE_COLUMNS)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            y_source
        )
    )

    source_tensor = torch.from_numpy(
        x_source
    )

    source_label_tensor = torch.from_numpy(
        y_source
    )

    source_weight_tensor = torch.from_numpy(
        source_target_ratio.astype(
            np.float32
        )
    )

    target_tensor = torch.from_numpy(
        x_target_train
    )

    rng_source = np.random.default_rng(
        RANDOM_STATE
    )

    rng_target = np.random.default_rng(
        RANDOM_STATE + 1
    )

    history = []

    for epoch in range(
        EPOCHS
    ):
        model.train()

        source_indices = rng_source.choice(
            len(source_tensor),
            size=min(
                BATCH_SIZE_SOURCE,
                len(source_tensor),
            ),
            replace=False,
        )

        target_indices = rng_target.choice(
            len(target_tensor),
            size=min(
                BATCH_SIZE_TARGET,
                len(target_tensor),
            ),
            replace=False,
        )

        source_batch = source_tensor[
            source_indices
        ]

        source_labels_batch = source_label_tensor[
            source_indices
        ]

        source_weights_batch = source_weight_tensor[
            source_indices
        ]

        target_batch = target_tensor[
            target_indices
        ]

        optimizer.zero_grad()

        source_logits = model(
            source_batch
        )

        target_logits = model(
            target_batch
        )

        source_loss = weighted_cross_entropy(
            source_logits,
            source_labels_batch,
            source_weights_batch,
            criterion,
        )

        target_probabilities = torch.softmax(
            target_logits,
            dim=1,
        )

        target_confidence, target_pseudo_labels = torch.max(
            target_probabilities,
            dim=1,
        )

        confidence_mask = (
            target_confidence
            >= PSEUDO_LABEL_THRESHOLD
        )

        target_support_weights = torch.from_numpy(
            target_source_ratio[
                target_indices
            ].astype(
                np.float32
            )
        )

        target_loss = torch.tensor(
            0.0,
            dtype=torch.float32,
        )

        selected_count = int(
            confidence_mask.sum().item()
        )

        if selected_count > 0:
            selected_logits = target_logits[
                confidence_mask
            ]

            selected_labels = target_pseudo_labels[
                confidence_mask
            ]

            selected_weights = target_support_weights[
                confidence_mask.cpu()
            ]

            selected_ce = nn.functional.cross_entropy(
                selected_logits,
                selected_labels,
                reduction="none",
            )

            target_loss = torch.sum(
                selected_ce
                * selected_weights
            ) / torch.clamp(
                torch.sum(
                    selected_weights
                ),
                min=1e-8,
            )

        total_loss = (
            source_loss
            + TARGET_LOSS_WEIGHT
            * target_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        record = {
            "epoch": epoch + 1,
            "source_loss": float(
                source_loss.item()
            ),
            "target_loss": float(
                target_loss.item()
            ),
            "total_loss": float(
                total_loss.item()
            ),
            "pseudo_label_count": selected_count,
        }

        history.append(
            record
        )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"EPOCH {epoch + 1:03d} "
                f"SOURCE={record['source_loss']:.6f} "
                f"TARGET={record['target_loss']:.6f} "
                f"TOTAL={record['total_loss']:.6f} "
                f"PSEUDO={selected_count}"
            )

    print()
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    source_result = evaluate(
        model,
        x_source,
        y_source,
    )

    target_train_result = evaluate(
        model,
        x_target_train,
        y_target_train,
    )

    target_test_result = evaluate(
        model,
        x_target_test,
        y_target_test,
    )

    print()
    print("SOURCE TRAIN")

    for key, value in source_result[
        "metrics"
    ].items():
        print(
            f"{key.upper()}: "
            f"{value:.6f}"
        )

    print()
    print("EXPERIMENTAL TRAIN")

    for key, value in target_train_result[
        "metrics"
    ].items():
        print(
            f"{key.upper()}: "
            f"{value:.6f}"
        )

    print()
    print("EXPERIMENTAL TEST")

    for key, value in target_test_result[
        "metrics"
    ].items():
        print(
            f"{key.upper()}: "
            f"{value:.6f}"
        )

    print()
    print(
        "EXPERIMENTAL TEST CONFUSION MATRIX"
    )

    print(
        np.asarray(
            target_test_result[
                "confusion_matrix"
            ]
        )
    )

    final_pseudo = history[-1][
        "pseudo_label_count"
    ]

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "epochs": EPOCHS,
        "batch_size_source": BATCH_SIZE_SOURCE,
        "batch_size_target": BATCH_SIZE_TARGET,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "target_loss_weight": TARGET_LOSS_WEIGHT,
        "pseudo_label_threshold": PSEUDO_LABEL_THRESHOLD,
        "source_ratio_min": SOURCE_RATIO_MIN,
        "source_ratio_max": SOURCE_RATIO_MAX,
        "support_ratio_min": SUPPORT_RATIO_MIN,
        "support_ratio_max": SUPPORT_RATIO_MAX,
        "history": history,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_FILE,
    )

    results = {
        "experiment": "realmat_bag_drst_mat_v1",
        "threshold_ev": THRESHOLD,
        "random_state": RANDOM_STATE,
        "epochs": EPOCHS,
        "batch_size_source": BATCH_SIZE_SOURCE,
        "batch_size_target": BATCH_SIZE_TARGET,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "target_loss_weight": TARGET_LOSS_WEIGHT,
        "pseudo_label_threshold": PSEUDO_LABEL_THRESHOLD,
        "source_ratio_min": SOURCE_RATIO_MIN,
        "source_ratio_max": SOURCE_RATIO_MAX,
        "support_ratio_min": SUPPORT_RATIO_MIN,
        "support_ratio_max": SUPPORT_RATIO_MAX,
        "source_samples": len(source),
        "target_train_samples": len(target_train),
        "target_test_samples": len(target_test),
        "source_weight_statistics": {
            "mean": float(
                np.mean(
                    source_target_ratio
                )
            ),
            "std": float(
                np.std(
                    source_target_ratio
                )
            ),
            "min": float(
                np.min(
                    source_target_ratio
                )
            ),
            "max": float(
                np.max(
                    source_target_ratio
                )
            ),
        },
        "target_support_statistics": {
            "mean": float(
                np.mean(
                    target_source_ratio
                )
            ),
            "std": float(
                np.std(
                    target_source_ratio
                )
            ),
            "min": float(
                np.min(
                    target_source_ratio
                )
            ),
            "max": float(
                np.max(
                    target_source_ratio
                )
            ),
        },
        "final_pseudo_label_count": int(
            final_pseudo
        ),
        "source_train": source_result[
            "metrics"
        ],
        "experimental_train": target_train_result[
            "metrics"
        ],
        "experimental_test": target_test_result[
            "metrics"
        ],
        "experimental_test_confusion_matrix": target_test_result[
            "confusion_matrix"
        ],
        "history": history,
        "checkpoint": str(
            CHECKPOINT_FILE
        ),
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
    print("DRST-MAT COMPLETE")
    print("=" * 80)
    print(
        f"FINAL PSEUDO-LABEL COUNT: "
        f"{final_pseudo}"
    )
    print(
        f"CHECKPOINT: "
        f"{CHECKPOINT_FILE}"
    )
    print(
        f"RESULTS: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()