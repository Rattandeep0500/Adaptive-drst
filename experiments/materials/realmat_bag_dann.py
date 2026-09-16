from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
FEATURE_FILE = DATA_ROOT / "features" / "realmat_bag_structure_features.parquet"
OUTPUT_DIR = DATA_ROOT / "features"
CHECKPOINT_DIR = ROOT / "checkpoints" / "realmat_bag_dann"

RESULT_FILE = OUTPUT_DIR / "realmat_bag_dann_results.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "dann_seed42.pt"

THRESHOLD = 1.5
RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DOMAIN_WEIGHT = 1.0

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


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


def gradient_reverse(x, lambda_value):
    return GradientReverse.apply(x, lambda_value)


class DANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(HIDDEN_2, 2)

        self.domain_classifier = nn.Sequential(
            nn.Linear(HIDDEN_2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, lambda_value=1.0):
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        reversed_features = gradient_reverse(
            features,
            lambda_value,
        )
        domain_logits = self.domain_classifier(reversed_features).squeeze(1)
        return class_logits, domain_logits, features


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


def make_class_weights(y):
    counts = np.bincount(y, minlength=2).astype(np.float64)

    weights = np.zeros(2, dtype=np.float32)

    for index in range(2):
        if counts[index] > 0:
            weights[index] = len(y) / (2.0 * counts[index])
        else:
            weights[index] = 1.0

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def evaluate(model, x, y):
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(x)
        logits, _, _ = model(
            x_tensor,
            lambda_value=0.0,
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
    set_seed(RANDOM_STATE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("REALMAT-BAG DANN BASELINE")
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

    print(f"COMPUTATIONAL SAMPLES: {len(source)}")
    print(f"EXPERIMENTAL TRAIN SAMPLES: {len(target_train)}")
    print(f"EXPERIMENTAL TEST SAMPLES: {len(target_test)}")
    print(f"FEATURES: {len(FEATURE_COLUMNS)}")
    print(f"THRESHOLD: {THRESHOLD:.3f} eV")
    print(f"SEED: {RANDOM_STATE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"BATCH SIZE: {BATCH_SIZE}")
    print(f"LEARNING RATE: {LEARNING_RATE}")
    print(f"DOMAIN WEIGHT: {DOMAIN_WEIGHT}")

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

    source_tensor = torch.from_numpy(
        x_source
    )

    source_labels = torch.from_numpy(
        y_source
    )

    target_tensor = torch.from_numpy(
        x_target_train
    )

    domain_source = torch.zeros(
        len(source_tensor),
        dtype=torch.float32,
    )

    domain_target = torch.ones(
        len(target_tensor),
        dtype=torch.float32,
    )

    class_weights = make_class_weights(
        y_source
    )

    model = DANN(
        input_dim=len(FEATURE_COLUMNS)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    classification_loss = nn.CrossEntropyLoss(
        weight=class_weights
    )

    domain_loss = nn.BCEWithLogitsLoss()

    source_rng = np.random.default_rng(
        RANDOM_STATE
    )

    target_rng = np.random.default_rng(
        RANDOM_STATE + 1
    )

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    history = []

    source_batches = max(
        1,
        int(
            np.ceil(
                len(source_tensor) / BATCH_SIZE
            )
        ),
    )

    target_batches = max(
        1,
        int(
            np.ceil(
                len(target_tensor) / BATCH_SIZE
            )
        ),
    )

    batches_per_epoch = max(
        source_batches,
        target_batches,
    )

    for epoch in range(EPOCHS):
        model.train()

        total_classification = 0.0
        total_domain = 0.0
        total_loss = 0.0

        progress = (
            float(epoch)
            / max(
                EPOCHS - 1,
                1,
            )
        )

        lambda_value = (
            2.0
            / (
                1.0
                + np.exp(
                    -10.0 * progress
                )
            )
            - 1.0
        )

        for _ in range(batches_per_epoch):
            source_indices = source_rng.choice(
                len(source_tensor),
                size=min(
                    BATCH_SIZE,
                    len(source_tensor),
                ),
                replace=False,
            )

            target_indices = target_rng.choice(
                len(target_tensor),
                size=min(
                    BATCH_SIZE,
                    len(target_tensor),
                ),
                replace=False,
            )

            source_batch = source_tensor[
                source_indices
            ]

            source_y_batch = source_labels[
                source_indices
            ]

            target_batch = target_tensor[
                target_indices
            ]

            domain_source_batch = domain_source[
                source_indices
            ]

            domain_target_batch = domain_target[
                target_indices
            ]

            optimizer.zero_grad()

            source_class_logits, source_domain_logits, _ = model(
                source_batch,
                lambda_value=lambda_value,
            )

            target_class_logits, target_domain_logits, _ = model(
                target_batch,
                lambda_value=lambda_value,
            )

            source_task_loss = classification_loss(
                source_class_logits,
                source_y_batch,
            )

            source_domain_loss = domain_loss(
                source_domain_logits,
                domain_source_batch,
            )

            target_domain_loss = domain_loss(
                target_domain_logits,
                domain_target_batch,
            )

            current_domain_loss = (
                0.5
                * (
                    source_domain_loss
                    + target_domain_loss
                )
            )

            loss = (
                source_task_loss
                + DOMAIN_WEIGHT
                * current_domain_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_classification += float(
                source_task_loss.item()
            )

            total_domain += float(
                current_domain_loss.item()
            )

            total_loss += float(
                loss.item()
            )

        epoch_record = {
            "epoch": epoch + 1,
            "classification_loss": total_classification / batches_per_epoch,
            "domain_loss": total_domain / batches_per_epoch,
            "total_loss": total_loss / batches_per_epoch,
            "lambda": float(lambda_value),
        }

        history.append(
            epoch_record
        )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"EPOCH {epoch + 1:03d} "
                f"CLS={epoch_record['classification_loss']:.6f} "
                f"DOMAIN={epoch_record['domain_loss']:.6f} "
                f"TOTAL={epoch_record['total_loss']:.6f} "
                f"LAMBDA={epoch_record['lambda']:.6f}"
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

    for key, value in source_result["metrics"].items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    print()
    print("EXPERIMENTAL TRAIN")

    for key, value in target_train_result["metrics"].items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    print()
    print("EXPERIMENTAL TEST")

    for key, value in target_test_result["metrics"].items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    print()
    print("EXPERIMENTAL TEST CONFUSION MATRIX")

    print(
        np.asarray(
            target_test_result["confusion_matrix"]
        )
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "domain_weight": DOMAIN_WEIGHT,
        "history": history,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_FILE,
    )

    results = {
        "experiment": "realmat_bag_dann",
        "threshold_ev": THRESHOLD,
        "random_state": RANDOM_STATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "domain_weight": DOMAIN_WEIGHT,
        "feature_count": len(FEATURE_COLUMNS),
        "source_samples": len(source),
        "target_train_samples": len(target_train),
        "target_test_samples": len(target_test),
        "source_train": source_result["metrics"],
        "experimental_train": target_train_result["metrics"],
        "experimental_test": target_test_result["metrics"],
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
    print("DANN COMPLETE")
    print("=" * 80)
    print(f"CHECKPOINT: {CHECKPOINT_FILE}")
    print(f"RESULTS: {RESULT_FILE}")


if __name__ == "__main__":
    main()