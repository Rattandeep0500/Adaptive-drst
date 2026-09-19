import copy
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "results"
    / "water"
    / "biochar_ec"
    / "biochar_ec_unique_conditions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "water"
    / "biochar_ec"
    / "dann_regression"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DOMAIN = "Lake water"
TARGET_DOMAIN = "Secondary effluent"

DOMAIN_COLUMN = "Wastewater type"
TARGET_COLUMN = "Capacity"
FINAL_CONCENTRATION = "Final concentration"

DEVICE = torch.device("cpu")

SPLIT_SEED = 42
MODEL_SEEDS = [11, 23, 37, 51, 71]

BATCH_SIZE = 64
MAX_EPOCHS = 500
PATIENCE = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

CONFIGS = {
    "mlp_source_only": 0.0,
    "dann_lambda_0.1": 0.1,
    "dann_lambda_0.5": 0.5,
    "dann_lambda_1.0": 1.0,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_one_hot():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def canonicalize(series):
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")

        return x.map(
            lambda value: "__NA__"
            if pd.isna(value)
            else format(float(value), ".12g")
        )

    return (
        series.astype("string")
        .fillna("__NA__")
        .str.strip()
        .str.lower()
    )


def make_signature(df, columns):
    canonical = pd.DataFrame(index=df.index)

    for column in columns:
        canonical[column] = canonicalize(
            df[column]
        )

    return pd.util.hash_pandas_object(
        canonical,
        index=False,
    ).astype("uint64")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rank_true = pd.Series(
        y_true
    ).rank(method="average")

    rank_pred = pd.Series(
        y_pred
    ).rank(method="average")

    spearman = rank_true.corr(
        rank_pred,
        method="pearson",
    )

    return {
        "n": int(len(y_true)),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(mse)
        ),
        "median_ae": float(
            median_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ) if len(y_true) > 1 else np.nan,
        "spearman": float(spearman)
        if pd.notna(spearman)
        else np.nan,
        "mean_error": float(
            np.mean(
                y_pred - y_true
            )
        ),
        "p90_ae": float(
            np.quantile(
                np.abs(
                    y_pred - y_true
                ),
                0.90,
            )
        ),
    }


class GradientReversalFunction(
    torch.autograd.Function
):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            -ctx.coefficient * grad_output,
            None,
        )


def gradient_reverse(x, coefficient):
    return GradientReversalFunction.apply(
        x,
        coefficient,
    )


class DANNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def encode(self, x):
        return self.encoder(x)

    def predict_regression(self, x):
        z = self.encode(x)
        return self.regressor(z).squeeze(1)

    def predict_domain(
        self,
        x,
        coefficient,
    ):
        z = self.encode(x)

        z = gradient_reverse(
            z,
            coefficient,
        )

        return self.domain_head(
            z
        ).squeeze(1)


def tensor(x):
    return torch.as_tensor(
        x,
        dtype=torch.float32,
        device=DEVICE,
    )


def make_batches(
    n,
    batch_size,
    generator,
):
    order = torch.randperm(
        n,
        generator=generator,
    )

    return [
        order[start:start + batch_size]
        for start in range(
            0,
            n,
            batch_size,
        )
    ]


def domain_metrics(
    model,
    x_source,
    x_target,
):
    model.eval()

    with torch.no_grad():
        source_z = model.encode(
            tensor(x_source)
        )

        target_z = model.encode(
            tensor(x_target)
        )

        source_logits = model.domain_head(
            source_z
        ).squeeze(1)

        target_logits = model.domain_head(
            target_z
        ).squeeze(1)

        logits = torch.cat(
            [
                source_logits,
                target_logits,
            ]
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    labels = np.concatenate(
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

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    source_accuracy = np.mean(
        predictions[:len(x_source)] == 0
    )

    target_accuracy = np.mean(
        predictions[len(x_source):] == 1
    )

    balanced_accuracy = (
        source_accuracy
        + target_accuracy
    ) / 2.0

    return {
        "domain_auc": float(auc),
        "domain_balanced_accuracy": float(
            balanced_accuracy
        ),
    }


def train_one(
    config_name,
    lambda_max,
    seed,
    x_train,
    y_train,
    x_val,
    y_val,
    x_target,
    y_mean,
    y_std,
):
    set_seed(seed)

    model = DANNRegressor(
        input_dim=x_train.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    task_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    x_train_t = tensor(x_train)
    y_train_t = tensor(y_train)

    x_target_t = tensor(x_target)

    best_state = None
    best_val_mae = float("inf")
    best_epoch = 0

    patience_counter = 0

    source_generator = torch.Generator(
        device="cpu"
    )

    target_generator = torch.Generator(
        device="cpu"
    )

    source_generator.manual_seed(
        seed + 1000
    )

    target_generator.manual_seed(
        seed + 2000
    )

    total_steps_estimate = (
        MAX_EPOCHS
        * max(
            math.ceil(
                len(x_train) / BATCH_SIZE
            ),
            math.ceil(
                len(x_target) / BATCH_SIZE
            ),
        )
    )

    global_step = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        source_batches = make_batches(
            len(x_train),
            BATCH_SIZE,
            source_generator,
        )

        target_batches = make_batches(
            len(x_target),
            BATCH_SIZE,
            target_generator,
        )

        steps = max(
            len(source_batches),
            len(target_batches),
        )

        for step in range(steps):
            source_idx = source_batches[
                step % len(source_batches)
            ]

            target_idx = target_batches[
                step % len(target_batches)
            ]

            xs = x_train_t[source_idx]
            ys = y_train_t[source_idx]

            xt = x_target_t[target_idx]

            progress = min(
                1.0,
                global_step
                / max(
                    total_steps_estimate - 1,
                    1,
                ),
            )

            schedule = (
                2.0
                / (
                    1.0
                    + math.exp(
                        -10.0 * progress
                    )
                )
                - 1.0
            )

            coefficient = (
                lambda_max
                * schedule
            )

            optimizer.zero_grad()

            source_prediction = (
                model.predict_regression(
                    xs
                )
            )

            task_loss = task_loss_fn(
                source_prediction,
                ys,
            )

            combined = torch.cat(
                [xs, xt],
                dim=0,
            )

            domain_labels = torch.cat(
                [
                    torch.zeros(
                        len(xs),
                        device=DEVICE,
                    ),
                    torch.ones(
                        len(xt),
                        device=DEVICE,
                    ),
                ]
            )

            domain_logits = (
                model.predict_domain(
                    combined,
                    coefficient,
                )
            )

            domain_loss = (
                domain_loss_fn(
                    domain_logits,
                    domain_labels,
                )
            )

            loss = (
                task_loss
                + domain_loss
            )

            loss.backward()

            optimizer.step()

            global_step += 1

        model.eval()

        with torch.no_grad():
            val_standardized = (
                model.predict_regression(
                    tensor(x_val)
                )
                .cpu()
                .numpy()
            )

        val_prediction = (
            val_standardized
            * y_std
            + y_mean
        )

        val_mae = mean_absolute_error(
            y_val,
            val_prediction,
        )

        if val_mae < (
            best_val_mae - 1e-7
        ):
            best_val_mae = float(
                val_mae
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:
            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "Training produced no valid checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_val_mae,
        best_epoch,
    )


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST BIOCHAR CPU DANN REGRESSION CONTROL")
    print("=" * 100)

    print(f"Device: {DEVICE}")
    print(
        f"Source: {SOURCE_DOMAIN}"
    )
    print(
        f"Unlabeled adaptation target: "
        f"{TARGET_DOMAIN}"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    source = df[
        df[DOMAIN_COLUMN]
        == SOURCE_DOMAIN
    ].copy()

    target = df[
        df[DOMAIN_COLUMN]
        == TARGET_DOMAIN
    ].copy()

    if len(source) != 322:
        raise RuntimeError(
            f"Expected 322 source conditions, "
            f"got {len(source)}"
        )

    if len(target) != 140:
        raise RuntimeError(
            f"Expected 140 target conditions, "
            f"got {len(target)}"
        )

    forbidden = {
        DOMAIN_COLUMN,
        TARGET_COLUMN,
        FINAL_CONCENTRATION,
        "_condition_signature",
        "n_replicates",
        "capacity_mean",
        "capacity_std",
        "capacity_min",
        "capacity_max",
        "capacity_spread",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in forbidden
        and not column.startswith("_")
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    train_idx, val_idx = (
        train_test_split(
            np.arange(len(source)),
            test_size=0.20,
            random_state=SPLIT_SEED,
        )
    )

    source_train = (
        source.iloc[train_idx]
        .reset_index(drop=True)
    )

    source_val = (
        source.iloc[val_idx]
        .reset_index(drop=True)
    )

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                make_one_hot(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    x_train = preprocessor.fit_transform(
        source_train[
            feature_columns
        ]
    ).astype(np.float32)

    x_val = preprocessor.transform(
        source_val[
            feature_columns
        ]
    ).astype(np.float32)

    x_source_all = preprocessor.transform(
        source[
            feature_columns
        ]
    ).astype(np.float32)

    x_target = preprocessor.transform(
        target[
            feature_columns
        ]
    ).astype(np.float32)

    y_train_original = (
        pd.to_numeric(
            source_train[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .to_numpy(dtype=np.float32)
    )

    y_val = (
        pd.to_numeric(
            source_val[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .to_numpy(dtype=np.float32)
    )

    y_source_all = (
        pd.to_numeric(
            source[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .to_numpy(dtype=np.float32)
    )

    y_target = (
        pd.to_numeric(
            target[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .to_numpy(dtype=np.float32)
    )

    y_mean = float(
        y_train_original.mean()
    )

    y_std = float(
        y_train_original.std()
    )

    if y_std <= 0:
        raise RuntimeError(
            "Source target standard deviation is zero"
        )

    y_train = (
        y_train_original
        - y_mean
    ) / y_std

    source["_support_signature"] = (
        make_signature(
            source,
            feature_columns,
        )
    )

    target["_support_signature"] = (
        make_signature(
            target,
            feature_columns,
        )
    )

    shared_signatures = (
        set(
            source[
                "_support_signature"
            ]
        )
        & set(
            target[
                "_support_signature"
            ]
        )
    )

    supported_mask = (
        target[
            "_support_signature"
        ]
        .isin(
            shared_signatures
        )
        .to_numpy()
    )

    unsupported_mask = (
        ~target[
            "_support_signature"
        ]
        .isin(
            shared_signatures
        )
        .to_numpy()
    )

    print()
    print("DATA")
    print("-" * 100)
    print(
        f"Source train: {len(source_train)}"
    )
    print(
        f"Source validation: {len(source_val)}"
    )
    print(
        f"Target unlabeled adaptation samples: "
        f"{len(target)}"
    )
    print(
        f"Input dimension after preprocessing: "
        f"{x_train.shape[1]}"
    )
    print(
        f"Supported target: "
        f"{supported_mask.sum()}"
    )
    print(
        f"Unsupported target: "
        f"{unsupported_mask.sum()}"
    )

    all_rows = []
    prediction_rows = []

    print()
    print("=" * 100)
    print("TRAINING FIXED CONFIGURATIONS")
    print("TARGET LABELS ARE NOT USED FOR TRAINING OR EARLY STOPPING")
    print("=" * 100)

    for config_name, lambda_max in CONFIGS.items():
        print()
        print(
            f"{config_name} "
            f"(lambda_max={lambda_max})"
        )
        print("-" * 100)

        for seed in MODEL_SEEDS:
            model, val_mae, best_epoch = (
                train_one(
                    config_name=config_name,
                    lambda_max=lambda_max,
                    seed=seed,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_target=x_target,
                    y_mean=y_mean,
                    y_std=y_std,
                )
            )

            model.eval()

            with torch.no_grad():
                source_prediction_std = (
                    model.predict_regression(
                        tensor(
                            x_source_all
                        )
                    )
                    .cpu()
                    .numpy()
                )

                target_prediction_std = (
                    model.predict_regression(
                        tensor(
                            x_target
                        )
                    )
                    .cpu()
                    .numpy()
                )

            source_prediction = (
                source_prediction_std
                * y_std
                + y_mean
            )

            target_prediction = (
                target_prediction_std
                * y_std
                + y_mean
            )

            source_result = (
                regression_metrics(
                    y_source_all,
                    source_prediction,
                )
            )

            target_result = (
                regression_metrics(
                    y_target,
                    target_prediction,
                )
            )

            supported_result = (
                regression_metrics(
                    y_target[
                        supported_mask
                    ],
                    target_prediction[
                        supported_mask
                    ],
                )
            )

            unsupported_result = (
                regression_metrics(
                    y_target[
                        unsupported_mask
                    ],
                    target_prediction[
                        unsupported_mask
                    ],
                )
            )

            domain_result = (
                domain_metrics(
                    model,
                    x_train,
                    x_target,
                )
            )

            row = {
                "config": config_name,
                "lambda_max": lambda_max,
                "seed": seed,
                "best_epoch": best_epoch,
                "source_val_mae": val_mae,
                "source_all_mae": source_result[
                    "mae"
                ],
                "target_mae": target_result[
                    "mae"
                ],
                "target_rmse": target_result[
                    "rmse"
                ],
                "target_r2": target_result[
                    "r2"
                ],
                "target_spearman": target_result[
                    "spearman"
                ],
                "target_mean_error": target_result[
                    "mean_error"
                ],
                "supported_mae": supported_result[
                    "mae"
                ],
                "unsupported_mae": unsupported_result[
                    "mae"
                ],
                **domain_result,
            }

            all_rows.append(row)

            print(
                f"seed={seed:<3} "
                f"epoch={best_epoch:<3} "
                f"src_val_MAE={val_mae:8.4f} "
                f"target_MAE={target_result['mae']:8.4f} "
                f"R2={target_result['r2']:7.4f} "
                f"domain_AUC={domain_result['domain_auc']:7.4f}"
            )

            pred = target[
                [
                    "_condition_signature",
                    "Adsorbent",
                    "Pollutant",
                    TARGET_COLUMN,
                ]
            ].copy()

            pred["config"] = config_name
            pred["seed"] = seed
            pred["prediction"] = (
                target_prediction
            )

            pred["absolute_error"] = np.abs(
                target_prediction
                - y_target
            )

            pred["exact_source_support"] = (
                supported_mask
            )

            prediction_rows.append(
                pred
            )

    results = pd.DataFrame(
        all_rows
    )

    results.to_csv(
        OUTPUT_DIR
        / "dann_all_runs.csv",
        index=False,
    )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "dann_target_predictions.csv",
        index=False,
    )

    aggregate = (
        results.groupby(
            [
                "config",
                "lambda_max",
            ],
            as_index=False,
        )
        .agg(
            target_mae_mean=(
                "target_mae",
                "mean",
            ),
            target_mae_std=(
                "target_mae",
                "std",
            ),
            target_rmse_mean=(
                "target_rmse",
                "mean",
            ),
            target_r2_mean=(
                "target_r2",
                "mean",
            ),
            target_spearman_mean=(
                "target_spearman",
                "mean",
            ),
            supported_mae_mean=(
                "supported_mae",
                "mean",
            ),
            unsupported_mae_mean=(
                "unsupported_mae",
                "mean",
            ),
            domain_auc_mean=(
                "domain_auc",
                "mean",
            ),
            domain_auc_std=(
                "domain_auc",
                "std",
            ),
            source_val_mae_mean=(
                "source_val_mae",
                "mean",
            ),
        )
        .sort_values(
            "lambda_max"
        )
        .reset_index(drop=True)
    )

    aggregate.to_csv(
        OUTPUT_DIR
        / "dann_aggregate.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("MULTI-SEED AGGREGATE RESULTS")
    print("=" * 100)

    print(
        aggregate.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("REFERENCE RANDOM-FOREST SOURCE-ONLY RESULT")
    print("=" * 100)

    print("Target MAE: 10.432588")
    print("Target RMSE: 16.689850")
    print("Target R2: 0.910521")
    print("Supported MAE: 11.482721")
    print("Unsupported MAE: 6.888387")

    print()
    print("=" * 100)
    print("INTERPRETATION RULE")
    print("=" * 100)

    print(
        "Do not choose a DANN lambda using target labels."
    )

    print(
        "All fixed lambda configurations are reported."
    )

    print(
        "The experiment asks whether adversarial alignment "
        "consistently improves transfer relative to the identical "
        "lambda=0 neural architecture."
    )

    print(
        "Random forest remains the stronger non-neural "
        "source-only reference unless DANN clearly surpasses it."
    )


if __name__ == "__main__":
    main()
