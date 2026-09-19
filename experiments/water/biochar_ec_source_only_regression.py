from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    / "source_only"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DOMAIN = "Lake water"
TARGET_DOMAIN = "Secondary effluent"

DOMAIN_COLUMN = "Wastewater type"
TARGET_COLUMN = "Capacity"
FINAL_CONCENTRATION = "Final concentration"

SEED = 42


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
        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        return values.map(
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


def spearman_correlation(y_true, y_pred):
    a = pd.Series(
        np.asarray(y_true, dtype=float)
    ).rank(method="average")

    b = pd.Series(
        np.asarray(y_pred, dtype=float)
    ).rank(method="average")

    value = a.corr(
        b,
        method="pearson",
    )

    return float(value)


def metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
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
        "median_absolute_error": float(
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
        "spearman": (
            spearman_correlation(
                y_true,
                y_pred,
            )
            if len(y_true) > 1
            else np.nan
        ),
        "mean_actual": float(
            np.mean(y_true)
        ),
        "mean_prediction": float(
            np.mean(y_pred)
        ),
        "mean_error": float(
            np.mean(
                y_pred - y_true
            )
        ),
        "p90_absolute_error": float(
            np.quantile(
                np.abs(
                    y_pred - y_true
                ),
                0.90,
            )
        ),
    }


def print_metrics(name, result):
    print()
    print(name)
    print("-" * 100)

    for key, value in result.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )


def source_cv_mae(model, x, y):
    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    fold_mae = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(x),
        start=1,
    ):
        candidate = clone(model)

        candidate.fit(
            x.iloc[train_idx],
            y.iloc[train_idx],
        )

        prediction = candidate.predict(
            x.iloc[val_idx]
        )

        fold_mae.append(
            mean_absolute_error(
                y.iloc[val_idx],
                prediction,
            )
        )

    return {
        "mean_mae": float(
            np.mean(fold_mae)
        ),
        "std_mae": float(
            np.std(
                fold_mae,
                ddof=1,
            )
        ),
        "fold_mae": [
            float(x)
            for x in fold_mae
        ],
    }


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST BIOCHAR SOURCE-ONLY REGRESSION BASELINE")
    print("=" * 100)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing condition-level dataset: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    source = df[
        df[DOMAIN_COLUMN] == SOURCE_DOMAIN
    ].copy()

    target = df[
        df[DOMAIN_COLUMN] == TARGET_DOMAIN
    ].copy()

    if len(source) != 322:
        raise RuntimeError(
            f"Expected 322 Lake-water conditions, got {len(source)}"
        )

    if len(target) != 140:
        raise RuntimeError(
            f"Expected 140 secondary-effluent conditions, got {len(target)}"
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

    categorical_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    print()
    print(f"SOURCE DOMAIN: {SOURCE_DOMAIN}")
    print(f"TARGET DOMAIN: {TARGET_DOMAIN}")
    print(f"Source unique conditions: {len(source)}")
    print(f"Target unique conditions: {len(target)}")
    print(f"Total model features: {len(feature_columns)}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    print()
    print("CATEGORICAL FEATURES")
    print("-" * 100)

    for column in categorical_columns:
        print(column)

    print()
    print("NUMERIC FEATURES")
    print("-" * 100)

    for column in numeric_columns:
        print(column)

    source["_transfer_signature"] = make_signature(
        source,
        feature_columns,
    )

    target["_transfer_signature"] = make_signature(
        target,
        feature_columns,
    )

    shared_signatures = (
        set(source["_transfer_signature"])
        & set(target["_transfer_signature"])
    )

    target_supported = target[
        target["_transfer_signature"].isin(
            shared_signatures
        )
    ].copy()

    target_unsupported = target[
        ~target["_transfer_signature"].isin(
            shared_signatures
        )
    ].copy()

    print()
    print("=" * 100)
    print("TARGET SUPPORT PARTITION")
    print("=" * 100)

    print(
        f"Shared exact signatures: "
        f"{len(shared_signatures)}"
    )

    print(
        f"Supported target conditions: "
        f"{len(target_supported)}"
    )

    print(
        f"Unsupported target conditions: "
        f"{len(target_unsupported)}"
    )

    print(
        f"Supported target fraction: "
        f"{100.0 * len(target_supported) / len(target):.2f}%"
    )

    x_source = source[
        feature_columns
    ].copy()

    y_source = pd.to_numeric(
        source[TARGET_COLUMN],
        errors="raise",
    )

    x_target = target[
        feature_columns
    ].copy()

    y_target = pd.to_numeric(
        target[TARGET_COLUMN],
        errors="raise",
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

    preprocess = ColumnTransformer(
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

    models = {}

    models["dummy_median"] = Pipeline(
        [
            (
                "preprocess",
                clone(preprocess),
            ),
            (
                "model",
                DummyRegressor(
                    strategy="median"
                ),
            ),
        ]
    )

    ridge_alphas = [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]

    ridge_cv = []

    print()
    print("=" * 100)
    print("RIDGE SOURCE-ONLY HYPERPARAMETER SELECTION")
    print("TARGET LABELS ARE NOT USED")
    print("=" * 100)

    for alpha in ridge_alphas:
        model = Pipeline(
            [
                (
                    "preprocess",
                    clone(preprocess),
                ),
                (
                    "model",
                    Ridge(
                        alpha=alpha
                    ),
                ),
            ]
        )

        result = source_cv_mae(
            model,
            x_source,
            y_source,
        )

        ridge_cv.append(
            {
                "alpha": alpha,
                **result,
            }
        )

        print(
            f"alpha={alpha:<8} "
            f"source CV MAE="
            f"{result['mean_mae']:.6f} "
            f"+/- {result['std_mae']:.6f}"
        )

    ridge_cv_df = pd.DataFrame(
        ridge_cv
    )

    best_ridge_row = (
        ridge_cv_df
        .sort_values(
            "mean_mae"
        )
        .iloc[0]
    )

    best_alpha = float(
        best_ridge_row["alpha"]
    )

    print()
    print(
        f"Selected Ridge alpha from source CV only: "
        f"{best_alpha}"
    )

    models["ridge"] = Pipeline(
        [
            (
                "preprocess",
                clone(preprocess),
            ),
            (
                "model",
                Ridge(
                    alpha=best_alpha
                ),
            ),
        ]
    )

    models["random_forest"] = Pipeline(
        [
            (
                "preprocess",
                clone(preprocess),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features=1.0,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["extra_trees"] = Pipeline(
        [
            (
                "preprocess",
                clone(preprocess),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features=1.0,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print()
    print("=" * 100)
    print("SOURCE-DOMAIN CROSS-VALIDATION")
    print("MODEL SELECTION USES SOURCE LABELS ONLY")
    print("=" * 100)

    cv_rows = []

    for name, model in models.items():
        result = source_cv_mae(
            model,
            x_source,
            y_source,
        )

        cv_rows.append(
            {
                "model": name,
                "source_cv_mae": result[
                    "mean_mae"
                ],
                "source_cv_mae_std": result[
                    "std_mae"
                ],
            }
        )

        print(
            f"{name:<20} "
            f"MAE={result['mean_mae']:.6f} "
            f"+/- {result['std_mae']:.6f}"
        )

    cv_table = (
        pd.DataFrame(cv_rows)
        .sort_values(
            "source_cv_mae"
        )
        .reset_index(drop=True)
    )

    best_name = str(
        cv_table.iloc[0]["model"]
    )

    print()
    print(
        f"SELECTED SOURCE-ONLY MODEL: "
        f"{best_name}"
    )

    print()
    print(
        "No secondary-effluent labels were used "
        "for model or hyperparameter selection."
    )

    selected_model = clone(
        models[best_name]
    )

    selected_model.fit(
        x_source,
        y_source,
    )

    source_prediction = (
        selected_model.predict(
            x_source
        )
    )

    target_prediction = (
        selected_model.predict(
            x_target
        )
    )

    source_result = metrics(
        y_source,
        source_prediction,
    )

    target_result = metrics(
        y_target,
        target_prediction,
    )

    supported_mask = (
        target["_transfer_signature"]
        .isin(
            shared_signatures
        )
        .to_numpy()
    )

    unsupported_mask = (
        ~target["_transfer_signature"]
        .isin(
            shared_signatures
        )
        .to_numpy()
    )

    supported_result = metrics(
        y_target.to_numpy()[
            supported_mask
        ],
        target_prediction[
            supported_mask
        ],
    )

    unsupported_result = metrics(
        y_target.to_numpy()[
            unsupported_mask
        ],
        target_prediction[
            unsupported_mask
        ],
    )

    print()
    print("=" * 100)
    print("FINAL SOURCE-ONLY RESULTS")
    print("=" * 100)

    print_metrics(
        "SOURCE TRAIN DOMAIN",
        source_result,
    )

    print_metrics(
        "FULL SECONDARY-EFFLUENT TARGET",
        target_result,
    )

    print_metrics(
        "SUPPORTED TARGET CONDITIONS",
        supported_result,
    )

    print_metrics(
        "UNSUPPORTED TARGET CONDITIONS",
        unsupported_result,
    )

    predictions = target[
        [
            "_condition_signature",
            "Adsorbent",
            "Pollutant",
            "Adsorption type",
            TARGET_COLUMN,
            "n_replicates",
            "capacity_std",
            "capacity_spread",
        ]
    ].copy()

    predictions["prediction"] = (
        target_prediction
    )

    predictions["error"] = (
        predictions["prediction"]
        - predictions[TARGET_COLUMN]
    )

    predictions["absolute_error"] = (
        predictions["error"].abs()
    )

    predictions["exact_source_support"] = (
        supported_mask
    )

    predictions[
        "transfer_signature"
    ] = target[
        "_transfer_signature"
    ].to_numpy()

    predictions.to_csv(
        OUTPUT_DIR
        / "secondary_effluent_predictions.csv",
        index=False,
    )

    cv_table.to_csv(
        OUTPUT_DIR
        / "source_cv_model_selection.csv",
        index=False,
    )

    ridge_cv_df.to_csv(
        OUTPUT_DIR
        / "ridge_source_cv.csv",
        index=False,
    )

    summary = {
        "source_domain": SOURCE_DOMAIN,
        "target_domain": TARGET_DOMAIN,
        "source_conditions": len(source),
        "target_conditions": len(target),
        "supported_target_conditions": int(
            supported_mask.sum()
        ),
        "unsupported_target_conditions": int(
            unsupported_mask.sum()
        ),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "selected_model": best_name,
        "selected_ridge_alpha": best_alpha,
        "source_train": source_result,
        "target_full": target_result,
        "target_supported": supported_result,
        "target_unsupported": unsupported_result,
    }

    with open(
        OUTPUT_DIR
        / "source_only_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("=" * 100)
    print("FILES WRITTEN")
    print("=" * 100)

    print(
        OUTPUT_DIR
        / "source_cv_model_selection.csv"
    )

    print(
        OUTPUT_DIR
        / "ridge_source_cv.csv"
    )

    print(
        OUTPUT_DIR
        / "secondary_effluent_predictions.csv"
    )

    print(
        OUTPUT_DIR
        / "source_only_summary.json"
    )

    print()
    print(
        "Next experiment: CPU-only DANN regression "
        "using Lake-water labels and unlabeled "
        "Secondary-effluent features."
    )


if __name__ == "__main__":
    main()
