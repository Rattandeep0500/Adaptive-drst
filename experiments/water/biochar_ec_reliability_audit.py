import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
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
    / "reliability_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DOMAIN = "Lake water"
TARGET_DOMAIN = "Secondary effluent"

DOMAIN_COLUMN = "Wastewater type"
TARGET_COLUMN = "Capacity"
FINAL_CONCENTRATION = "Final concentration"

SEED = 42
N_FOLDS = 5
K_NEIGHBORS = 10

EXPECTED_RF_TARGET_MAE = 10.432588


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


def make_preprocessor(
    numeric_columns,
    categorical_columns,
):
    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
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
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                make_one_hot(),
            ),
        ]
    )

    return ColumnTransformer(
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


def make_rf_pipeline(
    numeric_columns,
    categorical_columns,
):
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_columns,
                    categorical_columns,
                ),
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


def canonicalize(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        return values.map(
            lambda x: "__NA__"
            if pd.isna(x)
            else format(float(x), ".12g")
        )

    return (
        series.astype("string")
        .fillna("__NA__")
        .str.strip()
        .str.lower()
    )


def make_signature(df, columns):
    canonical = pd.DataFrame(
        index=df.index
    )

    for column in columns:
        canonical[column] = canonicalize(
            df[column]
        )

    return pd.util.hash_pandas_object(
        canonical,
        index=False,
    ).astype("uint64")


def spearman(a, b):
    a = pd.Series(
        np.asarray(a, dtype=float)
    ).rank(method="average")

    b = pd.Series(
        np.asarray(b, dtype=float)
    ).rank(method="average")

    value = a.corr(
        b,
        method="pearson",
    )

    return (
        float(value)
        if pd.notna(value)
        else np.nan
    )


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if len(np.unique(labels)) != 2:
        return np.nan

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


def basic_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
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
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ) if len(y_true) > 1 else np.nan,
        "spearman": spearman(
            y_true,
            y_pred,
        ),
        "mean_error": float(
            np.mean(
                y_pred - y_true
            )
        ),
    }


def tree_prediction_std(
    fitted_pipeline,
    x_df,
):
    preprocessor = (
        fitted_pipeline
        .named_steps["preprocess"]
    )

    forest = (
        fitted_pipeline
        .named_steps["model"]
    )

    transformed = preprocessor.transform(
        x_df
    )

    predictions = np.stack(
        [
            estimator.predict(
                transformed
            )
            for estimator in forest.estimators_
        ],
        axis=1,
    )

    return predictions.std(
        axis=1,
        ddof=1,
    )


def source_oof_predictions(
    x_source,
    y_source,
    numeric_columns,
    categorical_columns,
):
    splitter = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    predictions = np.full(
        len(x_source),
        np.nan,
        dtype=float,
    )

    tree_std = np.full(
        len(x_source),
        np.nan,
        dtype=float,
    )

    fold_ids = np.full(
        len(x_source),
        -1,
        dtype=int,
    )

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        splitter.split(x_source),
        start=1,
    ):
        model = make_rf_pipeline(
            numeric_columns,
            categorical_columns,
        )

        model.fit(
            x_source.iloc[train_idx],
            y_source[train_idx],
        )

        predictions[val_idx] = (
            model.predict(
                x_source.iloc[val_idx]
            )
        )

        tree_std[val_idx] = (
            tree_prediction_std(
                model,
                x_source.iloc[val_idx],
            )
        )

        fold_ids[val_idx] = fold

    if (
        np.isnan(predictions).any()
        or
        np.isnan(tree_std).any()
    ):
        raise RuntimeError(
            "OOF prediction generation failed."
        )

    return (
        predictions,
        tree_std,
        fold_ids,
    )


def domain_oof_probabilities(
    x_source,
    x_target,
):
    x_domain = np.vstack(
        [
            x_source,
            x_target,
        ]
    )

    labels = np.concatenate(
        [
            np.zeros(
                len(x_source),
                dtype=int,
            ),
            np.ones(
                len(x_target),
                dtype=int,
            ),
        ]
    )

    probabilities = np.full(
        len(labels),
        np.nan,
        dtype=float,
    )

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    for train_idx, val_idx in splitter.split(
        x_domain,
        labels,
    ):
        classifier = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        )

        classifier.fit(
            x_domain[train_idx],
            labels[train_idx],
        )

        probabilities[val_idx] = (
            classifier.predict_proba(
                x_domain[val_idx]
            )[:, 1]
        )

    if np.isnan(probabilities).any():
        raise RuntimeError(
            "Domain OOF probability generation failed."
        )

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    return (
        probabilities[:len(x_source)],
        probabilities[len(x_source):],
        float(auc),
    )


def source_neighbor_features(
    x_source,
    y_source,
    k,
):
    n_neighbors = min(
        k + 1,
        len(x_source),
    )

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    nn.fit(
        x_source
    )

    distances, indices = nn.kneighbors(
        x_source
    )

    clean_distances = []
    clean_indices = []

    for row in range(len(x_source)):
        mask = (
            indices[row] != row
        )

        idx = indices[row][mask][:k]
        dst = distances[row][mask][:k]

        if len(idx) == 0:
            raise RuntimeError(
                "Could not find non-self source neighbor."
            )

        clean_indices.append(idx)
        clean_distances.append(dst)

    mean_distance = np.array(
        [
            np.mean(x)
            for x in clean_distances
        ],
        dtype=float,
    )

    local_std = np.array(
        [
            np.std(
                y_source[idx],
                ddof=1,
            )
            if len(idx) > 1
            else 0.0
            for idx in clean_indices
        ],
        dtype=float,
    )

    nearest_label = np.array(
        [
            y_source[idx[0]]
            for idx in clean_indices
        ],
        dtype=float,
    )

    return (
        mean_distance,
        local_std,
        nearest_label,
    )


def target_neighbor_features(
    x_source,
    x_target,
    y_source,
    k,
):
    n_neighbors = min(
        k,
        len(x_source),
    )

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    nn.fit(
        x_source
    )

    distances, indices = nn.kneighbors(
        x_target
    )

    mean_distance = distances.mean(
        axis=1
    )

    local_std = np.array(
        [
            np.std(
                y_source[idx],
                ddof=1,
            )
            if len(idx) > 1
            else 0.0
            for idx in indices
        ],
        dtype=float,
    )

    nearest_label = np.array(
        [
            y_source[idx[0]]
            for idx in indices
        ],
        dtype=float,
    )

    return (
        mean_distance,
        local_std,
        nearest_label,
    )


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST WATER RELIABILITY / RISK MECHANISM AUDIT")
    print("=" * 100)

    df = pd.read_csv(
        DATA_PATH
    )

    source = (
        df[
            df[DOMAIN_COLUMN]
            == SOURCE_DOMAIN
        ]
        .copy()
        .reset_index(drop=True)
    )

    target = (
        df[
            df[DOMAIN_COLUMN]
            == TARGET_DOMAIN
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(source) != 322:
        raise RuntimeError(
            f"Expected 322 source conditions, got {len(source)}"
        )

    if len(target) != 140:
        raise RuntimeError(
            f"Expected 140 target conditions, got {len(target)}"
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

    x_source_df = source[
        feature_columns
    ].copy()

    x_target_df = target[
        feature_columns
    ].copy()

    y_source = pd.to_numeric(
        source[TARGET_COLUMN],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    y_target = pd.to_numeric(
        target[TARGET_COLUMN],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    print()
    print("DATA")
    print("-" * 100)
    print(f"Source conditions: {len(source)}")
    print(f"Target conditions: {len(target)}")
    print(f"Features: {len(feature_columns)}")

    print()
    print("=" * 100)
    print("SOURCE OUT-OF-FOLD ERROR CONSTRUCTION")
    print("=" * 100)

    (
        source_oof_prediction,
        source_oof_tree_std,
        source_fold,
    ) = source_oof_predictions(
        x_source_df,
        y_source,
        numeric_columns,
        categorical_columns,
    )

    source_oof_error = np.abs(
        source_oof_prediction
        - y_source
    )

    source_oof_mae = float(
        source_oof_error.mean()
    )

    print(
        f"Source OOF RF MAE: "
        f"{source_oof_mae:.6f}"
    )

    print(
        f"Source OOF error median: "
        f"{np.median(source_oof_error):.6f}"
    )

    print(
        f"Source OOF error 75th percentile: "
        f"{np.quantile(source_oof_error, 0.75):.6f}"
    )

    final_model = make_rf_pipeline(
        numeric_columns,
        categorical_columns,
    )

    final_model.fit(
        x_source_df,
        y_source,
    )

    target_prediction = final_model.predict(
        x_target_df
    )

    target_tree_std = tree_prediction_std(
        final_model,
        x_target_df,
    )

    target_error = np.abs(
        target_prediction
        - y_target
    )

    target_metrics = basic_metrics(
        y_target,
        target_prediction,
    )

    print()
    print("=" * 100)
    print("FROZEN RANDOM-FOREST TARGET BASELINE")
    print("=" * 100)

    for key, value in target_metrics.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    difference = (
        target_metrics["mae"]
        - EXPECTED_RF_TARGET_MAE
    )

    print(
        f"Difference from previous RF MAE: "
        f"{difference:.8f}"
    )

    global_preprocessor = (
        make_preprocessor(
            numeric_columns,
            categorical_columns,
        )
    )

    x_source = (
        global_preprocessor
        .fit_transform(
            x_source_df
        )
    )

    x_target = (
        global_preprocessor
        .transform(
            x_target_df
        )
    )

    x_source = np.asarray(
        x_source,
        dtype=np.float64,
    )

    x_target = np.asarray(
        x_target,
        dtype=np.float64,
    )

    (
        source_nn_distance,
        source_local_label_std,
        source_nearest_label,
    ) = source_neighbor_features(
        x_source,
        y_source,
        K_NEIGHBORS,
    )

    (
        target_nn_distance,
        target_local_label_std,
        target_nearest_label,
    ) = target_neighbor_features(
        x_source,
        x_target,
        y_source,
        K_NEIGHBORS,
    )

    (
        source_domain_probability,
        target_domain_probability,
        domain_auc,
    ) = domain_oof_probabilities(
        x_source,
        x_target,
    )

    print()
    print("=" * 100)
    print("DOMAIN-SEPARABILITY AUDIT")
    print("=" * 100)

    print(
        f"Cross-validated source-vs-target domain AUC: "
        f"{domain_auc:.6f}"
    )

    source_signature = make_signature(
        source,
        feature_columns,
    )

    target_signature = make_signature(
        target,
        feature_columns,
    )

    shared_signatures = (
        set(source_signature)
        & set(target_signature)
    )

    source_supported = (
        source_signature
        .isin(shared_signatures)
        .to_numpy()
    )

    target_supported = (
        target_signature
        .isin(shared_signatures)
        .to_numpy()
    )

    print(
        f"Source conditions with exact target counterpart: "
        f"{source_supported.sum()}"
    )

    print(
        f"Target conditions with exact source counterpart: "
        f"{target_supported.sum()}"
    )

    source_nearest_gap = np.abs(
        source_oof_prediction
        - source_nearest_label
    )

    target_nearest_gap = np.abs(
        target_prediction
        - target_nearest_label
    )

    eps = 1e-6

    source_log_domain_odds = np.log(
        (
            np.clip(
                source_domain_probability,
                eps,
                1.0 - eps,
            )
        )
        /
        (
            1.0
            - np.clip(
                source_domain_probability,
                eps,
                1.0 - eps,
            )
        )
    )

    target_log_domain_odds = np.log(
        (
            np.clip(
                target_domain_probability,
                eps,
                1.0 - eps,
            )
        )
        /
        (
            1.0
            - np.clip(
                target_domain_probability,
                eps,
                1.0 - eps,
            )
        )
    )

    risk_features = [
        "rf_tree_std",
        "nn_distance_mean",
        "local_label_std",
        "nearest_source_label_gap",
        "domain_target_probability",
        "log_target_source_odds",
        "unsupported_flag",
        "absolute_prediction",
    ]

    source_risk_features = pd.DataFrame(
        {
            "rf_tree_std":
                source_oof_tree_std,

            "nn_distance_mean":
                source_nn_distance,

            "local_label_std":
                source_local_label_std,

            "nearest_source_label_gap":
                source_nearest_gap,

            "domain_target_probability":
                source_domain_probability,

            "log_target_source_odds":
                source_log_domain_odds,

            "unsupported_flag":
                (~source_supported).astype(float),

            "absolute_prediction":
                np.abs(source_oof_prediction),
        }
    )

    target_risk_features = pd.DataFrame(
        {
            "rf_tree_std":
                target_tree_std,

            "nn_distance_mean":
                target_nn_distance,

            "local_label_std":
                target_local_label_std,

            "nearest_source_label_gap":
                target_nearest_gap,

            "domain_target_probability":
                target_domain_probability,

            "log_target_source_odds":
                target_log_domain_odds,

            "unsupported_flag":
                (~target_supported).astype(float),

            "absolute_prediction":
                np.abs(target_prediction),
        }
    )

    if (
        source_risk_features.isna().any().any()
        or
        target_risk_features.isna().any().any()
    ):
        raise RuntimeError(
            "Risk feature matrix contains NaN values."
        )

    print()
    print("=" * 100)
    print("SOURCE-DERIVED RISK MODEL")
    print("TARGET CAPACITY LABELS ARE NOT USED")
    print("=" * 100)

    risk_model = RandomForestRegressor(
        n_estimators=800,
        min_samples_leaf=8,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
    )

    risk_model.fit(
        source_risk_features[
            risk_features
        ],
        source_oof_error,
    )

    source_oob_risk = np.asarray(
        risk_model.oob_prediction_,
        dtype=float,
    )

    target_predicted_risk = (
        risk_model.predict(
            target_risk_features[
                risk_features
            ]
        )
    )

    source_oob_risk_spearman = spearman(
        source_oof_error,
        source_oob_risk,
    )

    target_risk_spearman = spearman(
        target_error,
        target_predicted_risk,
    )

    print(
        f"Source OOB predicted-risk vs OOF-error Spearman: "
        f"{source_oob_risk_spearman:.6f}"
    )

    print(
        f"Target predicted-risk vs actual-error Spearman: "
        f"{target_risk_spearman:.6f}"
    )

    source_error_median = float(
        np.quantile(
            source_oof_error,
            0.50,
        )
    )

    source_error_q75 = float(
        np.quantile(
            source_oof_error,
            0.75,
        )
    )

    target_high_error_median = (
        target_error
        >= source_error_median
    ).astype(int)

    target_high_error_q75 = (
        target_error
        >= source_error_q75
    ).astype(int)

    risk_auc_median = safe_auc(
        target_high_error_median,
        target_predicted_risk,
    )

    risk_auc_q75 = safe_auc(
        target_high_error_q75,
        target_predicted_risk,
    )

    print(
        f"Target high-error AUROC "
        f"(source median threshold={source_error_median:.6f}): "
        f"{risk_auc_median:.6f}"
    )

    print(
        f"Target high-error AUROC "
        f"(source Q75 threshold={source_error_q75:.6f}): "
        f"{risk_auc_q75:.6f}"
    )

    print()
    print("=" * 100)
    print("INDIVIDUAL RELIABILITY SIGNAL AUDIT")
    print("=" * 100)

    signal_rows = []

    for signal in risk_features:
        values = (
            target_risk_features[
                signal
            ].to_numpy(
                dtype=float
            )
        )

        row = {
            "signal": signal,
            "target_error_spearman":
                spearman(
                    target_error,
                    values,
                ),
            "high_error_q75_auc":
                safe_auc(
                    target_high_error_q75,
                    values,
                ),
        }

        signal_rows.append(row)

    signal_rows.append(
        {
            "signal":
                "source_derived_risk_model",

            "target_error_spearman":
                target_risk_spearman,

            "high_error_q75_auc":
                risk_auc_q75,
        }
    )

    signal_table = (
        pd.DataFrame(
            signal_rows
        )
        .sort_values(
            "target_error_spearman",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        signal_table.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("SELECTIVE RISK / COVERAGE CURVE")
    print("=" * 100)

    order = np.argsort(
        target_predicted_risk
    )

    coverage_rows = []

    for coverage in [
        0.25,
        0.50,
        0.75,
        1.00,
    ]:
        n_selected = max(
            1,
            int(
                np.floor(
                    coverage
                    * len(target)
                )
            ),
        )

        idx = order[
            :n_selected
        ]

        row = {
            "coverage":
                coverage,

            "n":
                n_selected,

            "mae":
                float(
                    target_error[
                        idx
                    ].mean()
                ),

            "median_absolute_error":
                float(
                    np.median(
                        target_error[
                            idx
                        ]
                    )
                ),

            "p90_absolute_error":
                float(
                    np.quantile(
                        target_error[
                            idx
                        ],
                        0.90,
                    )
                ),

            "mean_predicted_risk":
                float(
                    target_predicted_risk[
                        idx
                    ].mean()
                ),
        }

        coverage_rows.append(
            row
        )

    coverage_table = pd.DataFrame(
        coverage_rows
    )

    print(
        coverage_table.to_string(
            index=False
        )
    )

    quartile_n = max(
        1,
        len(target) // 4,
    )

    low_risk_idx = order[
        :quartile_n
    ]

    high_risk_idx = order[
        -quartile_n:
    ]

    low_risk_mae = float(
        target_error[
            low_risk_idx
        ].mean()
    )

    high_risk_mae = float(
        target_error[
            high_risk_idx
        ].mean()
    )

    print()
    print(
        f"Lowest-risk quartile MAE: "
        f"{low_risk_mae:.6f}"
    )

    print(
        f"Highest-risk quartile MAE: "
        f"{high_risk_mae:.6f}"
    )

    print(
        f"High/low risk MAE ratio: "
        f"{high_risk_mae / low_risk_mae:.6f}"
    )

    supported_mae = float(
        target_error[
            target_supported
        ].mean()
    )

    unsupported_mae = float(
        target_error[
            ~target_supported
        ].mean()
    )

    print()
    print("=" * 100)
    print("EXACT SUPPORT CHECK")
    print("=" * 100)

    print(
        f"Supported target MAE: "
        f"{supported_mae:.6f}"
    )

    print(
        f"Unsupported target MAE: "
        f"{unsupported_mae:.6f}"
    )

    print(
        "Exact support is evaluated as a candidate signal, "
        "not assumed to imply reliability."
    )

    target_output = target[
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

    target_output[
        "prediction"
    ] = target_prediction

    target_output[
        "absolute_error"
    ] = target_error

    target_output[
        "predicted_risk"
    ] = target_predicted_risk

    target_output[
        "exact_source_support"
    ] = target_supported

    for column in risk_features:
        target_output[
            column
        ] = target_risk_features[
            column
        ].to_numpy()

    target_output.to_csv(
        OUTPUT_DIR
        / "target_reliability_predictions.csv",
        index=False,
    )

    source_output = source[
        [
            "_condition_signature",
            "Adsorbent",
            "Pollutant",
            TARGET_COLUMN,
        ]
    ].copy()

    source_output[
        "oof_prediction"
    ] = source_oof_prediction

    source_output[
        "oof_absolute_error"
    ] = source_oof_error

    source_output[
        "oob_predicted_risk"
    ] = source_oob_risk

    source_output[
        "fold"
    ] = source_fold

    for column in risk_features:
        source_output[
            column
        ] = source_risk_features[
            column
        ].to_numpy()

    source_output.to_csv(
        OUTPUT_DIR
        / "source_oof_reliability.csv",
        index=False,
    )

    signal_table.to_csv(
        OUTPUT_DIR
        / "reliability_signal_audit.csv",
        index=False,
    )

    coverage_table.to_csv(
        OUTPUT_DIR
        / "selective_risk_coverage.csv",
        index=False,
    )

    feature_importance = pd.DataFrame(
        {
            "feature":
                risk_features,

            "importance":
                risk_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        OUTPUT_DIR
        / "risk_model_feature_importance.csv",
        index=False,
    )

    summary = {
        "source_domain":
            SOURCE_DOMAIN,

        "target_domain":
            TARGET_DOMAIN,

        "source_conditions":
            len(source),

        "target_conditions":
            len(target),

        "source_oof_mae":
            source_oof_mae,

        "target_baseline":
            target_metrics,

        "domain_auc":
            domain_auc,

        "source_oob_risk_spearman":
            source_oob_risk_spearman,

        "target_risk_spearman":
            target_risk_spearman,

        "source_error_median":
            source_error_median,

        "source_error_q75":
            source_error_q75,

        "target_risk_auc_median":
            risk_auc_median,

        "target_risk_auc_q75":
            risk_auc_q75,

        "low_risk_quartile_mae":
            low_risk_mae,

        "high_risk_quartile_mae":
            high_risk_mae,

        "high_low_risk_mae_ratio":
            high_risk_mae
            / low_risk_mae,

        "supported_target_mae":
            supported_mae,

        "unsupported_target_mae":
            unsupported_mae,
    }

    with open(
        OUTPUT_DIR
        / "reliability_summary.json",
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
        / "target_reliability_predictions.csv"
    )

    print(
        OUTPUT_DIR
        / "source_oof_reliability.csv"
    )

    print(
        OUTPUT_DIR
        / "reliability_signal_audit.csv"
    )

    print(
        OUTPUT_DIR
        / "selective_risk_coverage.csv"
    )

    print(
        OUTPUT_DIR
        / "risk_model_feature_importance.csv"
    )

    print(
        OUTPUT_DIR
        / "reliability_summary.json"
    )

    print()
    print("=" * 100)
    print("DECISION RULE")
    print("=" * 100)

    print(
        "Target labels were used only after all prediction and "
        "risk scores were frozen."
    )

    print(
        "If predicted risk ranks target error and low-risk coverage "
        "has materially lower MAE, proceed to the final "
        "Adaptive-DRST reliability experiment."
    )

    print(
        "If it fails, do not manufacture a reliability result; "
        "the failure becomes another documented mechanism result."
    )


if __name__ == "__main__":
    main()
