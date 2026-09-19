import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
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
    / "adaptive_drst_final"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DOMAIN = "Lake water"

TARGET_DOMAINS = [
    "Secondary effluent",
    "Ground water",
]

DOMAIN_COLUMN = "Wastewater type"
TARGET_COLUMN = "Capacity"
FINAL_CONCENTRATION = "Final concentration"

SEED = 42
N_FOLDS = 5
K_NEIGHBORS = 10


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


def make_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
                SimpleImputer(strategy="most_frequent"),
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


def make_rf(n_estimators=500, min_samples_leaf=2):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=1.0,
        random_state=SEED,
        n_jobs=-1,
    )


def make_prediction_pipeline(numeric_columns, categorical_columns):
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
                make_rf(),
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
    canonical = pd.DataFrame(index=df.index)

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

    if pd.isna(value):
        return np.nan

    return float(value)


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


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    absolute_error = np.abs(
        y_pred - y_true
    )

    return {
        "n": int(len(y_true)),
        "mae": float(
            absolute_error.mean()
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
        ),
        "spearman_prediction": spearman(
            y_true,
            y_pred,
        ),
        "median_absolute_error": float(
            np.median(
                absolute_error
            )
        ),
        "p90_absolute_error": float(
            np.quantile(
                absolute_error,
                0.90,
            )
        ),
        "mean_error": float(
            np.mean(
                y_pred - y_true
            )
        ),
    }


def tree_std_from_pipeline(model, x):
    preprocessor = (
        model.named_steps[
            "preprocess"
        ]
    )

    forest = (
        model.named_steps[
            "model"
        ]
    )

    transformed = (
        preprocessor.transform(x)
    )

    predictions = np.stack(
        [
            tree.predict(
                transformed
            )
            for tree in forest.estimators_
        ],
        axis=1,
    )

    return predictions.std(
        axis=1,
        ddof=1,
    )


def build_source_oof(
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

    oof_prediction = np.full(
        len(x_source),
        np.nan,
        dtype=float,
    )

    oof_tree_std = np.full(
        len(x_source),
        np.nan,
        dtype=float,
    )

    fold_id = np.full(
        len(x_source),
        -1,
        dtype=int,
    )

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        splitter.split(x_source),
        start=1,
    ):
        model = make_prediction_pipeline(
            numeric_columns,
            categorical_columns,
        )

        model.fit(
            x_source.iloc[
                train_index
            ],
            y_source[
                train_index
            ],
        )

        oof_prediction[
            validation_index
        ] = model.predict(
            x_source.iloc[
                validation_index
            ]
        )

        oof_tree_std[
            validation_index
        ] = tree_std_from_pipeline(
            model,
            x_source.iloc[
                validation_index
            ],
        )

        fold_id[
            validation_index
        ] = fold

    if np.isnan(
        oof_prediction
    ).any():
        raise RuntimeError(
            "OOF predictions contain NaN."
        )

    if np.isnan(
        oof_tree_std
    ).any():
        raise RuntimeError(
            "OOF tree uncertainty contains NaN."
        )

    return (
        oof_prediction,
        oof_tree_std,
        fold_id,
    )


def neighbor_source_features(
    source_features,
    source_targets,
):
    neighbors = NearestNeighbors(
        n_neighbors=min(
            K_NEIGHBORS + 1,
            len(source_features),
        ),
        metric="euclidean",
    )

    neighbors.fit(
        source_features
    )

    distances, indices = (
        neighbors.kneighbors(
            source_features
        )
    )

    mean_distance = []
    local_std = []
    nearest_label = []

    for row in range(
        len(source_features)
    ):
        valid = (
            indices[row] != row
        )

        row_indices = (
            indices[row][valid]
            [:K_NEIGHBORS]
        )

        row_distances = (
            distances[row][valid]
            [:K_NEIGHBORS]
        )

        if len(row_indices) == 0:
            raise RuntimeError(
                "Source nearest-neighbor failure."
            )

        mean_distance.append(
            float(
                np.mean(
                    row_distances
                )
            )
        )

        if len(row_indices) > 1:
            local_std.append(
                float(
                    np.std(
                        source_targets[
                            row_indices
                        ],
                        ddof=1,
                    )
                )
            )
        else:
            local_std.append(0.0)

        nearest_label.append(
            float(
                source_targets[
                    row_indices[0]
                ]
            )
        )

    return (
        np.asarray(
            mean_distance
        ),
        np.asarray(
            local_std
        ),
        np.asarray(
            nearest_label
        ),
    )


def neighbor_target_features(
    source_features,
    target_features,
    source_targets,
):
    neighbors = NearestNeighbors(
        n_neighbors=min(
            K_NEIGHBORS,
            len(source_features),
        ),
        metric="euclidean",
    )

    neighbors.fit(
        source_features
    )

    distances, indices = (
        neighbors.kneighbors(
            target_features
        )
    )

    mean_distance = (
        distances.mean(
            axis=1
        )
    )

    local_std = np.array(
        [
            np.std(
                source_targets[
                    row
                ],
                ddof=1,
            )
            if len(row) > 1
            else 0.0
            for row in indices
        ],
        dtype=float,
    )

    nearest_label = np.array(
        [
            source_targets[
                row[0]
            ]
            for row in indices
        ],
        dtype=float,
    )

    return (
        mean_distance,
        local_std,
        nearest_label,
    )


def domain_probabilities(
    source_features,
    target_features,
):
    features = np.vstack(
        [
            source_features,
            target_features,
        ]
    )

    labels = np.concatenate(
        [
            np.zeros(
                len(source_features),
                dtype=int,
            ),
            np.ones(
                len(target_features),
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

    for (
        train_index,
        validation_index,
    ) in splitter.split(
        features,
        labels,
    ):
        classifier = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        )

        classifier.fit(
            features[
                train_index
            ],
            labels[
                train_index
            ],
        )

        probabilities[
            validation_index
        ] = classifier.predict_proba(
            features[
                validation_index
            ]
        )[:, 1]

    auc = float(
        roc_auc_score(
            labels,
            probabilities,
        )
    )

    return (
        probabilities[
            :len(source_features)
        ],
        probabilities[
            len(source_features):
        ],
        auc,
    )


def train_risk_model(
    features,
    source_error,
):
    model = RandomForestRegressor(
        n_estimators=1000,
        min_samples_leaf=8,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(
        features,
        source_error,
    )

    return model


def selective_curve(
    target_error,
    predicted_risk,
):
    order = np.argsort(
        predicted_risk
    )

    rows = []

    for coverage in [
        0.25,
        0.50,
        0.75,
        1.00,
    ]:
        n = max(
            1,
            int(
                np.floor(
                    coverage
                    * len(order)
                )
            ),
        )

        selected = order[:n]

        error = target_error[
            selected
        ]

        rows.append(
            {
                "coverage": coverage,
                "n": n,
                "mae": float(
                    error.mean()
                ),
                "median_absolute_error": float(
                    np.median(
                        error
                    )
                ),
                "p90_absolute_error": float(
                    np.quantile(
                        error,
                        0.90,
                    )
                ),
                "mean_predicted_risk": float(
                    predicted_risk[
                        selected
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_target(
    target_name,
    source,
    target,
    feature_columns,
    numeric_columns,
    categorical_columns,
    x_source_df,
    y_source,
    source_oof_prediction,
    source_oof_tree_std,
    source_oof_error,
    source_fold,
    final_prediction_model,
    source_feature_space,
    source_nn_distance,
    source_local_std,
    source_nearest_label,
):
    print()
    print("=" * 110)
    print(
        f"FINAL ADAPTIVE-DRST RELIABILITY TEST: "
        f"{SOURCE_DOMAIN} -> {target_name}"
    )
    print("=" * 110)

    x_target_df = target[
        feature_columns
    ].copy()

    y_target = pd.to_numeric(
        target[TARGET_COLUMN],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    target_prediction = (
        final_prediction_model.predict(
            x_target_df
        )
    )

    target_error = np.abs(
        target_prediction
        - y_target
    )

    target_tree_std = (
        tree_std_from_pipeline(
            final_prediction_model,
            x_target_df,
        )
    )

    global_preprocessor = (
        make_preprocessor(
            numeric_columns,
            categorical_columns,
        )
    )

    global_preprocessor.fit(
        x_source_df
    )

    target_feature_space = (
        np.asarray(
            global_preprocessor.transform(
                x_target_df
            ),
            dtype=np.float64,
        )
    )

    if (
        target_feature_space.shape[1]
        != source_feature_space.shape[1]
    ):
        raise RuntimeError(
            "Source/target transformed dimensions differ."
        )

    (
        target_nn_distance,
        target_local_std,
        target_nearest_label,
    ) = neighbor_target_features(
        source_feature_space,
        target_feature_space,
        y_source,
    )

    (
        source_domain_probability,
        target_domain_probability,
        domain_auc,
    ) = domain_probabilities(
        source_feature_space,
        target_feature_space,
    )

    source_signature = (
        make_signature(
            source,
            feature_columns,
        )
    )

    target_signature = (
        make_signature(
            target,
            feature_columns,
        )
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

    eps = 1e-6

    source_probability = (
        np.clip(
            source_domain_probability,
            eps,
            1.0 - eps,
        )
    )

    target_probability = (
        np.clip(
            target_domain_probability,
            eps,
            1.0 - eps,
        )
    )

    source_log_odds = np.log(
        source_probability
        /
        (
            1.0
            - source_probability
        )
    )

    target_log_odds = np.log(
        target_probability
        /
        (
            1.0
            - target_probability
        )
    )

    source_nearest_gap = np.abs(
        source_oof_prediction
        - source_nearest_label
    )

    target_nearest_gap = np.abs(
        target_prediction
        - target_nearest_label
    )

    strict_features = [
        "rf_tree_std",
        "nn_distance_mean",
        "local_label_std",
        "nearest_source_label_gap",
        "domain_target_probability",
        "log_target_source_odds",
        "unsupported_flag",
    ]

    full_features = (
        strict_features
        + [
            "absolute_prediction",
        ]
    )

    source_risk = pd.DataFrame(
        {
            "rf_tree_std":
                source_oof_tree_std,

            "nn_distance_mean":
                source_nn_distance,

            "local_label_std":
                source_local_std,

            "nearest_source_label_gap":
                source_nearest_gap,

            "domain_target_probability":
                source_domain_probability,

            "log_target_source_odds":
                source_log_odds,

            "unsupported_flag":
                (~source_supported).astype(
                    float
                ),

            "absolute_prediction":
                np.abs(
                    source_oof_prediction
                ),
        }
    )

    target_risk = pd.DataFrame(
        {
            "rf_tree_std":
                target_tree_std,

            "nn_distance_mean":
                target_nn_distance,

            "local_label_std":
                target_local_std,

            "nearest_source_label_gap":
                target_nearest_gap,

            "domain_target_probability":
                target_domain_probability,

            "log_target_source_odds":
                target_log_odds,

            "unsupported_flag":
                (~target_supported).astype(
                    float
                ),

            "absolute_prediction":
                np.abs(
                    target_prediction
                ),
        }
    )

    strict_model = train_risk_model(
        source_risk[
            strict_features
        ],
        source_oof_error,
    )

    full_model = train_risk_model(
        source_risk[
            full_features
        ],
        source_oof_error,
    )

    strict_source_oob = np.asarray(
        strict_model.oob_prediction_,
        dtype=float,
    )

    full_source_oob = np.asarray(
        full_model.oob_prediction_,
        dtype=float,
    )

    strict_target_risk = (
        strict_model.predict(
            target_risk[
                strict_features
            ]
        )
    )

    full_target_risk = (
        full_model.predict(
            target_risk[
                full_features
            ]
        )
    )

    source_scale = float(
        np.median(
            np.abs(
                y_source
            )
        )
    )

    source_relative_error = (
        source_oof_error
        /
        (
            np.abs(
                y_source
            )
            + source_scale
        )
    )

    target_relative_error = (
        target_error
        /
        (
            np.abs(
                y_target
            )
            + source_scale
        )
    )

    relative_model = train_risk_model(
        source_risk[
            strict_features
        ],
        source_relative_error,
    )

    relative_target_risk = (
        relative_model.predict(
            target_risk[
                strict_features
            ]
        )
    )

    source_error_q75 = float(
        np.quantile(
            source_oof_error,
            0.75,
        )
    )

    target_high_error = (
        target_error
        >= source_error_q75
    ).astype(int)

    base_metrics = regression_metrics(
        y_target,
        target_prediction,
    )

    strict_spearman = spearman(
        target_error,
        strict_target_risk,
    )

    full_spearman = spearman(
        target_error,
        full_target_risk,
    )

    tree_spearman = spearman(
        target_error,
        target_tree_std,
    )

    relative_spearman = spearman(
        target_relative_error,
        relative_target_risk,
    )

    strict_auc = safe_auc(
        target_high_error,
        strict_target_risk,
    )

    full_auc = safe_auc(
        target_high_error,
        full_target_risk,
    )

    tree_auc = safe_auc(
        target_high_error,
        target_tree_std,
    )

    order = np.argsort(
        strict_target_risk
    )

    quartile_n = max(
        1,
        len(target) // 4,
    )

    low_idx = order[
        :quartile_n
    ]

    high_idx = order[
        -quartile_n:
    ]

    low_risk_mae = float(
        target_error[
            low_idx
        ].mean()
    )

    high_risk_mae = float(
        target_error[
            high_idx
        ].mean()
    )

    high_low_ratio = (
        high_risk_mae
        / low_risk_mae
    )

    supported_mae = float(
        target_error[
            target_supported
        ].mean()
    ) if target_supported.any() else np.nan

    unsupported_mae = float(
        target_error[
            ~target_supported
        ].mean()
    ) if (~target_supported).any() else np.nan

    print()
    print("SOURCE-ONLY PREDICTION")
    print("-" * 110)

    for key, value in base_metrics.items():
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print()
    print("SHIFT / SUPPORT")
    print("-" * 110)

    print(
        f"Domain AUC: "
        f"{domain_auc:.6f}"
    )

    print(
        f"Exact supported target conditions: "
        f"{target_supported.sum()}/{len(target)}"
    )

    print(
        f"Supported target MAE: "
        f"{supported_mae:.6f}"
    )

    print(
        f"Unsupported target MAE: "
        f"{unsupported_mae:.6f}"
    )

    print()
    print("PRIMARY STRICT RELIABILITY MODEL")
    print("Prediction magnitude excluded")
    print("-" * 110)

    print(
        f"Target risk-error Spearman: "
        f"{strict_spearman:.6f}"
    )

    print(
        f"Target high-error AUROC: "
        f"{strict_auc:.6f}"
    )

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
        f"{high_low_ratio:.6f}"
    )

    print()
    print("ROBUSTNESS CHECKS")
    print("-" * 110)

    print(
        f"RF tree-std error Spearman: "
        f"{tree_spearman:.6f}"
    )

    print(
        f"RF tree-std high-error AUROC: "
        f"{tree_auc:.6f}"
    )

    print(
        f"Full model including |prediction| Spearman: "
        f"{full_spearman:.6f}"
    )

    print(
        f"Full model high-error AUROC: "
        f"{full_auc:.6f}"
    )

    print(
        f"Relative-error strict-risk Spearman: "
        f"{relative_spearman:.6f}"
    )

    curve = selective_curve(
        target_error,
        strict_target_risk,
    )

    print()
    print("STRICT SELECTIVE RISK / COVERAGE")
    print("-" * 110)

    print(
        curve.to_string(
            index=False
        )
    )

    threshold_rows = []

    print()
    print("SOURCE-DERIVED DEPLOYMENT THRESHOLDS")
    print("-" * 110)

    for quantile in [
        0.50,
        0.75,
        0.90,
    ]:
        threshold = float(
            np.quantile(
                strict_source_oob,
                quantile,
            )
        )

        accepted = (
            strict_target_risk
            <= threshold
        )

        n = int(
            accepted.sum()
        )

        if n == 0:
            row = {
                "source_risk_quantile":
                    quantile,
                "risk_threshold":
                    threshold,
                "target_coverage":
                    0.0,
                "accepted_n":
                    0,
                "accepted_mae":
                    np.nan,
                "rejected_n":
                    len(target),
                "rejected_mae":
                    float(
                        target_error.mean()
                    ),
            }
        else:
            rejected = (
                ~accepted
            )

            row = {
                "source_risk_quantile":
                    quantile,

                "risk_threshold":
                    threshold,

                "target_coverage":
                    float(
                        accepted.mean()
                    ),

                "accepted_n":
                    n,

                "accepted_mae":
                    float(
                        target_error[
                            accepted
                        ].mean()
                    ),

                "rejected_n":
                    int(
                        rejected.sum()
                    ),

                "rejected_mae":
                    float(
                        target_error[
                            rejected
                        ].mean()
                    )
                    if rejected.any()
                    else np.nan,
            }

        threshold_rows.append(
            row
        )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    print(
        threshold_table.to_string(
            index=False
        )
    )

    slug = (
        target_name
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    curve.to_csv(
        OUTPUT_DIR
        / f"{slug}_selective_curve.csv",
        index=False,
    )

    threshold_table.to_csv(
        OUTPUT_DIR
        / f"{slug}_deployment_thresholds.csv",
        index=False,
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
        "relative_error"
    ] = target_relative_error

    target_output[
        "strict_predicted_risk"
    ] = strict_target_risk

    target_output[
        "full_predicted_risk"
    ] = full_target_risk

    target_output[
        "relative_predicted_risk"
    ] = relative_target_risk

    target_output[
        "rf_tree_std"
    ] = target_tree_std

    target_output[
        "exact_source_support"
    ] = target_supported

    target_output[
        "domain_target_probability"
    ] = target_domain_probability

    target_output.to_csv(
        OUTPUT_DIR
        / f"{slug}_target_predictions.csv",
        index=False,
    )

    strict_importance = pd.DataFrame(
        {
            "feature":
                strict_features,

            "importance":
                strict_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    strict_importance.to_csv(
        OUTPUT_DIR
        / f"{slug}_strict_risk_feature_importance.csv",
        index=False,
    )

    return {
        "source":
            SOURCE_DOMAIN,

        "target":
            target_name,

        "target_n":
            len(target),

        "target_mae":
            base_metrics[
                "mae"
            ],

        "target_rmse":
            base_metrics[
                "rmse"
            ],

        "target_r2":
            base_metrics[
                "r2"
            ],

        "domain_auc":
            domain_auc,

        "supported_n":
            int(
                target_supported.sum()
            ),

        "unsupported_n":
            int(
                (~target_supported).sum()
            ),

        "supported_mae":
            supported_mae,

        "unsupported_mae":
            unsupported_mae,

        "strict_risk_error_spearman":
            strict_spearman,

        "strict_high_error_auc":
            strict_auc,

        "tree_std_error_spearman":
            tree_spearman,

        "tree_std_high_error_auc":
            tree_auc,

        "full_risk_error_spearman":
            full_spearman,

        "relative_risk_error_spearman":
            relative_spearman,

        "low_risk_quartile_mae":
            low_risk_mae,

        "high_risk_quartile_mae":
            high_risk_mae,

        "high_low_mae_ratio":
            high_low_ratio,
    }


def main():
    print("=" * 110)
    print("ADAPTIVE-DRST FINAL ENVIRONMENTAL RELIABILITY EXPERIMENT")
    print("=" * 110)

    print(
        "Primary reliability model excludes absolute prediction magnitude."
    )

    print(
        "Target labels are used only after prediction and risk scores are frozen."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    source = (
        df[
            df[
                DOMAIN_COLUMN
            ]
            == SOURCE_DOMAIN
        ]
        .copy()
        .reset_index(
            drop=True
        )
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
        and not column.startswith(
            "_"
        )
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

    y_source = pd.to_numeric(
        source[
            TARGET_COLUMN
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    (
        source_oof_prediction,
        source_oof_tree_std,
        source_fold,
    ) = build_source_oof(
        x_source_df,
        y_source,
        numeric_columns,
        categorical_columns,
    )

    source_oof_error = np.abs(
        source_oof_prediction
        - y_source
    )

    print()
    print("SOURCE OOF BASELINE")
    print("-" * 110)

    print(
        f"Source conditions: "
        f"{len(source)}"
    )

    print(
        f"Source OOF MAE: "
        f"{source_oof_error.mean():.6f}"
    )

    final_prediction_model = (
        make_prediction_pipeline(
            numeric_columns,
            categorical_columns,
        )
    )

    final_prediction_model.fit(
        x_source_df,
        y_source,
    )

    global_preprocessor = (
        make_preprocessor(
            numeric_columns,
            categorical_columns,
        )
    )

    source_feature_space = (
        np.asarray(
            global_preprocessor
            .fit_transform(
                x_source_df
            ),
            dtype=np.float64,
        )
    )

    (
        source_nn_distance,
        source_local_std,
        source_nearest_label,
    ) = neighbor_source_features(
        source_feature_space,
        y_source,
    )

    summaries = []

    for target_name in TARGET_DOMAINS:
        target = (
            df[
                df[
                    DOMAIN_COLUMN
                ]
                == target_name
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if len(target) == 0:
            raise RuntimeError(
                f"No rows found for target domain {target_name}"
            )

        result = evaluate_target(
            target_name=target_name,
            source=source,
            target=target,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            x_source_df=x_source_df,
            y_source=y_source,
            source_oof_prediction=source_oof_prediction,
            source_oof_tree_std=source_oof_tree_std,
            source_oof_error=source_oof_error,
            source_fold=source_fold,
            final_prediction_model=final_prediction_model,
            source_feature_space=source_feature_space,
            source_nn_distance=source_nn_distance,
            source_local_std=source_local_std,
            source_nearest_label=source_nearest_label,
        )

        summaries.append(
            result
        )

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 110)
    print("CROSS-DOMAIN FINAL SUMMARY")
    print("=" * 110)

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        OUTPUT_DIR
        / "cross_domain_reliability_summary.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "final_experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "source_domain":
                    SOURCE_DOMAIN,

                "target_domains":
                    TARGET_DOMAINS,

                "source_conditions":
                    len(source),

                "features":
                    feature_columns,

                "strict_risk_features":
                    [
                        "rf_tree_std",
                        "nn_distance_mean",
                        "local_label_std",
                        "nearest_source_label_gap",
                        "domain_target_probability",
                        "log_target_source_odds",
                        "unsupported_flag",
                    ],

                "target_labels_used_for_training":
                    False,

                "prediction_magnitude_in_primary_risk_model":
                    False,
            },
            handle,
            indent=2,
        )

    print()
    print("=" * 110)
    print("FINAL INTERPRETATION RULE")
    print("=" * 110)

    print(
        "The primary result is the strict source-derived risk model."
    )

    print(
        "It must rank target error without access to target labels "
        "and without using absolute prediction magnitude."
    )

    print(
        "Secondary Effluent is the primary environmental shift."
    )

    print(
        "Ground Water is the independent matrix-shift replication."
    )

    print(
        "Exact support and domain probability remain diagnostic "
        "signals rather than assumed reliability measures."
    )


if __name__ == "__main__":
    main()
