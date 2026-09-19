from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "water"
    / "biochar_ec"
    / "Raw_data.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "water"
    / "biochar_ec"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "Capacity"
FINAL_CONCENTRATION = "Final concentration"
WASTEWATER = "Wastewater type"
POLLUTANT = "Pollutant"
ADSORBENT = "Adsorbent"
ADSORPTION_TYPE = "Adsorption type"


def canonicalize(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")

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


def signature(df, columns):
    canonical = pd.DataFrame(index=df.index)

    for col in columns:
        canonical[col] = canonicalize(df[col])

    return pd.util.hash_pandas_object(
        canonical,
        index=False,
    ).astype("uint64")


def value_counts_report(df, column):
    print()
    print(column.upper())
    print("-" * 100)

    counts = df[column].value_counts(dropna=False)

    for value, count in counts.items():
        pct = 100.0 * count / len(df)

        print(
            f"{value}: {count} ({pct:.2f}%)"
        )


def set_overlap(a, b):
    a = {
        str(x).strip()
        for x in a.dropna()
    }

    b = {
        str(x).strip()
        for x in b.dropna()
    }

    shared = a & b
    union = a | b

    return (
        len(a),
        len(b),
        len(shared),
        len(shared) / len(union)
        if union else np.nan,
    )


def pairwise_matrix_audit(df):
    categories = [
        x
        for x in df[WASTEWATER].dropna().unique()
    ]

    signature_columns = [
        col
        for col in df.columns
        if col not in {
            WASTEWATER,
            TARGET,
            FINAL_CONCENTRATION,
        }
    ]

    work = df.copy()

    work["_nonmatrix_signature"] = signature(
        work,
        signature_columns,
    )

    rows = []

    for a, b in combinations(categories, 2):
        left = work[
            work[WASTEWATER] == a
        ]

        right = work[
            work[WASTEWATER] == b
        ]

        (
            ads_a,
            ads_b,
            ads_shared,
            ads_jaccard,
        ) = set_overlap(
            left[ADSORBENT],
            right[ADSORBENT],
        )

        (
            pol_a,
            pol_b,
            pol_shared,
            pol_jaccard,
        ) = set_overlap(
            left[POLLUTANT],
            right[POLLUTANT],
        )

        left_sigs = set(
            left["_nonmatrix_signature"]
        )

        right_sigs = set(
            right["_nonmatrix_signature"]
        )

        shared_sigs = (
            left_sigs
            & right_sigs
        )

        rows.append({
            "domain_a": a,
            "domain_b": b,
            "n_a": len(left),
            "n_b": len(right),
            "adsorbents_a": ads_a,
            "adsorbents_b": ads_b,
            "shared_adsorbents": ads_shared,
            "adsorbent_jaccard": ads_jaccard,
            "pollutants_a": pol_a,
            "pollutants_b": pol_b,
            "shared_pollutants": pol_shared,
            "pollutant_jaccard": pol_jaccard,
            "shared_exact_nonmatrix_signatures": (
                len(shared_sigs)
            ),
            "rows_a_in_exact_overlap": int(
                left[
                    "_nonmatrix_signature"
                ].isin(shared_sigs).sum()
            ),
            "rows_b_in_exact_overlap": int(
                right[
                    "_nonmatrix_signature"
                ].isin(shared_sigs).sum()
            ),
        })

    return pd.DataFrame(rows)


def pairwise_pollutant_audit(df):
    pollutants = [
        x
        for x in df[POLLUTANT].dropna().unique()
    ]

    signature_columns = [
        col
        for col in df.columns
        if col not in {
            POLLUTANT,
            TARGET,
            FINAL_CONCENTRATION,
        }
    ]

    work = df.copy()

    work["_nonpollutant_signature"] = signature(
        work,
        signature_columns,
    )

    rows = []

    for a, b in combinations(pollutants, 2):
        left = work[
            work[POLLUTANT] == a
        ]

        right = work[
            work[POLLUTANT] == b
        ]

        (
            ads_a,
            ads_b,
            ads_shared,
            ads_jaccard,
        ) = set_overlap(
            left[ADSORBENT],
            right[ADSORBENT],
        )

        (
            water_a,
            water_b,
            water_shared,
            water_jaccard,
        ) = set_overlap(
            left[WASTEWATER],
            right[WASTEWATER],
        )

        left_sigs = set(
            left["_nonpollutant_signature"]
        )

        right_sigs = set(
            right["_nonpollutant_signature"]
        )

        shared_sigs = (
            left_sigs
            & right_sigs
        )

        rows.append({
            "pollutant_a": a,
            "pollutant_b": b,
            "n_a": len(left),
            "n_b": len(right),
            "shared_adsorbents": ads_shared,
            "adsorbent_jaccard": ads_jaccard,
            "shared_wastewater_types": (
                water_shared
            ),
            "wastewater_jaccard": (
                water_jaccard
            ),
            "shared_exact_nonpollutant_signatures": (
                len(shared_sigs)
            ),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST BIOCHAR / EMERGING-CONTAMINANT DATASET AUDIT")
    print("=" * 100)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print()
    print(f"RAW ROWS: {len(df)}")
    print(f"RAW COLUMNS: {len(df.columns)}")

    print()
    print("COLUMNS")
    print("-" * 100)

    for i, col in enumerate(df.columns):
        print(f"{i:02d}: {col}")

    required = {
        TARGET,
        FINAL_CONCENTRATION,
        WASTEWATER,
        POLLUTANT,
        ADSORBENT,
        ADSORPTION_TYPE,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    print()
    print("=" * 100)
    print("DATA QUALITY")
    print("=" * 100)

    exact_duplicates = int(
        df.duplicated().sum()
    )

    print(
        f"Exact duplicate rows: "
        f"{exact_duplicates}"
    )

    print(
        f"Rows containing any missing value: "
        f"{df.isna().any(axis=1).sum()}"
    )

    missing_report = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_report = missing_report[
        missing_report > 0
    ]

    if len(missing_report):
        print()
        print("Missing values by column:")
        print(
            missing_report.to_string()
        )
    else:
        print("Missing values: none")

    value_counts_report(
        df,
        WASTEWATER,
    )

    value_counts_report(
        df,
        POLLUTANT,
    )

    value_counts_report(
        df,
        ADSORBENT,
    )

    value_counts_report(
        df,
        ADSORPTION_TYPE,
    )

    print()
    print("=" * 100)
    print("TARGET AUDIT")
    print("=" * 100)

    target = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    print(
        f"Valid target values: "
        f"{target.notna().sum()}"
    )

    print(
        f"Missing target values: "
        f"{target.isna().sum()}"
    )

    print(
        f"Negative capacities: "
        f"{(target < 0).sum()}"
    )

    print(
        f"Zero capacities: "
        f"{(target == 0).sum()}"
    )

    print(
        f"Minimum capacity: "
        f"{target.min():.6f}"
    )

    print(
        f"Median capacity: "
        f"{target.median():.6f}"
    )

    print(
        f"Mean capacity: "
        f"{target.mean():.6f}"
    )

    print(
        f"Maximum capacity: "
        f"{target.max():.6f}"
    )

    print()
    print("Capacity quantiles:")

    print(
        target.quantile(
            [
                0.00,
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
                1.00,
            ]
        ).to_string()
    )

    input_columns = [
        col
        for col in df.columns
        if col not in {
            TARGET,
            FINAL_CONCENTRATION,
        }
    ]

    df["_input_signature"] = signature(
        df,
        input_columns,
    )

    unique_conditions = int(
        df["_input_signature"].nunique()
    )

    print()
    print("=" * 100)
    print("EXPERIMENTAL-CONDITION AUDIT")
    print("=" * 100)

    print(
        f"Unique recorded input conditions: "
        f"{unique_conditions}"
    )

    condition_sizes = (
        df.groupby(
            "_input_signature"
        )
        .size()
    )

    print(
        f"Input conditions represented once: "
        f"{(condition_sizes == 1).sum()}"
    )

    print(
        f"Repeated input conditions: "
        f"{(condition_sizes > 1).sum()}"
    )

    grouped = (
        df.assign(
            _target_numeric=target
        )
        .groupby(
            "_input_signature",
            as_index=False,
        )
        .agg(
            n_rows=(
                "_input_signature",
                "size",
            ),
            target_unique=(
                "_target_numeric",
                "nunique",
            ),
            target_min=(
                "_target_numeric",
                "min",
            ),
            target_max=(
                "_target_numeric",
                "max",
            ),
        )
    )

    grouped[
        "target_spread"
    ] = (
        grouped["target_max"]
        - grouped["target_min"]
    )

    repeated = grouped[
        grouped["n_rows"] > 1
    ]

    conflicting = repeated[
        repeated["target_unique"] > 1
    ]

    print(
        f"Repeated conditions with conflicting capacity: "
        f"{len(conflicting)}"
    )

    if len(conflicting):
        print(
            f"Largest capacity spread within "
            f"identical recorded inputs: "
            f"{conflicting['target_spread'].max():.6f}"
        )

    grouped.to_csv(
        OUTPUT_DIR
        / "biochar_ec_input_condition_audit.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("WASTEWATER-TYPE DOMAIN AUDIT")
    print("=" * 100)

    matrix_pairs = pairwise_matrix_audit(
        df.drop(
            columns=["_input_signature"]
        )
    )

    if matrix_pairs.empty:
        print(
            "Only one wastewater type is present."
        )
    else:
        matrix_pairs = matrix_pairs.sort_values(
            [
                "shared_exact_nonmatrix_signatures",
                "shared_pollutants",
                "shared_adsorbents",
            ],
            ascending=False,
        ).reset_index(drop=True)

        print(
            matrix_pairs.to_string(
                index=False
            )
        )

        matrix_pairs.to_csv(
            OUTPUT_DIR
            / "biochar_ec_wastewater_domain_pairs.csv",
            index=False,
        )

    print()
    print("=" * 100)
    print("POLLUTANT-SHIFT AUDIT")
    print("=" * 100)

    pollutant_pairs = pairwise_pollutant_audit(
        df.drop(
            columns=["_input_signature"]
        )
    )

    pollutant_pairs = (
        pollutant_pairs
        .sort_values(
            [
                "shared_adsorbents",
                "shared_wastewater_types",
                "n_a",
                "n_b",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        pollutant_pairs.head(
            30
        ).to_string(
            index=False
        )
    )

    pollutant_pairs.to_csv(
        OUTPUT_DIR
        / "biochar_ec_pollutant_domain_pairs.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)

    print(
        "No source/target domain has been selected."
    )

    print(
        "No model has been trained."
    )

    print(
        "The next decision will be based on domain sample size, "
        "material overlap, pollutant overlap, exact covariate support, "
        "repeated-condition structure, and confounding."
    )


if __name__ == "__main__":
    main()
