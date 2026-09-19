from pathlib import Path
import json

import numpy as np
import pandas as pd

from aqua_fetch import mg_degradation


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "water" / "photocatalysis"
OUTPUT_DIR = DATA_ROOT / "data"

OUTPUT_FILE = OUTPUT_DIR / "mg_degradation_raw.csv"
AUDIT_FILE = OUTPUT_DIR / "mg_degradation_audit.json"


def serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("WATER-DRST PHOTOCATALYSIS DATASET AUDIT")
    print("=" * 80)

    print("Downloading/loading AquaFetch mg_degradation dataset...")

    data, encoders = mg_degradation()

    print()
    print("=" * 80)
    print("DATASET SIZE")
    print("=" * 80)

    print(
        f"ROWS: {len(data)}"
    )

    print(
        f"COLUMNS: {len(data.columns)}"
    )

    print()
    print("=" * 80)
    print("COLUMNS")
    print("=" * 80)

    for column in data.columns:
        print(column)

    print()
    print("=" * 80)
    print("DATA TYPES")
    print("=" * 80)

    print(
        data.dtypes.to_string()
    )

    print()
    print("=" * 80)
    print("MISSING VALUES")
    print("=" * 80)

    print(
        data.isna()
        .sum()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print()
    print("=" * 80)
    print("UNIQUE VALUES")
    print("=" * 80)

    unique_summary = {}

    for column in data.columns:
        unique_summary[column] = int(
            data[column].nunique(
                dropna=True
            )
        )

        print(
            f"{column}: "
            f"{unique_summary[column]}"
        )

    categorical_candidates = [
        column
        for column in data.columns
        if data[column].dtype == object
    ]

    print()
    print("=" * 80)
    print("CATEGORICAL FEATURES")
    print("=" * 80)

    for column in categorical_candidates:
        print()
        print(
            f"{column}:"
        )

        counts = (
            data[column]
            .value_counts(
                dropna=False
            )
        )

        print(
            counts.head(30).to_string()
        )

    print()
    print("=" * 80)
    print("NUMERIC SUMMARY")
    print("=" * 80)

    numeric = data.select_dtypes(
        include=[np.number]
    )

    if len(numeric.columns):
        summary = numeric.describe().T

        print(
            summary[
                [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ]
            ].to_string()
        )

    print()
    print("=" * 80)
    print("POTENTIAL WATER-MATRIX VARIABLES")
    print("=" * 80)

    matrix_candidates = [
        column
        for column in data.columns
        if any(
            token in column.lower()
            for token in [
                "ha",
                "anion",
                "water",
                "ph",
                "ion",
                "concentration",
            ]
        )
    ]

    for column in matrix_candidates:
        print(
            f"{column}: "
            f"{data[column].nunique(dropna=True)} unique values"
        )

    print()
    print("=" * 80)
    print("TARGET CANDIDATES")
    print("=" * 80)

    target_candidates = [
        column
        for column in data.columns
        if any(
            token in column.lower()
            for token in [
                "efficiency",
                "k_first",
                "k_2nd",
                "removal",
                "degradation",
            ]
        )
    ]

    for column in target_candidates:
        print(
            f"{column}"
        )

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            values = data[column].dropna()

            if len(values):
                print(
                    f"  MIN: {values.min()}"
                )
                print(
                    f"  MAX: {values.max()}"
                )
                print(
                    f"  MEAN: {values.mean()}"
                )
                print(
                    f"  MEDIAN: {values.median()}"
                )

    print()
    print("=" * 80)
    print("HUMIC ACID AUDIT")
    print("=" * 80)

    ha_columns = [
        column
        for column in data.columns
        if "ha" in column.lower()
    ]

    for column in ha_columns:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        print(
            f"{column}"
        )

        print(
            f"  ZERO: "
            f"{int((values == 0).sum())}"
        )

        print(
            f"  NONZERO: "
            f"{int((values > 0).sum())}"
        )

        print(
            f"  MIN: "
            f"{values.min()}"
        )

        print(
            f"  MAX: "
            f"{values.max()}"
        )

    print()
    print("=" * 80)
    print("ANION AUDIT")
    print("=" * 80)

    anion_columns = [
        column
        for column in data.columns
        if "anion" in column.lower()
    ]

    for column in anion_columns:
        print(
            f"{column}"
        )

        print(
            data[column]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()
    print("=" * 80)
    print("SAMPLE ROWS")
    print("=" * 80)

    print(
        data.head(10).to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("DOMAIN-SHIFT CANDIDATE")
    print("=" * 80)

    if ha_columns:
        ha_column = ha_columns[0]

        ha_values = pd.to_numeric(
            data[ha_column],
            errors="coerce",
        )

        simple_matrix = (
            ha_values.fillna(0) == 0
        )

        complex_matrix = (
            ha_values.fillna(0) > 0
        )

        print(
            f"MATRIX VARIABLE: "
            f"{ha_column}"
        )

        print(
            f"SIMPLE MATRIX SAMPLES: "
            f"{int(simple_matrix.sum())}"
        )

        print(
            f"HUMIC-ACID MATRIX SAMPLES: "
            f"{int(complex_matrix.sum())}"
        )

        if simple_matrix.sum() > 0:
            print(
                f"SIMPLE MATRIX MEAN "
                f"TARGET CANDIDATE VALUES:"
            )

        if complex_matrix.sum() > 0:
            print(
                f"HUMIC-ACID MATRIX MEAN "
                f"TARGET CANDIDATE VALUES:"
            )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit = {
        "dataset": "AquaFetch mg_degradation",
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "columns_list": list(data.columns),
        "dtypes": {
            column: str(data[column].dtype)
            for column in data.columns
        },
        "missing_values": {
            column: int(
                data[column].isna().sum()
            )
            for column in data.columns
        },
        "unique_values": unique_summary,
        "categorical_candidates": categorical_candidates,
        "target_candidates": target_candidates,
        "matrix_candidates": matrix_candidates,
    }

    with AUDIT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            default=serializable,
        )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"RAW DATA: {OUTPUT_FILE}"
    )

    print(
        f"AUDIT: {AUDIT_FILE}"
    )


if __name__ == "__main__":
    main()