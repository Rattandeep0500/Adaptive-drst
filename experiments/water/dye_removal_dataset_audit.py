from pathlib import Path
import json

import numpy as np
import pandas as pd

from aqua_fetch import dye_removal


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "water" / "photocatalysis"
OUTPUT_DIR = DATA_ROOT / "data"

OUTPUT_FILE = OUTPUT_DIR / "dye_removal_raw.csv"
AUDIT_FILE = OUTPUT_DIR / "dye_removal_audit.json"


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("WATER-DRST DYE REMOVAL DATASET AUDIT")
    print("=" * 80)

    print(
        "Loading AquaFetch dye_removal dataset..."
    )

    result = dye_removal()

    if isinstance(result, tuple):
        data = result[0]
        encoders = result[1:]
    else:
        data = result
        encoders = []

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    print_section("DATASET SIZE")

    print(
        f"ROWS: {len(data)}"
    )

    print(
        f"COLUMNS: {len(data.columns)}"
    )

    print_section("COLUMNS")

    for column in data.columns:
        print(column)

    print_section("DATA TYPES")

    print(
        data.dtypes.to_string()
    )

    print_section("MISSING VALUES")

    print(
        data.isna()
        .sum()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print_section("UNIQUE VALUE COUNTS")

    unique_counts = {}

    for column in data.columns:
        count = int(
            data[column].nunique(
                dropna=True
            )
        )

        unique_counts[column] = count

        print(
            f"{column}: {count}"
        )

    categorical = [
        column
        for column in data.columns
        if data[column].dtype == object
    ]

    print_section("CATEGORICAL VARIABLES")

    for column in categorical:
        print()
        print(
            f"{column}:"
        )

        print(
            data[column]
            .value_counts(
                dropna=False
            )
            .head(50)
            .to_string()
        )

    print_section("NUMERIC SUMMARY")

    numeric = data.select_dtypes(
        include=[np.number]
    )

    if len(numeric.columns):
        numeric_summary = numeric.describe().T

        print(
            numeric_summary[
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

    print_section("POTENTIAL TARGET VARIABLES")

    target_keywords = [
        "k",
        "rate",
        "removal",
        "degradation",
        "efficiency",
        "percent",
        "yield",
        "time",
    ]

    target_candidates = []

    for column in data.columns:
        lowered = column.lower()

        if any(
            keyword in lowered
            for keyword in target_keywords
        ):
            target_candidates.append(
                column
            )

    for column in target_candidates:
        print()
        print(
            column
        )

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            finite_values = values[
                np.isfinite(
                    values
                )
            ]

            print(
                f"COUNT: {len(finite_values)}"
            )

            if len(finite_values):
                print(
                    f"MIN: {finite_values.min()}"
                )

                print(
                    f"MAX: {finite_values.max()}"
                )

                print(
                    f"MEAN: {finite_values.mean()}"
                )

                print(
                    f"MEDIAN: {finite_values.median()}"
                )

    print_section("POTENTIAL DOMAIN VARIABLES")

    domain_keywords = [
        "dye",
        "pollut",
        "catalyst",
        "light",
        "ph",
        "concentration",
        "temperature",
        "water",
        "anion",
        "matrix",
        "source",
    ]

    domain_candidates = []

    for column in data.columns:
        lowered = column.lower()

        if any(
            keyword in lowered
            for keyword in domain_keywords
        ):
            domain_candidates.append(
                column
            )

    for column in domain_candidates:
        print()
        print(
            f"{column}:"
        )

        if data[column].dtype == object:
            print(
                data[column]
                .value_counts(
                    dropna=False
                )
                .head(50)
                .to_string()
            )
        else:
            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            print(
                f"UNIQUE: "
                f"{values.nunique(dropna=True)}"
            )

            print(
                f"MIN: "
                f"{values.min()}"
            )

            print(
                f"MAX: "
                f"{values.max()}"
            )

    print_section("EXACT DUPLICATE ROWS")

    duplicate_rows = int(
        data.duplicated().sum()
    )

    print(
        f"EXACT DUPLICATES: "
        f"{duplicate_rows}"
    )

    print_section("DUPLICATE CONDITION ANALYSIS")

    condition_columns = [
        column
        for column in data.columns
        if column not in target_candidates
    ]

    if condition_columns:
        condition_key = (
            data[
                condition_columns
            ]
            .astype(str)
            .agg(
                "||".join,
                axis=1,
            )
        )

        condition_counts = (
            condition_key
            .value_counts()
        )

        print(
            f"UNIQUE NON-TARGET CONDITIONS: "
            f"{len(condition_counts)}"
        )

        print(
            f"CONDITIONS WITH REPEATS: "
            f"{int((condition_counts > 1).sum())}"
        )

    print_section("SAMPLE ROWS")

    print(
        data.head(15).to_string(
            index=False
        )
    )

    print_section("POTENTIAL DYE SPLIT")

    dye_columns = [
        column
        for column in data.columns
        if "dye" in column.lower()
        or "pollut" in column.lower()
    ]

    if dye_columns:
        for column in dye_columns:
            print()
            print(
                f"DYE/POLLUTANT CANDIDATE: "
                f"{column}"
            )

            print(
                data[column]
                .value_counts(
                    dropna=False
                )
                .head(50)
                .to_string()
            )
    else:
        print(
            "No obvious dye/pollutant column found."
        )

    print_section("ENCODER INFORMATION")

    print(
        f"ENCODER OBJECTS: "
        f"{len(encoders)}"
    )

    for index, encoder in enumerate(
        encoders
    ):
        print()
        print(
            f"ENCODER {index}: "
            f"{type(encoder).__name__}"
        )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit = {
        "dataset": "AquaFetch dye_removal",
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "columns_list": list(data.columns),
        "dtypes": {
            column: str(
                data[column].dtype
            )
            for column in data.columns
        },
        "missing_values": {
            column: int(
                data[column].isna().sum()
            )
            for column in data.columns
        },
        "unique_values": unique_counts,
        "categorical_variables": categorical,
        "target_candidates": target_candidates,
        "domain_candidates": domain_candidates,
        "exact_duplicate_rows": duplicate_rows,
        "encoder_count": len(encoders),
    }

    with AUDIT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
        )

    print_section("AUDIT COMPLETE")

    print(
        f"RAW DATA: {OUTPUT_FILE}"
    )

    print(
        f"AUDIT: {AUDIT_FILE}"
    )


if __name__ == "__main__":
    main()