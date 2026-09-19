from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
import pandas as pd

from pymatgen.core import Structure

from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
DATA_DIR = DATA_ROOT / "data"
CIF_DIR = DATA_DIR / "cif_file"

PRETRAIN_FILE = DATA_DIR / "pretrain_data.json"
TRAIN_FILE = DATA_DIR / "fine_tune" / "train_data.json"
TEST_FILE = DATA_DIR / "fine_tune" / "test_data.json"

OUTPUT_DIR = DATA_ROOT / "features"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "realmat_bag_rich_structure_features.parquet"
)

FAILURE_FILE = (
    OUTPUT_DIR
    / "realmat_bag_rich_structure_feature_failures.json"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "realmat_bag_rich_structure_features_checkpoint.parquet"
)

RANDOM_STATE = 42

BATCH_SIZE = 1000


def load_records(path, domain):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for mpid, value in data.items():
        rows.append(
            {
                "mpid": str(mpid),
                "bandgap": float(value["bg"]),
                "domain": domain,
            }
        )

    return rows


def finite_or_nan(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:
        return np.nan


def featurize_composition(composition, featurizers):
    result = {}

    for featurizer in featurizers:
        labels = featurizer.feature_labels()
        values = featurizer.featurize(composition)

        prefix = featurizer.__class__.__name__.lower()

        if np.isscalar(values):
            values = [values]

        values = list(values)

        if len(labels) == len(values):
            for label, value in zip(labels, values):
                result[
                    f"{prefix}_{label}"
                ] = finite_or_nan(value)
        else:
            for index, value in enumerate(values):
                result[
                    f"{prefix}_{index}"
                ] = finite_or_nan(value)

    return result


def clean_numeric_features(dataframe):
    id_columns = [
        "mpid",
        "bandgap",
        "domain",
        "formula",
    ]

    numeric_columns = [
        column
        for column in dataframe.columns
        if column not in id_columns
    ]

    numeric = dataframe[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    missing_fraction = numeric.isna().mean()

    keep_columns = [
        column
        for column in numeric.columns
        if missing_fraction[column] <= 0.25
    ]

    numeric = numeric[keep_columns]

    numeric = numeric.fillna(
        numeric.median(
            numeric_only=True
        )
    )

    constant_columns = [
        column
        for column in numeric.columns
        if numeric[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:
        numeric = numeric.drop(
            columns=constant_columns
        )

    final_dataframe = pd.concat(
        [
            dataframe[
                [
                    "mpid",
                    "bandgap",
                    "domain",
                    "formula",
                ]
            ].reset_index(drop=True),
            numeric.reset_index(drop=True),
        ],
        axis=1,
    )

    return final_dataframe, keep_columns, constant_columns


def main():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(
        RANDOM_STATE
    )

    records = []

    records.extend(
        load_records(
            PRETRAIN_FILE,
            "computational",
        )
    )

    records.extend(
        load_records(
            TRAIN_FILE,
            "experimental_train",
        )
    )

    records.extend(
        load_records(
            TEST_FILE,
            "experimental_test",
        )
    )

    print("=" * 80)
    print("REALMAT-BAG RICH MATERIALS FEATURE EXTRACTION")
    print("=" * 80)
    print(
        f"TOTAL RECORDS: {len(records)}"
    )

    featurizers = [
        ElementProperty.from_preset(
            "magpie"
        ),
        Stoichiometry(),
        ValenceOrbital(),
        IonProperty(),
    ]

    rows = []
    failures = []

    total = len(records)

    for index, record in enumerate(
        records,
        start=1,
    ):
        mpid = record["mpid"]

        cif_path = (
            CIF_DIR
            / f"{mpid}.cif"
        )

        try:
            structure = Structure.from_file(
                cif_path
            )

            composition = (
                structure.composition
            )

            row = {
                "mpid": mpid,
                "bandgap": record["bandgap"],
                "domain": record["domain"],
                "formula": composition.reduced_formula,
            }

            row.update(
                featurize_composition(
                    composition,
                    featurizers,
                )
            )

            rows.append(
                row
            )

        except Exception as exc:
            failures.append(
                {
                    "mpid": mpid,
                    "domain": record["domain"],
                    "error": str(exc),
                }
            )

        if (
            index % 100 == 0
            or index == total
        ):
            print(
                f"PROCESSED: "
                f"{index} / {total}"
            )

        if (
            index % BATCH_SIZE == 0
            or index == total
        ):
            checkpoint_dataframe = pd.DataFrame(
                rows
            )

            checkpoint_dataframe.to_parquet(
                CHECKPOINT_FILE,
                index=False,
            )

    dataframe = pd.DataFrame(
        rows
    )

    print()
    print("=" * 80)
    print("FIRST-PASS RESULT")
    print("=" * 80)

    print(
        f"SUCCESSFUL: "
        f"{len(dataframe)}"
    )

    print(
        f"FAILED: "
        f"{len(failures)}"
    )

    print(
        f"COLUMNS BEFORE CLEANING: "
        f"{len(dataframe.columns)}"
    )

    if len(dataframe) == 0:
        raise RuntimeError(
            "No records were successfully featurized."
        )

    final_dataframe, retained_columns, constant_columns = (
        clean_numeric_features(
            dataframe
        )
    )

    final_dataframe.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    with FAILURE_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            failures,
            f,
            indent=2,
        )

    digest = hashlib.sha256()

    with OUTPUT_FILE.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        f"ROWS: "
        f"{len(final_dataframe)}"
    )

    print(
        f"FINAL COLUMNS: "
        f"{len(final_dataframe.columns)}"
    )

    print(
        f"FINAL NUMERIC FEATURES: "
        f"{len(retained_columns)}"
    )

    print(
        f"CONSTANT FEATURES DROPPED: "
        f"{len(constant_columns)}"
    )

    print()
    print("DOMAIN COUNTS:")
    print(
        final_dataframe[
            "domain"
        ].value_counts()
    )

    print()
    print("MISSING VALUES:")
    print(
        final_dataframe.isna().sum()
        .sort_values(
            ascending=False
        )
        .head(20)
    )

    print()
    print(
        f"FAILURE RECORDS: "
        f"{len(failures)}"
    )

    print(
        f"OUTPUT: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"SHA256: "
        f"{digest.hexdigest()}"
    )

    print()
    print("=" * 80)
    print("RICH FEATURE EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()