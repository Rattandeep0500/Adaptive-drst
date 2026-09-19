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
DOMAIN = "Wastewater type"
POLLUTANT = "Pollutant"
ADSORBENT = "Adsorbent"
ADSORPTION_TYPE = "Adsorption type"

CANDIDATE_PAIRS = [
    ("Lake water", "Secondary effluent"),
    ("Lake water", "Ground water"),
]


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

    for col in columns:
        canonical[col] = canonicalize(df[col])

    return pd.util.hash_pandas_object(
        canonical,
        index=False,
    ).astype("uint64")


def collapse_conditions(df):
    input_columns = [
        col
        for col in df.columns
        if col not in {
            TARGET,
            FINAL_CONCENTRATION,
        }
    ]

    work = df.copy()

    work["_target_numeric"] = pd.to_numeric(
        work[TARGET],
        errors="coerce",
    )

    work["_final_numeric"] = pd.to_numeric(
        work[FINAL_CONCENTRATION],
        errors="coerce",
    )

    work["_condition_signature"] = make_signature(
        work,
        input_columns,
    )

    rows = []

    for signature, group in work.groupby(
        "_condition_signature",
        sort=False,
    ):
        target = group["_target_numeric"].dropna()

        final = group["_final_numeric"].dropna()

        base = group.iloc[0]

        row = {
            col: base[col]
            for col in input_columns
        }

        row["_condition_signature"] = int(signature)
        row["n_replicates"] = int(len(group))

        row[TARGET] = (
            float(target.median())
            if len(target)
            else np.nan
        )

        row["capacity_mean"] = (
            float(target.mean())
            if len(target)
            else np.nan
        )

        row["capacity_std"] = (
            float(target.std(ddof=1))
            if len(target) > 1
            else 0.0
        )

        row["capacity_min"] = (
            float(target.min())
            if len(target)
            else np.nan
        )

        row["capacity_max"] = (
            float(target.max())
            if len(target)
            else np.nan
        )

        row["capacity_spread"] = (
            float(target.max() - target.min())
            if len(target)
            else np.nan
        )

        row[FINAL_CONCENTRATION] = (
            float(final.median())
            if len(final)
            else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)


def describe_target(df, title):
    y = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    ).dropna()

    print()
    print(title)
    print("-" * 100)
    print(f"N unique conditions: {len(df)}")
    print(f"Mean capacity: {y.mean():.6f}")
    print(f"Median capacity: {y.median():.6f}")
    print(f"Std capacity: {y.std(ddof=1):.6f}")
    print(f"Min capacity: {y.min():.6f}")
    print(f"Max capacity: {y.max():.6f}")
    print(f"Negative capacity conditions: {(y < 0).sum()}")
    print(f"Zero capacity conditions: {(y == 0).sum()}")


def print_counts(df, column, title):
    print()
    print(title)
    print("-" * 100)

    print(
        df[column]
        .value_counts(dropna=False)
        .to_string()
    )


def analyze_pair(
    conditions,
    source_name,
    target_name,
):
    print()
    print("=" * 100)
    print(
        f"CANDIDATE MATRIX SHIFT: "
        f"{source_name} -> {target_name}"
    )
    print("=" * 100)

    source = conditions[
        conditions[DOMAIN] == source_name
    ].copy()

    target = conditions[
        conditions[DOMAIN] == target_name
    ].copy()

    print(
        f"Source unique conditions: {len(source)}"
    )

    print(
        f"Target unique conditions: {len(target)}"
    )

    describe_target(
        source,
        f"{source_name.upper()} TARGET DISTRIBUTION",
    )

    describe_target(
        target,
        f"{target_name.upper()} TARGET DISTRIBUTION",
    )

    excluded = {
        DOMAIN,
        TARGET,
        FINAL_CONCENTRATION,
        "capacity_mean",
        "capacity_std",
        "capacity_min",
        "capacity_max",
        "capacity_spread",
        "n_replicates",
        "_condition_signature",
    }

    matching_columns = [
        col
        for col in conditions.columns
        if col not in excluded
        and not col.startswith("_")
    ]

    source["_cross_domain_signature"] = make_signature(
        source,
        matching_columns,
    )

    target["_cross_domain_signature"] = make_signature(
        target,
        matching_columns,
    )

    source_signatures = set(
        source["_cross_domain_signature"]
    )

    target_signatures = set(
        target["_cross_domain_signature"]
    )

    shared = (
        source_signatures
        & target_signatures
    )

    source_overlap = source[
        source["_cross_domain_signature"].isin(shared)
    ].copy()

    target_overlap = target[
        target["_cross_domain_signature"].isin(shared)
    ].copy()

    print()
    print("EXACT SUPPORT OVERLAP AFTER REPLICATE COLLAPSE")
    print("-" * 100)

    print(
        f"Source unique signatures: "
        f"{len(source_signatures)}"
    )

    print(
        f"Target unique signatures: "
        f"{len(target_signatures)}"
    )

    print(
        f"Shared exact signatures: "
        f"{len(shared)}"
    )

    print(
        f"Source conditions in exact overlap: "
        f"{len(source_overlap)}"
    )

    print(
        f"Target conditions in exact overlap: "
        f"{len(target_overlap)}"
    )

    print(
        f"Source overlap coverage: "
        f"{100.0 * len(source_overlap) / len(source):.2f}%"
        if len(source)
        else "Source overlap coverage: N/A"
    )

    print(
        f"Target overlap coverage: "
        f"{100.0 * len(target_overlap) / len(target):.2f}%"
        if len(target)
        else "Target overlap coverage: N/A"
    )

    print_counts(
        source_overlap,
        POLLUTANT,
        "SOURCE POLLUTANTS INSIDE EXACT OVERLAP",
    )

    print_counts(
        target_overlap,
        POLLUTANT,
        "TARGET POLLUTANTS INSIDE EXACT OVERLAP",
    )

    print_counts(
        source_overlap,
        ADSORBENT,
        "SOURCE ADSORBENTS INSIDE EXACT OVERLAP",
    )

    print_counts(
        target_overlap,
        ADSORBENT,
        "TARGET ADSORBENTS INSIDE EXACT OVERLAP",
    )

    print_counts(
        source_overlap,
        ADSORPTION_TYPE,
        "SOURCE ADSORPTION TYPES INSIDE EXACT OVERLAP",
    )

    print_counts(
        target_overlap,
        ADSORPTION_TYPE,
        "TARGET ADSORPTION TYPES INSIDE EXACT OVERLAP",
    )

    source_pair = (
        source_overlap[
            [
                "_cross_domain_signature",
                TARGET,
                "n_replicates",
                "capacity_std",
                "capacity_spread",
            ]
        ]
        .rename(
            columns={
                TARGET: "source_capacity",
                "n_replicates": "source_replicates",
                "capacity_std": "source_capacity_std",
                "capacity_spread": "source_capacity_spread",
            }
        )
    )

    target_pair = (
        target_overlap[
            [
                "_cross_domain_signature",
                TARGET,
                "n_replicates",
                "capacity_std",
                "capacity_spread",
            ]
        ]
        .rename(
            columns={
                TARGET: "target_capacity",
                "n_replicates": "target_replicates",
                "capacity_std": "target_capacity_std",
                "capacity_spread": "target_capacity_spread",
            }
        )
    )

    paired = source_pair.merge(
        target_pair,
        on="_cross_domain_signature",
        how="inner",
        validate="one_to_one",
    )

    paired["capacity_shift"] = (
        paired["target_capacity"]
        - paired["source_capacity"]
    )

    paired["absolute_capacity_shift"] = (
        paired["capacity_shift"].abs()
    )

    print()
    print("PAIRED MATRIX EFFECT")
    print("-" * 100)

    print(
        f"Matched condition pairs: "
        f"{len(paired)}"
    )

    if len(paired):
        print(
            f"Mean source capacity: "
            f"{paired['source_capacity'].mean():.6f}"
        )

        print(
            f"Mean target capacity: "
            f"{paired['target_capacity'].mean():.6f}"
        )

        print(
            f"Mean target-source capacity shift: "
            f"{paired['capacity_shift'].mean():.6f}"
        )

        print(
            f"Median target-source capacity shift: "
            f"{paired['capacity_shift'].median():.6f}"
        )

        print(
            f"Median absolute matrix effect: "
            f"{paired['absolute_capacity_shift'].median():.6f}"
        )

        print(
            f"90th percentile absolute matrix effect: "
            f"{paired['absolute_capacity_shift'].quantile(0.90):.6f}"
        )

        if (
            paired["source_capacity"].nunique() > 1
            and
            paired["target_capacity"].nunique() > 1
        ):
            correlation = paired[
                [
                    "source_capacity",
                    "target_capacity",
                ]
            ].corr().iloc[0, 1]

            print(
                f"Source-target paired capacity correlation: "
                f"{correlation:.6f}"
            )

    source_noise = source_overlap[
        source_overlap["n_replicates"] > 1
    ]

    target_noise = target_overlap[
        target_overlap["n_replicates"] > 1
    ]

    print()
    print("REPLICATE NOISE INSIDE EXACT SUPPORT")
    print("-" * 100)

    print(
        f"Source repeated conditions: "
        f"{len(source_noise)}"
    )

    print(
        f"Target repeated conditions: "
        f"{len(target_noise)}"
    )

    if len(source_noise):
        print(
            f"Source median within-condition capacity spread: "
            f"{source_noise['capacity_spread'].median():.6f}"
        )

    if len(target_noise):
        print(
            f"Target median within-condition capacity spread: "
            f"{target_noise['capacity_spread'].median():.6f}"
        )

    slug = (
        source_name.lower().replace(" ", "_")
        + "_to_"
        + target_name.lower().replace(" ", "_")
    )

    paired.to_csv(
        OUTPUT_DIR / f"{slug}_exact_pairs.csv",
        index=False,
    )

    source_overlap.to_csv(
        OUTPUT_DIR / f"{slug}_source_overlap.csv",
        index=False,
    )

    target_overlap.to_csv(
        OUTPUT_DIR / f"{slug}_target_overlap.csv",
        index=False,
    )

    return {
        "source": source_name,
        "target": target_name,
        "source_conditions": len(source),
        "target_conditions": len(target),
        "shared_signatures": len(shared),
        "source_overlap_conditions": len(source_overlap),
        "target_overlap_conditions": len(target_overlap),
        "source_overlap_fraction": (
            len(source_overlap) / len(source)
            if len(source)
            else np.nan
        ),
        "target_overlap_fraction": (
            len(target_overlap) / len(target)
            if len(target)
            else np.nan
        ),
        "paired_mean_abs_shift": (
            paired["absolute_capacity_shift"].mean()
            if len(paired)
            else np.nan
        ),
        "paired_median_abs_shift": (
            paired["absolute_capacity_shift"].median()
            if len(paired)
            else np.nan
        ),
    }


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST BIOCHAR DOMAIN-PAIR AUDIT")
    print("=" * 100)

    raw = pd.read_csv(DATA_PATH)

    print(f"Raw rows: {len(raw)}")

    exact_duplicates = int(
        raw.duplicated().sum()
    )

    print(
        f"Exact duplicate rows: "
        f"{exact_duplicates}"
    )

    clean = (
        raw.drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Rows after exact deduplication: "
        f"{len(clean)}"
    )

    conditions = collapse_conditions(
        clean
    )

    print()
    print("=" * 100)
    print("CONDITION-LEVEL DATASET")
    print("=" * 100)

    print(
        f"Unique experimental conditions: "
        f"{len(conditions)}"
    )

    print(
        f"Conditions with >1 replicate: "
        f"{(conditions['n_replicates'] > 1).sum()}"
    )

    print(
        f"Conditions with nonzero capacity spread: "
        f"{(conditions['capacity_spread'] > 0).sum()}"
    )

    print(
        f"Median replicate count: "
        f"{conditions['n_replicates'].median():.2f}"
    )

    repeated = conditions[
        conditions["n_replicates"] > 1
    ]

    if len(repeated):
        print(
            f"Median within-condition capacity spread: "
            f"{repeated['capacity_spread'].median():.6f}"
        )

        print(
            f"90th percentile within-condition spread: "
            f"{repeated['capacity_spread'].quantile(0.90):.6f}"
        )

        print(
            f"Maximum within-condition spread: "
            f"{repeated['capacity_spread'].max():.6f}"
        )

    print()
    print("UNIQUE CONDITIONS BY WASTEWATER TYPE")
    print("-" * 100)

    print(
        conditions[DOMAIN]
        .value_counts()
        .to_string()
    )

    conditions.to_csv(
        OUTPUT_DIR
        / "biochar_ec_unique_conditions.csv",
        index=False,
    )

    summaries = []

    for source_name, target_name in CANDIDATE_PAIRS:
        result = analyze_pair(
            conditions,
            source_name,
            target_name,
        )

        summaries.append(result)

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 100)
    print("CANDIDATE PAIR SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        OUTPUT_DIR
        / "biochar_ec_candidate_pair_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("NEXT DECISION")
    print("=" * 100)

    print(
        "Do not train on raw rows."
    )

    print(
        "Use unique experimental conditions so repeated measurements "
        "cannot leak across model partitions."
    )

    print(
        "The primary matrix-shift benchmark will be chosen from "
        "the two audited pairs using condition-level sample size, "
        "exact support coverage, pollutant/material overlap, "
        "replicate noise, and scientific interpretability."
    )

    print(
        "The prediction target should initially remain continuous "
        "adsorption capacity rather than introducing an arbitrary "
        "classification threshold."
    )


if __name__ == "__main__":
    main()
