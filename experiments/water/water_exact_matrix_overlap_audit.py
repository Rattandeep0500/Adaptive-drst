import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "water"

INPUT = RESULTS / "dye_removal_endpoints_collapsed.csv"
OUTPUT = RESULTS / "exact_matrix_overlap_audit.csv"


def text_norm(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


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


def no_anion(value):
    if pd.isna(value):
        return True

    value = text_norm(value)

    return value in {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "0",
        "control",
        "no anion",
        "no anions",
    }


def no_ha(value):
    if pd.isna(value):
        return True

    try:
        return float(value) <= 0.0
    except Exception:
        value = text_norm(value)
        return value in {
            "",
            "na",
            "n/a",
            "nan",
            "none",
            "0",
            "control",
        }


def is_matrix_stressed(row):
    return not (
        no_ha(row["ha_mg/l"])
        and no_anion(row["anions"])
    )


def make_signature(df, columns):
    canonical = pd.DataFrame(index=df.index)

    for col in columns:
        canonical[col] = canonicalize(df[col])

    return pd.util.hash_pandas_object(
        canonical,
        index=False,
    ).astype("uint64")


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST EXACT WATER-MATRIX OVERLAP AUDIT - CORRECTED")
    print("=" * 100)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "catalyst",
        "dye",
        "ha_mg/l",
        "anions",
        "efficiency_%",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    df["_dye_norm"] = df["dye"].map(text_norm)
    df["_catalyst_norm"] = df["catalyst"].map(text_norm)

    df["_matrix_stressed_recomputed"] = df.apply(
        is_matrix_stressed,
        axis=1,
    )

    print()
    print("FULL ENDPOINT MATRIX DOMAIN")
    print("-" * 100)
    print(
        df["_matrix_stressed_recomputed"]
        .map({
            False: "controlled",
            True: "matrix-stressed",
        })
        .value_counts()
        .to_string()
    )

    matched = df[
        (df["_dye_norm"] == "melachite green")
        &
        (df["_catalyst_norm"] == "2 wt% pd-bfo")
    ].copy()

    controlled = matched[
        ~matched["_matrix_stressed_recomputed"]
    ].copy()

    stressed = matched[
        matched["_matrix_stressed_recomputed"]
    ].copy()

    print()
    print("=" * 100)
    print("MATCHED COHORT CHECK")
    print("=" * 100)
    print(f"Matched total: {len(matched)}")
    print(f"Controlled: {len(controlled)}")
    print(f"Matrix-stressed: {len(stressed)}")

    if len(matched) != 25:
        raise RuntimeError(
            f"Expected matched cohort N=25, found {len(matched)}"
        )

    if len(controlled) != 16:
        raise RuntimeError(
            f"Expected 16 controlled experiments, found {len(controlled)}"
        )

    if len(stressed) != 9:
        raise RuntimeError(
            f"Expected 9 matrix-stressed experiments, found {len(stressed)}"
        )

    print()
    print("CONTROLLED HA / ANION CONDITIONS")
    print("-" * 100)
    print(
        controlled[
            ["ha_mg/l", "anions"]
        ].value_counts(dropna=False).to_string()
    )

    print()
    print("MATRIX-STRESSED HA / ANION CONDITIONS")
    print("-" * 100)
    print(
        stressed[
            ["ha_mg/l", "anions"]
        ].value_counts(dropna=False).to_string()
    )

    excluded = {
        "ha_mg/l",
        "anions",
        "time_m",
        "efficiency_%",
        "final_concentration_mg/l",
        "k_1st",
        "k_2nd",
        "_trajectory_id",
        "_endpoint_rows",
        "_matrix_stressed",
        "_matrix_domain",
        "_matrix_stressed_recomputed",
        "_dye_norm",
        "_catalyst_norm",
    }

    signature_columns = [
        col
        for col in matched.columns
        if col not in excluded
        and not col.startswith("_")
    ]

    print()
    print("=" * 100)
    print("NON-MATRIX VARIABLES REQUIRED FOR EXACT OVERLAP")
    print("=" * 100)

    for col in signature_columns:
        print(col)

    matched["_exact_signature"] = make_signature(
        matched,
        signature_columns,
    )

    controlled = matched[
        ~matched["_matrix_stressed_recomputed"]
    ].copy()

    stressed = matched[
        matched["_matrix_stressed_recomputed"]
    ].copy()

    control_signatures = set(
        controlled["_exact_signature"].tolist()
    )

    stressed_signatures = set(
        stressed["_exact_signature"].tolist()
    )

    shared = control_signatures & stressed_signatures

    exact_controlled = controlled[
        controlled["_exact_signature"].isin(shared)
    ].copy()

    exact_stressed = stressed[
        stressed["_exact_signature"].isin(shared)
    ].copy()

    print()
    print("=" * 100)
    print("EXACT NON-MATRIX COVARIATE OVERLAP")
    print("=" * 100)

    print(
        f"Controlled unique signatures: "
        f"{len(control_signatures)}"
    )

    print(
        f"Stressed unique signatures: "
        f"{len(stressed_signatures)}"
    )

    print(f"Shared signatures: {len(shared)}")

    print(
        f"Controlled samples inside exact overlap: "
        f"{len(exact_controlled)}"
    )

    print(
        f"Stressed samples inside exact overlap: "
        f"{len(exact_stressed)}"
    )

    key_process_columns = [
        col
        for col in [
            "solution_ph",
            "dye_conc_mg/l",
            "light_intensity_watt",
            "light_source_dist_cm",
            "loading_g",
            "volume_l",
        ]
        if col in matched.columns
    ]

    print()
    print("=" * 100)
    print("KEY PROCESS VARIABLE SUPPORT")
    print("=" * 100)

    rows = []

    for feature in key_process_columns:
        control_values = sorted(
            controlled[feature]
            .dropna()
            .unique()
            .tolist()
        )

        stressed_values = sorted(
            stressed[feature]
            .dropna()
            .unique()
            .tolist()
        )

        common_values = sorted(
            set(control_values)
            & set(stressed_values)
        )

        rows.append({
            "feature": feature,
            "controlled_values": str(control_values),
            "stressed_values": str(stressed_values),
            "shared_values": str(common_values),
            "shared_value_count": len(common_values),
        })

        print()
        print(feature)
        print(f"  controlled: {control_values}")
        print(f"  stressed:   {stressed_values}")
        print(f"  shared:     {common_values}")

    support_report = pd.DataFrame(rows)

    output_rows = []

    for sig in sorted(shared):
        c = controlled[
            controlled["_exact_signature"] == sig
        ]

        s = stressed[
            stressed["_exact_signature"] == sig
        ]

        c_eff = pd.to_numeric(
            c["efficiency_%"],
            errors="coerce",
        )

        s_eff = pd.to_numeric(
            s["efficiency_%"],
            errors="coerce",
        )

        row = {
            "signature": int(sig),
            "controlled_n": len(c),
            "stressed_n": len(s),
            "controlled_efficiency_mean": c_eff.mean(),
            "stressed_efficiency_mean": s_eff.mean(),
            "efficiency_difference": (
                s_eff.mean() - c_eff.mean()
            ),
        }

        output_rows.append(row)

    overlap_report = pd.DataFrame(output_rows)

    if overlap_report.empty:
        overlap_report = pd.DataFrame(
            columns=[
                "signature",
                "controlled_n",
                "stressed_n",
                "controlled_efficiency_mean",
                "stressed_efficiency_mean",
                "efficiency_difference",
            ]
        )

    overlap_report.to_csv(
        OUTPUT,
        index=False,
    )

    support_report.to_csv(
        RESULTS / "matrix_process_support_audit.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("SCIENTIFIC INTERPRETATION")
    print("=" * 100)

    if len(shared) == 0:
        print(
            "No exact controlled/stressed pairs exist after holding "
            "all recorded non-matrix experimental variables fixed."
        )
        print()
        print(
            "Therefore AquaFetch dye_removal should NOT be used as "
            "the primary clean water-matrix domain-adaptation benchmark."
        )
        print()
        print(
            "It remains useful as an observational/confounded "
            "stress-test or secondary reliability dataset."
        )
    else:
        print(
            "Exact non-matrix overlap exists. These shared strata "
            "can be investigated as a small controlled matrix-shift subset."
        )

    print()
    print(f"Overlap report: {OUTPUT}")
    print(
        f"Process-support report: "
        f"{RESULTS / 'matrix_process_support_audit.csv'}"
    )


if __name__ == "__main__":
    main()
