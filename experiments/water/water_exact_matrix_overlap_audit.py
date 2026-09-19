import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "water"

INPUT = RESULTS / "dye_removal_endpoints_collapsed.csv"
OUTPUT = RESULTS / "exact_matrix_overlap_audit.csv"


def norm(x):
    if pd.isna(x):
        return "__NA__"
    return re.sub(r"\s+", " ", str(x).strip().lower())


def canon(series):
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        return x.map(
            lambda v: "__NA__"
            if pd.isna(v)
            else format(float(v), ".12g")
        )

    return series.map(norm)


def matrix_stressed(row):
    ha = pd.to_numeric(
        pd.Series([row["ha_mg/l"]]),
        errors="coerce"
    ).iloc[0]

    ha_flag = bool(pd.notna(ha) and ha > 0)

    anion = norm(row["anions"])

    anion_flag = anion not in {
        "__na__",
        "",
        "none",
        "nan",
        "0",
        "control",
        "no anion",
        "no anions",
    }

    return ha_flag or anion_flag


def build_signature(df, columns):
    tmp = pd.DataFrame(index=df.index)

    for col in columns:
        tmp[col] = canon(df[col])

    return pd.util.hash_pandas_object(
        tmp,
        index=False,
    ).astype("uint64")


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST EXACT WATER-MATRIX OVERLAP AUDIT")
    print("=" * 100)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing endpoint dataset: {INPUT}"
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

    df["_dye"] = df["dye"].map(norm)
    df["_catalyst"] = df["catalyst"].map(norm)
    df["_matrix_stressed"] = df.apply(
        matrix_stressed,
        axis=1,
    )

    matched = df[
        (df["_dye"] == "melachite green")
        &
        (df["_catalyst"] == "2 wt% pd-bfo")
    ].copy()

    print()
    print(f"Matched Melachite Green + 2 wt% Pd-BFO: {len(matched)}")
    print(
        f"Controlled: "
        f"{(~matched['_matrix_stressed']).sum()}"
    )
    print(
        f"Matrix-stressed: "
        f"{matched['_matrix_stressed'].sum()}"
    )

    exclude = {
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
        "_dye",
        "_catalyst",
    }

    feature_columns = [
        c for c in matched.columns
        if c not in exclude
        and not c.startswith("_")
    ]

    print()
    print("NON-MATRIX VARIABLES HELD FIXED IN EXACT MATCH")
    print("-" * 100)

    for col in feature_columns:
        print(col)

    matched["_signature"] = build_signature(
        matched,
        feature_columns,
    )

    control = matched[
        ~matched["_matrix_stressed"]
    ].copy()

    stressed = matched[
        matched["_matrix_stressed"]
    ].copy()

    control_signatures = set(control["_signature"])
    stressed_signatures = set(stressed["_signature"])

    shared = control_signatures & stressed_signatures

    exact_control = control[
        control["_signature"].isin(shared)
    ].copy()

    exact_stressed = stressed[
        stressed["_signature"].isin(shared)
    ].copy()

    print()
    print("=" * 100)
    print("EXACT NON-MATRIX COVARIATE OVERLAP")
    print("=" * 100)

    print(f"Controlled unique signatures: {len(control_signatures)}")
    print(f"Stressed unique signatures: {len(stressed_signatures)}")
    print(f"Shared signatures: {len(shared)}")

    print(
        f"Controlled samples inside exact overlap: "
        f"{len(exact_control)}"
    )

    print(
        f"Stressed samples inside exact overlap: "
        f"{len(exact_stressed)}"
    )

    if not shared:
        print()
        print(
            "RESULT: There are NO controlled and matrix-stressed "
            "experiments with all other recorded experimental "
            "conditions held exactly constant."
        )

        print()
        print(
            "Therefore the 16 -> 9 split remains observationally "
            "confounded and should NOT be treated as a clean "
            "water-matrix domain-adaptation benchmark."
        )

        return

    rows = []

    print()
    print("=" * 100)
    print("SHARED EXPERIMENTAL STRATA")
    print("=" * 100)

    for i, sig in enumerate(sorted(shared), 1):
        c = control[
            control["_signature"] == sig
        ]

        s = stressed[
            stressed["_signature"] == sig
        ]

        base = c.iloc[0]

        c_eff = pd.to_numeric(
            c["efficiency_%"],
            errors="coerce",
        )

        s_eff = pd.to_numeric(
            s["efficiency_%"],
            errors="coerce",
        )

        row = {
            "stratum": i,
            "control_n": len(c),
            "stressed_n": len(s),
            "control_eff_mean": c_eff.mean(),
            "stressed_eff_mean": s_eff.mean(),
            "eff_difference": (
                s_eff.mean() - c_eff.mean()
            ),
        }

        for col in [
            "solution_ph",
            "dye_conc_mg/l",
            "light_intensity_watt",
            "light_source_dist_cm",
            "loading_g",
            "volume_l",
        ]:
            if col in matched.columns:
                row[col] = base[col]

        rows.append(row)

    report = pd.DataFrame(rows)

    print(report.to_string(index=False))

    report.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 100)
    print("OVERLAP EFFICIENCY SUMMARY")
    print("=" * 100)

    print(
        f"Controlled mean efficiency: "
        f"{pd.to_numeric(exact_control['efficiency_%'], errors='coerce').mean():.4f}"
    )

    print(
        f"Stressed mean efficiency: "
        f"{pd.to_numeric(exact_stressed['efficiency_%'], errors='coerce').mean():.4f}"
    )

    print()
    print(f"Saved: {OUTPUT}")

    print()
    print(
        "This audit does not train a model. "
        "It tests whether water-matrix stress is identifiable "
        "independently of the other recorded process variables."
    )


if __name__ == "__main__":
    main()
