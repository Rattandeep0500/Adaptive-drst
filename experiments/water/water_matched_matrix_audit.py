import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "water"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TRAJECTORIES = 99


def norm(x):
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def canonicalize(series):
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        return x.map(
            lambda v: "__NA__" if pd.isna(v) else format(float(v), ".12g")
        )

    return (
        series.astype("string")
        .fillna("__NA__")
        .str.strip()
        .str.lower()
    )


def load_data():
    try:
        from aqua_fetch import dye_removal
        data = dye_removal()
    except ImportError:
        from water_datasets import dye_removal
        data = dye_removal()

    if isinstance(data, tuple):
        data = data[0]

    return data.copy()


def make_trajectory_ids(df, columns):
    canonical = pd.DataFrame(index=df.index)

    for col in columns:
        canonical[col] = canonicalize(df[col])

    hashes = pd.util.hash_pandas_object(
        canonical,
        index=False,
    ).astype("uint64")

    codes, _ = pd.factorize(hashes, sort=False)
    return codes.astype(np.int64)


def matrix_stressed(row):
    ha = pd.to_numeric(
        pd.Series([row["ha_mg/l"]]),
        errors="coerce",
    ).iloc[0]

    ha_stress = bool(pd.notna(ha) and ha > 0)

    anion = norm(row["anions"])
    anion_stress = anion not in {
        "",
        "na",
        "nan",
        "none",
        "0",
        "control",
        "noanion",
        "noanions",
    }

    return ha_stress or anion_stress


def smd(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(b, errors="coerce").dropna().astype(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)

    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0 if np.isclose(a.mean(), b.mean()) else np.nan

    return float((b.mean() - a.mean()) / pooled)


def print_efficiency_balance(df, title):
    y = pd.to_numeric(df["efficiency_%"], errors="coerce").dropna()

    print()
    print(title)
    print("-" * len(title))
    print(f"N: {len(y)}")
    print(f"Mean: {y.mean():.4f}")
    print(f"Median: {y.median():.4f}")
    print(f"Min: {y.min():.4f}")
    print(f"Max: {y.max():.4f}")

    for threshold in [50, 60, 70, 80, 90]:
        low = int((y < threshold).sum())
        high = int((y >= threshold).sum())
        print(
            f"{threshold}% threshold: "
            f"< {threshold} = {low}, >= {threshold} = {high}"
        )


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST WATER MATCHED-DOMAIN AUDIT")
    print("=" * 100)

    raw = load_data()

    print(f"Raw rows: {len(raw)}")
    print(f"Raw columns: {len(raw.columns)}")

    duplicate_count = int(raw.duplicated().sum())

    df = raw.drop_duplicates().reset_index(drop=True)

    print(f"Exact duplicates removed: {duplicate_count}")
    print(f"Rows after deduplication: {len(df)}")

    required = [
        "catalyst",
        "time_m",
        "dye",
        "ha_mg/l",
        "anions",
        "final_concentration_mg/l",
        "k_1st",
        "k_2nd",
        "efficiency_%",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    excluded_from_trajectory = {
        "time_m",
        "final_concentration_mg/l",
        "k_1st",
        "k_2nd",
        "efficiency_%",
    }

    trajectory_columns = [
        c for c in df.columns
        if c not in excluded_from_trajectory
    ]

    df["_trajectory_id"] = make_trajectory_ids(
        df,
        trajectory_columns,
    )

    n_trajectories = int(df["_trajectory_id"].nunique())

    print()
    print(f"UNIQUE TRAJECTORIES = {n_trajectories}")

    if n_trajectories != EXPECTED_TRAJECTORIES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAJECTORIES} trajectories, "
            f"found {n_trajectories}"
        )

    df["_time_numeric"] = pd.to_numeric(
        df["time_m"],
        errors="coerce",
    )

    if df["_time_numeric"].isna().any():
        raise RuntimeError("time_m contains invalid values")

    maximum_time = (
        df.groupby("_trajectory_id")["_time_numeric"]
        .transform("max")
    )

    endpoint_rows = df[
        df["_time_numeric"].eq(maximum_time)
    ].copy()

    tie_records = []

    for trajectory_id, group in endpoint_rows.groupby("_trajectory_id"):
        efficiency = pd.to_numeric(
            group["efficiency_%"],
            errors="coerce",
        ).dropna()

        final_concentration = pd.to_numeric(
            group["final_concentration_mg/l"],
            errors="coerce",
        ).dropna()

        k1 = pd.to_numeric(
            group["k_1st"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        k2 = pd.to_numeric(
            group["k_2nd"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        tie_records.append({
            "trajectory_id": int(trajectory_id),
            "n_endpoint_rows": int(len(group)),
            "efficiency_unique": int(efficiency.round(10).nunique()),
            "efficiency_min": float(efficiency.min()) if len(efficiency) else np.nan,
            "efficiency_max": float(efficiency.max()) if len(efficiency) else np.nan,
            "efficiency_spread": (
                float(efficiency.max() - efficiency.min())
                if len(efficiency) else np.nan
            ),
            "final_concentration_unique": int(
                final_concentration.round(10).nunique()
            ),
            "final_concentration_spread": (
                float(
                    final_concentration.max()
                    - final_concentration.min()
                )
                if len(final_concentration) else np.nan
            ),
            "k1_unique": int(k1.round(12).nunique()),
            "k2_unique": int(k2.round(12).nunique()),
        })

    ties = pd.DataFrame(tie_records)

    multi_endpoint = ties[
        ties["n_endpoint_rows"] > 1
    ]

    conflicting_efficiency = multi_endpoint[
        multi_endpoint["efficiency_unique"] > 1
    ]

    conflicting_final = multi_endpoint[
        multi_endpoint["final_concentration_unique"] > 1
    ]

    print()
    print("=" * 100)
    print("ENDPOINT-TIE AUDIT")
    print("=" * 100)

    print(
        f"Trajectories with >1 row at maximum time: "
        f"{len(multi_endpoint)}"
    )

    print(
        f"Trajectories with conflicting endpoint efficiency: "
        f"{len(conflicting_efficiency)}"
    )

    print(
        f"Trajectories with conflicting final concentration: "
        f"{len(conflicting_final)}"
    )

    if len(conflicting_efficiency):
        print(
            f"Largest efficiency spread: "
            f"{conflicting_efficiency['efficiency_spread'].max():.8f}"
        )
    else:
        print("Largest efficiency spread: 0")

    if len(conflicting_final):
        print(
            f"Largest final-concentration spread: "
            f"{conflicting_final['final_concentration_spread'].max():.8f}"
        )
    else:
        print("Largest final-concentration spread: 0")

    ties.to_csv(
        OUTPUT_DIR / "dye_removal_endpoint_tie_audit.csv",
        index=False,
    )

    collapsed_rows = []

    for trajectory_id, group in endpoint_rows.groupby(
        "_trajectory_id",
        sort=True,
    ):
        row = group.iloc[0].copy()

        for col in [
            "final_concentration_mg/l",
            "k_1st",
            "k_2nd",
            "efficiency_%",
        ]:
            values = pd.to_numeric(
                group[col],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

            if values.notna().any():
                row[col] = float(values.median())

        row["_endpoint_rows"] = len(group)

        collapsed_rows.append(row)

    endpoints = pd.DataFrame(collapsed_rows)
    endpoints = endpoints.reset_index(drop=True)

    print()
    print(f"COLLAPSED ENDPOINT RECORDS = {len(endpoints)}")

    if len(endpoints) != EXPECTED_TRAJECTORIES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAJECTORIES} collapsed endpoints, "
            f"got {len(endpoints)}"
        )

    endpoints["_dye_norm"] = endpoints["dye"].map(norm)
    endpoints["_catalyst_norm"] = endpoints["catalyst"].map(norm)
    endpoints["_matrix_stressed"] = endpoints.apply(
        matrix_stressed,
        axis=1,
    )

    endpoints["_matrix_domain"] = np.where(
        endpoints["_matrix_stressed"],
        "matrix-stressed",
        "controlled",
    )

    print()
    print("=" * 100)
    print("DYE SPELLING CHECK")
    print("=" * 100)

    print(endpoints["dye"].value_counts().to_string())

    indigo = endpoints[
        endpoints["_dye_norm"] == "indigo"
    ].copy()

    melachite = endpoints[
        endpoints["_dye_norm"] == "melachitegreen"
    ].copy()

    print()
    print(f"Indigo experiments: {len(indigo)}")
    print(f"Melachite Green experiments: {len(melachite)}")

    print()
    print("=" * 100)
    print("DYE × CATALYST CONFOUNDING")
    print("=" * 100)

    dye_catalyst = pd.crosstab(
        endpoints["dye"],
        endpoints["catalyst"],
        margins=True,
    )

    print(dye_catalyst.to_string())

    indigo_catalysts = set(indigo["catalyst"])
    melachite_catalysts = set(melachite["catalyst"])

    shared = indigo_catalysts & melachite_catalysts

    print()
    print(f"Indigo catalyst identities: {len(indigo_catalysts)}")
    print(
        f"Melachite catalyst identities: "
        f"{len(melachite_catalysts)}"
    )
    print(f"Shared catalyst identities: {len(shared)}")
    print(f"Shared catalysts: {sorted(shared)}")

    print()
    print("=" * 100)
    print("BROAD WATER-MATRIX SPLIT")
    print("=" * 100)

    controlled = endpoints[
        ~endpoints["_matrix_stressed"]
    ].copy()

    stressed = endpoints[
        endpoints["_matrix_stressed"]
    ].copy()

    print(f"Controlled: {len(controlled)}")
    print(f"Matrix-stressed: {len(stressed)}")

    print()
    print(
        pd.crosstab(
            endpoints["dye"],
            endpoints["_matrix_domain"],
            margins=True,
        ).to_string()
    )

    print()
    print("=" * 100)
    print("MATCHED WATER-MATRIX COHORT")
    print("Fixed dye = Melachite Green")
    print("Fixed catalyst = 2 wt% Pd-BFO")
    print("=" * 100)

    matched = endpoints[
        (endpoints["_dye_norm"] == "melachitegreen")
        &
        (endpoints["_catalyst_norm"] == "2wtpdbfo")
    ].copy()

    matched_control = matched[
        ~matched["_matrix_stressed"]
    ].copy()

    matched_stressed = matched[
        matched["_matrix_stressed"]
    ].copy()

    print(f"Matched total experiments: {len(matched)}")
    print(f"Matched controlled experiments: {len(matched_control)}")
    print(f"Matched matrix-stressed experiments: {len(matched_stressed)}")

    print()
    print("HA distribution")
    print("-" * 50)
    print(
        matched["ha_mg/l"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print()
    print("Anion distribution")
    print("-" * 50)
    print(
        matched["anions"]
        .fillna("N/A")
        .value_counts(dropna=False)
        .to_string()
    )

    print_efficiency_balance(
        matched_control,
        "MATCHED CONTROLLED EFFICIENCY",
    )

    print_efficiency_balance(
        matched_stressed,
        "MATCHED MATRIX-STRESSED EFFICIENCY",
    )

    excluded_shift_features = {
        "_trajectory_id",
        "_time_numeric",
        "_endpoint_rows",
        "_matrix_stressed",
        "time_m",
        "ha_mg/l",
        "anions",
        "efficiency_%",
        "final_concentration_mg/l",
        "k_1st",
        "k_2nd",
    }

    shift_rows = []

    for feature in matched.columns:
        if feature in excluded_shift_features:
            continue

        if str(feature).startswith("_"):
            continue

        if not pd.api.types.is_numeric_dtype(matched[feature]):
            continue

        a = pd.to_numeric(
            matched_control[feature],
            errors="coerce",
        )

        b = pd.to_numeric(
            matched_stressed[feature],
            errors="coerce",
        )

        if a.notna().sum() < 2 or b.notna().sum() < 2:
            continue

        value = smd(a, b)

        if not np.isfinite(value):
            continue

        shift_rows.append({
            "feature": feature,
            "controlled_mean": float(a.mean()),
            "stressed_mean": float(b.mean()),
            "smd_stressed_minus_controlled": value,
            "abs_smd": abs(value),
        })

    shift = pd.DataFrame(shift_rows)

    if not shift.empty:
        shift = shift.sort_values(
            "abs_smd",
            ascending=False,
        ).reset_index(drop=True)

        print()
        print("=" * 100)
        print("RESIDUAL COVARIATE SHIFT WITH DYE AND CATALYST HELD FIXED")
        print("=" * 100)

        print(
            shift[
                [
                    "feature",
                    "controlled_mean",
                    "stressed_mean",
                    "smd_stressed_minus_controlled",
                ]
            ].to_string(index=False)
        )

        shift.to_csv(
            OUTPUT_DIR / "matched_matrix_residual_shift.csv",
            index=False,
        )

    matched_export = matched.drop(
        columns=[
            "_dye_norm",
            "_catalyst_norm",
            "_time_numeric",
        ],
        errors="ignore",
    )

    matched_export.to_csv(
        OUTPUT_DIR / "matched_matrix_endpoints.csv",
        index=False,
    )

    endpoints.to_csv(
        OUTPUT_DIR / "dye_removal_endpoints_collapsed.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("SCIENTIFIC AUDIT SUMMARY")
    print("=" * 100)

    if len(shared) == 0:
        print(
            "DYE SHIFT WARNING: Indigo and Melachite Green have "
            "zero catalyst identity overlap."
        )
        print(
            "The raw Indigo -> Melachite Green split is therefore "
            "strongly catalyst-confounded."
        )
    else:
        print(
            f"DYE SHIFT: {len(shared)} shared catalyst identities "
            "require further inspection."
        )

    print()
    print(
        "The broad controlled -> matrix-stressed split should not "
        "be used directly because the stressed set is concentrated "
        "in Melachite Green / 2 wt% Pd-BFO."
    )

    print()
    print(
        "The matched Melachite Green + 2 wt% Pd-BFO subset is the "
        "cleaner water-matrix candidate because dye and catalyst "
        "identity are held fixed."
    )

    print()
    print(
        "No ML model has been trained. This remains an "
        "experimental-design audit."
    )


if __name__ == "__main__":
    main()
