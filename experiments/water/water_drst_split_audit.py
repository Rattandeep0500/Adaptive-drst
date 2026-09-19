import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EXPECTED_TRAJECTORIES = 99

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "water"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def load_dye_removal():
    errors = []

    try:
        from aqua_fetch import dye_removal
        result = dye_removal()
        if isinstance(result, tuple):
            return result[0].copy(), "aqua_fetch"
        return result.copy(), "aqua_fetch"
    except Exception as exc:
        errors.append(f"aqua_fetch: {exc}")

    try:
        from water_datasets import dye_removal
        result = dye_removal()
        if isinstance(result, tuple):
            return result[0].copy(), "water_datasets"
        return result.copy(), "water_datasets"
    except Exception as exc:
        errors.append(f"water_datasets: {exc}")

    raise RuntimeError(
        "Could not load dye_removal.\n" + "\n".join(errors)
    )


def find_column(columns, exact=None, contains=None, required=True):
    exact = exact or []
    contains = contains or []

    norm_map = {normalize_name(c): c for c in columns}

    for candidate in exact:
        key = normalize_name(candidate)
        if key in norm_map:
            return norm_map[key]

    for col in columns:
        n = normalize_name(col)
        if any(normalize_name(token) in n for token in contains):
            return col

    if required:
        raise RuntimeError(
            f"Could not identify required column. exact={exact}, contains={contains}"
        )

    return None


def canonicalize_series(series):
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.map(
            lambda x: "__NA__" if pd.isna(x) else format(float(x), ".12g")
        )

    return (
        series.astype("string")
        .fillna("__NA__")
        .str.strip()
        .str.lower()
    )


def make_trajectory_key(df, columns):
    canonical = pd.DataFrame(index=df.index)

    for col in columns:
        canonical[col] = canonicalize_series(df[col])

    hashes = pd.util.hash_pandas_object(
        canonical,
        index=False
    ).astype("uint64")

    codes, uniques = pd.factorize(hashes, sort=False)

    collision_check = canonical.copy()
    collision_check["_hash"] = hashes.values

    collision_counts = (
        collision_check.groupby("_hash", dropna=False)
        .size()
    )

    if len(uniques) != canonical.drop_duplicates().shape[0]:
        raise RuntimeError(
            "Hash collision or inconsistent canonical trajectory key detected."
        )

    return codes.astype(np.int64), canonical, collision_counts


def clean_category(value):
    if pd.isna(value):
        return "N/A"
    text = str(value).strip()
    if text == "":
        return "N/A"
    return text


def is_no_anion(value):
    if pd.isna(value):
        return True

    text = normalize_name(value)

    return text in {
        "",
        "na",
        "nan",
        "none",
        "noanion",
        "noanions",
        "control",
        "0",
    }


def is_no_ha(value):
    if pd.isna(value):
        return True

    try:
        return float(value) <= 0.0
    except Exception:
        text = normalize_name(value)
        return text in {
            "",
            "na",
            "nan",
            "none",
            "control",
            "0",
        }


def standardized_mean_difference(source, target):
    source = pd.to_numeric(source, errors="coerce").dropna().astype(float)
    target = pd.to_numeric(target, errors="coerce").dropna().astype(float)

    if len(source) < 2 or len(target) < 2:
        return np.nan

    mean_s = source.mean()
    mean_t = target.mean()

    var_s = source.var(ddof=1)
    var_t = target.var(ddof=1)

    pooled = np.sqrt((var_s + var_t) / 2.0)

    if not np.isfinite(pooled) or pooled <= 0:
        return 0.0 if np.isclose(mean_s, mean_t) else np.nan

    return float((mean_t - mean_s) / pooled)


def safe_numeric_summary(source, target, feature):
    s = pd.to_numeric(source[feature], errors="coerce")
    t = pd.to_numeric(target[feature], errors="coerce")

    valid_s = s.dropna()
    valid_t = t.dropna()

    if len(valid_s) == 0 or len(valid_t) == 0:
        return None

    return {
        "feature": feature,
        "source_n": int(len(valid_s)),
        "target_n": int(len(valid_t)),
        "source_mean": float(valid_s.mean()),
        "target_mean": float(valid_t.mean()),
        "source_median": float(valid_s.median()),
        "target_median": float(valid_t.median()),
        "smd_target_minus_source": standardized_mean_difference(valid_s, valid_t),
    }


def print_value_counts(df, column, title):
    print()
    print(title)
    print("-" * len(title))

    counts = df[column].map(clean_category).value_counts(dropna=False)

    for value, count in counts.items():
        pct = 100.0 * count / len(df)
        print(f"{value}: {count} ({pct:.2f}%)")


def catalyst_overlap(source, target, catalyst_col, title):
    source_catalysts = {
        clean_category(x)
        for x in source[catalyst_col]
        if clean_category(x) != "N/A"
    }

    target_catalysts = {
        clean_category(x)
        for x in target[catalyst_col]
        if clean_category(x) != "N/A"
    }

    overlap = source_catalysts & target_catalysts
    union = source_catalysts | target_catalysts

    print()
    print(title)
    print("-" * len(title))
    print(f"Source catalysts: {len(source_catalysts)}")
    print(f"Target catalysts: {len(target_catalysts)}")
    print(f"Shared catalysts: {len(overlap)}")
    print(
        f"Catalyst Jaccard overlap: "
        f"{len(overlap) / len(union):.4f}"
        if union else
        "Catalyst Jaccard overlap: N/A"
    )

    print(f"Shared: {sorted(overlap)}")
    print(f"Source-only: {sorted(source_catalysts - target_catalysts)}")
    print(f"Target-only: {sorted(target_catalysts - source_catalysts)}")

    return {
        "source_catalysts": len(source_catalysts),
        "target_catalysts": len(target_catalysts),
        "shared_catalysts": len(overlap),
        "jaccard": len(overlap) / len(union) if union else np.nan,
    }


def efficiency_balance(df, efficiency_col, label):
    values = pd.to_numeric(df[efficiency_col], errors="coerce")

    print()
    print(label)
    print("-" * len(label))
    print(f"N: {len(df)}")
    print(f"Valid efficiency: {values.notna().sum()}")

    if values.notna().sum() == 0:
        return

    print(f"Mean efficiency: {values.mean():.4f}")
    print(f"Median efficiency: {values.median():.4f}")
    print(f"Min efficiency: {values.min():.4f}")
    print(f"Max efficiency: {values.max():.4f}")

    for threshold in [50.0, 60.0, 70.0, 80.0, 90.0]:
        valid = values.dropna()
        low = int((valid < threshold).sum())
        high = int((valid >= threshold).sum())

        print(
            f"Threshold {threshold:.0f}% -> "
            f"class 0 (< threshold): {low}, "
            f"class 1 (>= threshold): {high}"
        )


def domain_shift_table(
    source,
    target,
    candidate_features,
    source_name,
    target_name,
):
    rows = []

    for feature in candidate_features:
        if feature not in source.columns:
            continue

        if not pd.api.types.is_numeric_dtype(source[feature]):
            continue

        result = safe_numeric_summary(source, target, feature)

        if result is not None:
            rows.append(result)

    shift = pd.DataFrame(rows)

    if shift.empty:
        print()
        print(
            f"No usable numeric domain-shift statistics for "
            f"{source_name} -> {target_name}"
        )
        return shift

    shift["abs_smd"] = shift["smd_target_minus_source"].abs()
    shift = shift.sort_values(
        ["abs_smd", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    print()
    print(f"NUMERIC DOMAIN SHIFT: {source_name} -> {target_name}")
    print("-" * 80)

    display_cols = [
        "feature",
        "source_n",
        "target_n",
        "source_mean",
        "target_mean",
        "smd_target_minus_source",
    ]

    print(
        shift[display_cols]
        .head(20)
        .to_string(index=False)
    )

    return shift


def crosstab_report(df, row_col, domain_col, title):
    print()
    print(title)
    print("-" * len(title))

    table = pd.crosstab(
        df[row_col].map(clean_category),
        df[domain_col],
        margins=True,
    )

    print(table.to_string())

    return table


def main():
    print("=" * 100)
    print("ADAPTIVE-DRST WATER EXPERIMENTAL-DESIGN AUDIT")
    print("=" * 100)

    raw, provider = load_dye_removal()

    print(f"Dataset provider: {provider}")
    print(f"RAW ROWS: {len(raw)}")
    print(f"RAW COLUMNS: {len(raw.columns)}")

    print()
    print("COLUMNS")
    print("-" * 100)
    for idx, col in enumerate(raw.columns):
        print(f"{idx:02d}: {col}")

    exact_duplicate_count = int(raw.duplicated().sum())
    df = raw.drop_duplicates().reset_index(drop=True)

    print()
    print(f"EXACT DUPLICATE ROWS REMOVED: {exact_duplicate_count}")
    print(f"ROWS AFTER EXACT DEDUPLICATION: {len(df)}")

    time_col = find_column(
        df.columns,
        exact=[
            "time_m",
            "time_min",
            "time",
        ],
        contains=[
            "timem",
            "timemin",
        ],
    )

    dye_col = find_column(
        df.columns,
        exact=["dye"],
        contains=["dye"],
    )

    catalyst_col = find_column(
        df.columns,
        exact=[
            "catalyst",
            "catalyst_type",
        ],
        contains=["catalyst"],
    )

    ha_col = find_column(
        df.columns,
        exact=[
            "HA_mg/L",
            "HA (mg/L)",
            "humic_acid",
            "humic_acid_mg/L",
        ],
        contains=[
            "hamgl",
            "humicacid",
        ],
        required=False,
    )

    anion_col = find_column(
        df.columns,
        exact=["anions", "anion"],
        contains=["anion"],
        required=False,
    )

    efficiency_col = find_column(
        df.columns,
        exact=[
            "efficiency",
            "Efficiency (%)",
            "removal_efficiency",
            "degradation_efficiency",
        ],
        contains=[
            "efficiency",
        ],
        required=False,
    )

    final_conc_col = find_column(
        df.columns,
        exact=[
            "final_concentration",
            "final_concentration_mg/L",
            "final_conc_mg/l",
            "final_conc",
        ],
        contains=[
            "finalconc",
            "finalconcentration",
        ],
        required=False,
    )

    initial_conc_col = find_column(
        df.columns,
        exact=[
            "dye_concentration_mg/L",
            "initial_concentration",
            "ini_conc_mg/l",
        ],
        contains=[
            "dyeconcentration",
            "initialconc",
            "iniconc",
        ],
        required=False,
    )

    normalized = {
        col: normalize_name(col)
        for col in df.columns
    }

    explicit_outcomes = {
        col
        for col in [
            efficiency_col,
            final_conc_col,
        ]
        if col is not None
    }

    outcome_patterns = [
        "efficiency",
        "finalconc",
        "finalconcentration",
        "k1st",
        "kfirst",
        "k2nd",
        "ksecond",
        "kinetic",
        "removalrate",
        "degradationrate",
    ]

    for col, n in normalized.items():
        if any(pattern in n for pattern in outcome_patterns):
            explicit_outcomes.add(col)

    trajectory_exclusions = {
        time_col,
        *explicit_outcomes,
    }

    trajectory_columns = [
        col
        for col in df.columns
        if col not in trajectory_exclusions
    ]

    print()
    print("IDENTIFIED SPECIAL COLUMNS")
    print("-" * 100)
    print(f"time: {time_col}")
    print(f"dye: {dye_col}")
    print(f"catalyst: {catalyst_col}")
    print(f"humic acid: {ha_col}")
    print(f"anions: {anion_col}")
    print(f"efficiency: {efficiency_col}")
    print(f"initial concentration: {initial_conc_col}")
    print(f"final concentration: {final_conc_col}")

    print()
    print("OUTCOME/TIME COLUMNS EXCLUDED FROM TRAJECTORY ID")
    print("-" * 100)
    for col in sorted(trajectory_exclusions):
        print(col)

    print()
    print("STATIC EXPERIMENTAL-CONDITION COLUMNS USED FOR TRAJECTORY ID")
    print("-" * 100)
    for col in trajectory_columns:
        print(col)

    trajectory_codes, canonical, _ = make_trajectory_key(
        df,
        trajectory_columns,
    )

    df["_trajectory_id"] = trajectory_codes

    unique_trajectories = int(df["_trajectory_id"].nunique())

    print()
    print("=" * 100)
    print(f"UNIQUE TRAJECTORIES = {unique_trajectories}")
    print("=" * 100)

    trajectory_sizes = (
        df.groupby("_trajectory_id", dropna=False)
        .size()
        .rename("n_timepoints")
    )

    print()
    print("TRAJECTORY SIZE DISTRIBUTION")
    print("-" * 100)
    print(f"Min timepoints: {trajectory_sizes.min()}")
    print(f"Median timepoints: {trajectory_sizes.median():.2f}")
    print(f"Mean timepoints: {trajectory_sizes.mean():.2f}")
    print(f"Max timepoints: {trajectory_sizes.max()}")
    print(f"Single-row trajectories: {(trajectory_sizes == 1).sum()}")
    print(f"Repeated-measurement trajectories: {(trajectory_sizes > 1).sum()}")

    if unique_trajectories != EXPECTED_TRAJECTORIES:
        diagnostics_path = OUTPUT_DIR / "dye_removal_trajectory_diagnostics.csv"

        diagnostic = (
            df.groupby("_trajectory_id", dropna=False)
            .agg(
                n_rows=(time_col, "size"),
                min_time=(time_col, "min"),
                max_time=(time_col, "max"),
            )
            .reset_index()
        )

        diagnostic.to_csv(diagnostics_path, index=False)

        raise RuntimeError(
            "\nTrajectory reconstruction invariant failed.\n"
            f"Expected {EXPECTED_TRAJECTORIES} trajectories but found "
            f"{unique_trajectories}.\n"
            f"Diagnostics written to: {diagnostics_path}\n"
            "No domain-split analysis was performed."
        )

    numeric_time = pd.to_numeric(df[time_col], errors="coerce")

    if numeric_time.isna().any():
        bad = int(numeric_time.isna().sum())
        raise RuntimeError(
            f"{time_col} contains {bad} non-numeric/missing values. "
            "Cannot select experimental endpoints safely."
        )

    df["_numeric_time"] = numeric_time

    max_time = df.groupby("_trajectory_id")["_numeric_time"].transform("max")
    endpoint_candidates = df.loc[
        df["_numeric_time"].eq(max_time)
    ].copy()

    endpoint_candidate_counts = (
        endpoint_candidates.groupby("_trajectory_id")
        .size()
    )

    endpoint_ties = int((endpoint_candidate_counts > 1).sum())

    if endpoint_ties:
        print()
        print(
            f"WARNING: {endpoint_ties} trajectories have multiple rows "
            "at their maximum time."
        )
        print(
            "Exact duplicate rows were already removed. "
            "The final row in original order will be retained deterministically."
        )

    endpoints = (
        endpoint_candidates
        .sort_index()
        .groupby("_trajectory_id", sort=True, as_index=False)
        .tail(1)
        .sort_values("_trajectory_id")
        .reset_index(drop=True)
    )

    print()
    print("=" * 100)
    print(f"ENDPOINT RECORDS = {len(endpoints)}")
    print("=" * 100)

    if len(endpoints) != EXPECTED_TRAJECTORIES:
        raise RuntimeError(
            f"Endpoint invariant failed: expected {EXPECTED_TRAJECTORIES}, "
            f"got {len(endpoints)}."
        )

    endpoints["_dye_domain"] = (
        endpoints[dye_col]
        .map(clean_category)
    )

    if ha_col is not None:
        no_ha = endpoints[ha_col].map(is_no_ha)
    else:
        no_ha = pd.Series(True, index=endpoints.index)

    if anion_col is not None:
        no_anion = endpoints[anion_col].map(is_no_anion)
    else:
        no_anion = pd.Series(True, index=endpoints.index)

    endpoints["_matrix_stressed"] = ~(no_ha & no_anion)
    endpoints["_matrix_domain"] = np.where(
        endpoints["_matrix_stressed"],
        "matrix-stressed",
        "controlled",
    )

    print_value_counts(
        endpoints,
        dye_col,
        "ENDPOINT DYE DISTRIBUTION",
    )

    print_value_counts(
        endpoints,
        catalyst_col,
        "ENDPOINT CATALYST DISTRIBUTION",
    )

    if ha_col is not None:
        print_value_counts(
            endpoints,
            ha_col,
            "ENDPOINT HUMIC-ACID DISTRIBUTION",
        )

    if anion_col is not None:
        print_value_counts(
            endpoints,
            anion_col,
            "ENDPOINT ANION DISTRIBUTION",
        )

    print_value_counts(
        endpoints,
        "_matrix_domain",
        "WATER-MATRIX DOMAIN DISTRIBUTION",
    )

    print()
    print("=" * 100)
    print("CANDIDATE DOMAIN SHIFT A: INDIGO -> MALACHITE GREEN")
    print("=" * 100)

    dye_normalized = endpoints[dye_col].map(
        lambda x: normalize_name(clean_category(x))
    )

    indigo = endpoints.loc[
        dye_normalized.eq("indigo")
    ].copy()

    malachite = endpoints.loc[
        dye_normalized.str.contains("malachite", na=False)
    ].copy()

    print(f"Indigo source endpoints: {len(indigo)}")
    print(f"Malachite Green target endpoints: {len(malachite)}")

    if len(indigo) == 0 or len(malachite) == 0:
        print("Dye shift cannot be evaluated because one domain is empty.")
    else:
        catalyst_overlap(
            indigo,
            malachite,
            catalyst_col,
            "DYE-SHIFT CATALYST OVERLAP",
        )

        if ha_col is not None:
            crosstab_report(
                endpoints.loc[
                    endpoints.index.isin(indigo.index)
                    | endpoints.index.isin(malachite.index)
                ],
                dye_col,
                "_matrix_domain",
                "DYE VS WATER-MATRIX CROSS-TAB",
            )

    print()
    print("=" * 100)
    print("CANDIDATE DOMAIN SHIFT B: CONTROLLED -> MATRIX-STRESSED WATER")
    print("=" * 100)

    controlled = endpoints.loc[
        ~endpoints["_matrix_stressed"]
    ].copy()

    stressed = endpoints.loc[
        endpoints["_matrix_stressed"]
    ].copy()

    print(f"Controlled-water endpoints: {len(controlled)}")
    print(f"Matrix-stressed endpoints: {len(stressed)}")

    if len(controlled) == 0 or len(stressed) == 0:
        print(
            "Water-matrix shift cannot be evaluated because one domain is empty."
        )
    else:
        catalyst_overlap(
            controlled,
            stressed,
            catalyst_col,
            "WATER-MATRIX CATALYST OVERLAP",
        )

        crosstab_report(
            endpoints,
            catalyst_col,
            "_matrix_domain",
            "CATALYST VS WATER-MATRIX DOMAIN",
        )

        crosstab_report(
            endpoints,
            dye_col,
            "_matrix_domain",
            "DYE VS WATER-MATRIX DOMAIN",
        )

        if anion_col is not None:
            crosstab_report(
                endpoints,
                anion_col,
                "_matrix_domain",
                "ANION VS WATER-MATRIX DOMAIN",
            )

    if efficiency_col is None and initial_conc_col is not None and final_conc_col is not None:
        ci = pd.to_numeric(
            endpoints[initial_conc_col],
            errors="coerce",
        )
        cf = pd.to_numeric(
            endpoints[final_conc_col],
            errors="coerce",
        )

        endpoints["_derived_efficiency"] = (
            100.0 * (ci - cf) / ci
        )

        efficiency_col = "_derived_efficiency"

        print()
        print(
            "No explicit efficiency column was detected; "
            "efficiency was derived from initial and final concentration."
        )

    if efficiency_col is not None:
        efficiency_balance(
            endpoints,
            efficiency_col,
            "ALL ENDPOINT EFFICIENCY / CLASS-BALANCE AUDIT",
        )

        if len(indigo):
            efficiency_balance(
                indigo,
                efficiency_col,
                "INDIGO ENDPOINT EFFICIENCY",
            )

        if len(malachite):
            efficiency_balance(
                malachite,
                efficiency_col,
                "MALACHITE GREEN ENDPOINT EFFICIENCY",
            )

        if len(controlled):
            efficiency_balance(
                controlled,
                efficiency_col,
                "CONTROLLED-WATER ENDPOINT EFFICIENCY",
            )

        if len(stressed):
            efficiency_balance(
                stressed,
                efficiency_col,
                "MATRIX-STRESSED ENDPOINT EFFICIENCY",
            )

    forbidden_shift_columns = {
        "_trajectory_id",
        "_numeric_time",
        "_matrix_stressed",
        time_col,
        dye_col,
        catalyst_col,
        efficiency_col,
        final_conc_col,
    }

    numeric_candidate_features = [
        col
        for col in endpoints.columns
        if col not in forbidden_shift_columns
        and not str(col).startswith("_")
        and pd.api.types.is_numeric_dtype(endpoints[col])
    ]

    if len(indigo) and len(malachite):
        dye_shift = domain_shift_table(
            indigo,
            malachite,
            numeric_candidate_features,
            "Indigo",
            "Malachite Green",
        )
    else:
        dye_shift = pd.DataFrame()

    if len(controlled) and len(stressed):
        matrix_shift = domain_shift_table(
            controlled,
            stressed,
            numeric_candidate_features,
            "Controlled",
            "Matrix-stressed",
        )
    else:
        matrix_shift = pd.DataFrame()

    print()
    print("=" * 100)
    print("CONFOUNDING / SPLIT VALIDITY AUDIT")
    print("=" * 100)

    def exclusive_catalysts(a, b):
        ca = {
            clean_category(x)
            for x in a[catalyst_col]
            if clean_category(x) != "N/A"
        }
        cb = {
            clean_category(x)
            for x in b[catalyst_col]
            if clean_category(x) != "N/A"
        }
        return sorted(ca - cb), sorted(cb - ca)

    if len(indigo) and len(malachite):
        a_only, b_only = exclusive_catalysts(
            indigo,
            malachite,
        )

        print()
        print("Dye shift:")
        print(f"  Source N: {len(indigo)}")
        print(f"  Target N: {len(malachite)}")
        print(f"  Indigo-only catalysts: {a_only}")
        print(f"  Malachite-only catalysts: {b_only}")

        shared_dye_catalysts = (
            set(indigo[catalyst_col].map(clean_category))
            & set(malachite[catalyst_col].map(clean_category))
        )

        print(
            f"  Shared catalyst identities: "
            f"{len(shared_dye_catalysts)}"
        )

    if len(controlled) and len(stressed):
        a_only, b_only = exclusive_catalysts(
            controlled,
            stressed,
        )

        print()
        print("Water-matrix shift:")
        print(f"  Source N: {len(controlled)}")
        print(f"  Target N: {len(stressed)}")
        print(f"  Controlled-only catalysts: {a_only}")
        print(f"  Stressed-only catalysts: {b_only}")

        shared_matrix_catalysts = (
            set(controlled[catalyst_col].map(clean_category))
            & set(stressed[catalyst_col].map(clean_category))
        )

        print(
            f"  Shared catalyst identities: "
            f"{len(shared_matrix_catalysts)}"
        )

    endpoint_export = endpoints.drop(
        columns=["_numeric_time"],
        errors="ignore",
    )

    endpoint_path = OUTPUT_DIR / "dye_removal_endpoints_99.csv"
    endpoint_export.to_csv(endpoint_path, index=False)

    trajectory_path = OUTPUT_DIR / "dye_removal_trajectory_sizes.csv"

    (
        trajectory_sizes
        .reset_index()
        .to_csv(trajectory_path, index=False)
    )

    if not dye_shift.empty:
        dye_shift.to_csv(
            OUTPUT_DIR / "dye_shift_smd.csv",
            index=False,
        )

    if not matrix_shift.empty:
        matrix_shift.to_csv(
            OUTPUT_DIR / "water_matrix_shift_smd.csv",
            index=False,
        )

    print()
    print("=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)
    print(f"UNIQUE TRAJECTORIES = {unique_trajectories}")
    print(f"ENDPOINT RECORDS = {len(endpoints)}")
    print(f"Endpoint dataset: {endpoint_path}")
    print(f"Trajectory sizes: {trajectory_path}")
    print()
    print(
        "No source/target split has been selected automatically. "
        "The dye and water-matrix candidates are reported for scientific "
        "comparison based on sample size, overlap, shift, and confounding."
    )


if __name__ == "__main__":
    main()
