from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
FEATURE_FILE = DATA_ROOT / "features" / "realmat_bag_structure_features.parquet"

OUTPUT_DIR = DATA_ROOT / "features"

REPORT_FILE = OUTPUT_DIR / "realmat_bag_feature_audit.csv"


def standardized_mean_difference(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    source_mean = np.mean(source)
    target_mean = np.mean(target)

    source_var = np.var(source, ddof=1)
    target_var = np.var(target, ddof=1)

    pooled_std = np.sqrt((source_var + target_var) / 2.0)

    if pooled_std == 0:
        return 0.0

    return (target_mean - source_mean) / pooled_std


def main():
    print("=" * 80)
    print("REALMAT-BAG FEATURE DISTRIBUTION AUDIT")
    print("=" * 80)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    df = pd.read_parquet(FEATURE_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FEATURE FILE: {FEATURE_FILE}")
    print(f"ROWS: {len(df)}")
    print(f"COLUMNS: {len(df.columns)}")

    feature_columns = [
        column
        for column in df.columns
        if column not in {"mpid", "bandgap", "domain"}
    ]

    source = df[df["domain"] == "computational"]
    experimental_train = df[df["domain"] == "experimental_train"]
    experimental_test = df[df["domain"] == "experimental_test"]

    experimental = df[
        df["domain"].isin(
            [
                "experimental_train",
                "experimental_test",
            ]
        )
    ]

    print()
    print("=" * 80)
    print("DOMAIN COUNTS")
    print("=" * 80)

    print(f"COMPUTATIONAL: {len(source)}")
    print(f"EXPERIMENTAL TRAIN: {len(experimental_train)}")
    print(f"EXPERIMENTAL TEST: {len(experimental_test)}")
    print(f"EXPERIMENTAL TOTAL: {len(experimental)}")

    print()
    print("=" * 80)
    print("FEATURE QUALITY")
    print("=" * 80)

    quality_rows = []

    for feature in feature_columns:
        values = df[feature].to_numpy(dtype=float)

        variance = float(np.var(values))

        quality_rows.append(
            {
                "feature": feature,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "variance": variance,
                "missing": int(df[feature].isna().sum()),
                "unique": int(df[feature].nunique()),
            }
        )

    quality_df = pd.DataFrame(quality_rows)

    print(
        quality_df[
            [
                "feature",
                "min",
                "max",
                "mean",
                "std",
                "unique",
                "missing",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("SOURCE vs EXPERIMENTAL DOMAIN SHIFT")
    print("=" * 80)

    shift_rows = []

    for feature in feature_columns:
        source_values = source[feature].to_numpy(dtype=float)
        target_values = experimental[feature].to_numpy(dtype=float)

        source_mean = float(np.mean(source_values))
        target_mean = float(np.mean(target_values))

        source_std = float(np.std(source_values))
        target_std = float(np.std(target_values))

        smd = standardized_mean_difference(
            source_values,
            target_values,
        )

        shift_rows.append(
            {
                "feature": feature,
                "source_mean": source_mean,
                "target_mean": target_mean,
                "source_std": source_std,
                "target_std": target_std,
                "mean_difference": target_mean - source_mean,
                "standardized_mean_difference": smd,
                "abs_standardized_mean_difference": abs(smd),
            }
        )

    shift_df = (
        pd.DataFrame(shift_rows)
        .sort_values(
            "abs_standardized_mean_difference",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        shift_df[
            [
                "feature",
                "source_mean",
                "target_mean",
                "source_std",
                "target_std",
                "standardized_mean_difference",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("BANDGAP DISTRIBUTION")
    print("=" * 80)

    for name, subset in [
        ("COMPUTATIONAL", source),
        ("EXPERIMENTAL TRAIN", experimental_train),
        ("EXPERIMENTAL TEST", experimental_test),
    ]:
        values = subset["bandgap"].to_numpy(dtype=float)

        print()
        print(name)
        print(f"MIN: {np.min(values):.6f}")
        print(f"MAX: {np.max(values):.6f}")
        print(f"MEAN: {np.mean(values):.6f}")
        print(f"STD: {np.std(values):.6f}")
        print(f"MEDIAN: {np.median(values):.6f}")

    print()
    print("=" * 80)
    print("FEATURE CORRELATION WITH BANDGAP")
    print("=" * 80)

    correlations = []

    for feature in feature_columns:
        correlation = df[[feature, "bandgap"]].corr().iloc[0, 1]

        correlations.append(
            {
                "feature": feature,
                "pearson_r": float(correlation),
                "abs_pearson_r": abs(float(correlation)),
            }
        )

    correlation_df = (
        pd.DataFrame(correlations)
        .sort_values(
            "abs_pearson_r",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        correlation_df[
            [
                "feature",
                "pearson_r",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("NEAR-CONSTANT FEATURES")
    print("=" * 80)

    near_constant = quality_df[
        quality_df["unique"] <= 2
    ]

    if len(near_constant) == 0:
        print("NONE")
    else:
        print(
            near_constant[
                [
                    "feature",
                    "unique",
                    "variance",
                ]
            ].to_string(index=False)
        )

    report_df = shift_df.copy()

    report_df.to_csv(
        REPORT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("AUDIT RESULT")
    print("=" * 80)

    print(f"FEATURES ANALYZED: {len(feature_columns)}")
    print(
        "LARGEST DOMAIN-SHIFT FEATURE: "
        f"{shift_df.iloc[0]['feature']}"
    )
    print(
        "LARGEST ABSOLUTE SMD: "
        f"{shift_df.iloc[0]['abs_standardized_mean_difference']:.6f}"
    )
    print(f"REPORT: {REPORT_FILE}")

    print()
    print("=" * 80)
    print("FEATURE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()