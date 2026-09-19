from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "water" / "photocatalysis"
DATA_FILE = DATA_ROOT / "data" / "mg_degradation_raw.csv"

OUTPUT_DIR = DATA_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "photocatalysis_experimental_design_audit.json"

INPUT_COLUMNS = [
    "surface_area",
    "pore_volume",
    "catalyst_loading_g/l",
    "Light_intensity (W)",
    "solution_pH",
    "HA (mg/L)",
    "ini_conc_mg/l",
    "catalyst_type",
    "anions",
]

THRESHOLDS = [
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
]


def make_run_key(dataframe):
    return dataframe[
        INPUT_COLUMNS
    ].astype(str).agg(
        "||".join,
        axis=1,
    )


def main():
    print("=" * 80)
    print("WATER-DRST EXPERIMENTAL DESIGN AUDIT")
    print("=" * 80)

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"TOTAL ROWS: {len(df)}"
    )

    df["run_key"] = make_run_key(
        df
    )

    print()
    print("=" * 80)
    print("TRAJECTORY STRUCTURE")
    print("=" * 80)

    run_sizes = (
        df.groupby("run_key")
        .size()
    )

    print(
        f"UNIQUE EXPERIMENTAL RUNS: "
        f"{len(run_sizes)}"
    )

    print(
        f"ROWS PER RUN - MIN: "
        f"{run_sizes.min()}"
    )

    print(
        f"ROWS PER RUN - MAX: "
        f"{run_sizes.max()}"
    )

    print(
        f"ROWS PER RUN - MEAN: "
        f"{run_sizes.mean():.3f}"
    )

    print()
    print("RUN SIZE COUNTS:")

    print(
        run_sizes.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 80)
    print("TIME POINT AUDIT")
    print("=" * 80)

    print(
        df["time_min"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"UNIQUE TIME POINTS: "
        f"{df['time_min'].nunique()}"
    )

    print()
    print("=" * 80)
    print("FINAL-ENDPOINT AUDIT")
    print("=" * 80)

    endpoint = (
        df.sort_values(
            [
                "run_key",
                "time_min",
            ]
        )
        .groupby(
            "run_key",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    print(
        f"ENDPOINT RUNS: "
        f"{len(endpoint)}"
    )

    print(
        f"ENDPOINT TIMES:"
    )

    print(
        endpoint[
            "time_min"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 80)
    print("WATER MATRIX DOMAINS")
    print("=" * 80)

    endpoint["matrix_domain"] = np.where(
        (
            (endpoint["HA (mg/L)"] == 0)
            &
            (
                endpoint["anions"]
                == "without Anion"
            )
        ),
        "controlled",
        "matrix_stressed",
    )

    print(
        endpoint[
            "matrix_domain"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("MATRIX DOMAIN BY HUMIC ACID:")

    print(
        pd.crosstab(
            endpoint["HA (mg/L)"],
            endpoint["matrix_domain"],
        ).to_string()
    )

    print()
    print("MATRIX DOMAIN BY ANION:")

    print(
        pd.crosstab(
            endpoint["anions"],
            endpoint["matrix_domain"],
        ).to_string()
    )

    print()
    print("=" * 80)
    print("CATALYST DISTRIBUTION")
    print("=" * 80)

    catalyst_table = pd.crosstab(
        endpoint["catalyst_type"],
        endpoint["matrix_domain"],
    )

    print(
        catalyst_table.to_string()
    )

    print()
    print("=" * 80)
    print("TARGET DISTRIBUTION")
    print("=" * 80)

    endpoint_target = endpoint[
        "Efficiency (%)"
    ].astype(float)

    for domain in [
        "controlled",
        "matrix_stressed",
    ]:
        subset = endpoint[
            endpoint["matrix_domain"]
            == domain
        ]

        values = subset[
            "Efficiency (%)"
        ].astype(float)

        print()
        print(
            domain.upper()
        )

        print(
            f"N: {len(values)}"
        )

        print(
            f"MIN: {values.min():.6f}"
        )

        print(
            f"MAX: {values.max():.6f}"
        )

        print(
            f"MEAN: {values.mean():.6f}"
        )

        print(
            f"MEDIAN: {values.median():.6f}"
        )

        print(
            f"STD: {values.std():.6f}"
        )

    print()
    print("=" * 80)
    print("CLASS THRESHOLD SENSITIVITY")
    print("=" * 80)

    threshold_results = []

    for threshold in THRESHOLDS:
        controlled = endpoint[
            endpoint["matrix_domain"]
            == "controlled"
        ]

        stressed = endpoint[
            endpoint["matrix_domain"]
            == "matrix_stressed"
        ]

        controlled_low = int(
            (
                controlled[
                    "Efficiency (%)"
                ]
                < threshold
            ).sum()
        )

        controlled_high = int(
            (
                controlled[
                    "Efficiency (%)"
                ]
                >= threshold
            ).sum()
        )

        stressed_low = int(
            (
                stressed[
                    "Efficiency (%)"
                ]
                < threshold
            ).sum()
        )

        stressed_high = int(
            (
                stressed[
                    "Efficiency (%)"
                ]
                >= threshold
            ).sum()
        )

        print(
            f"THRESHOLD={threshold:.1f}% "
            f"CONTROLLED=({controlled_low},{controlled_high}) "
            f"STRESSED=({stressed_low},{stressed_high})"
        )

        threshold_results.append(
            {
                "threshold_percent": threshold,
                "controlled_class_0": controlled_low,
                "controlled_class_1": controlled_high,
                "stressed_class_0": stressed_low,
                "stressed_class_1": stressed_high,
            }
        )

    print()
    print("=" * 80)
    print("DUPLICATE EXPERIMENTAL CONDITIONS")
    print("=" * 80)

    condition_counts = (
        endpoint[
            INPUT_COLUMNS
        ]
        .astype(str)
        .agg(
            "||".join,
            axis=1,
        )
        .value_counts()
    )

    duplicated_conditions = (
        condition_counts[
            condition_counts > 1
        ]
    )

    print(
        f"UNIQUE ENDPOINT CONDITIONS: "
        f"{len(condition_counts)}"
    )

    print(
        f"CONDITIONS APPEARING >1 TIME: "
        f"{len(duplicated_conditions)}"
    )

    if len(duplicated_conditions):
        print(
            duplicated_conditions.head(
                20
            ).to_string()
        )

    print()
    print("=" * 80)
    print("ANOMALY CHECK")
    print("=" * 80)

    print(
        f"MISSING EFFICIENCY: "
        f"{endpoint['Efficiency (%)'].isna().sum()}"
    )

    print(
        f"INFINITE EFFICIENCY: "
        f"{np.isinf(endpoint['Efficiency (%)']).sum()}"
    )

    print(
        f"MISSING FINAL CONCENTRATION: "
        f"{endpoint['final_conc_mg/l'].isna().sum()}"
    )

    print(
        f"MISSING CATALYST: "
        f"{endpoint['catalyst_type'].isna().sum()}"
    )

    print(
        f"MISSING ANION: "
        f"{endpoint['anions'].isna().sum()}"
    )

    print()
    print("=" * 80)
    print("RUN COMPLETENESS")
    print("=" * 80)

    expected_time_points = sorted(
        df["time_min"]
        .dropna()
        .unique()
        .tolist()
    )

    complete_runs = 0
    incomplete_runs = 0

    for run_key, group in df.groupby(
        "run_key"
    ):
        observed = sorted(
            group[
                "time_min"
            ].dropna().unique().tolist()
        )

        if observed == expected_time_points:
            complete_runs += 1
        else:
            incomplete_runs += 1

    print(
        f"COMPLETE RUNS: "
        f"{complete_runs}"
    )

    print(
        f"INCOMPLETE RUNS: "
        f"{incomplete_runs}"
    )

    audit = {
        "rows": int(len(df)),
        "unique_runs": int(len(run_sizes)),
        "run_size_min": int(run_sizes.min()),
        "run_size_max": int(run_sizes.max()),
        "run_size_mean": float(run_sizes.mean()),
        "time_points": [
            int(value)
            for value in sorted(
                df["time_min"]
                .unique()
            )
        ],
        "endpoint_runs": int(len(endpoint)),
        "matrix_domain_counts": {
            str(key): int(value)
            for key, value in endpoint[
                "matrix_domain"
            ]
            .value_counts()
            .items()
        },
        "catalyst_distribution": {
            str(key): {
                str(inner_key): int(inner_value)
                for inner_key, inner_value in row.items()
            }
            for key, row in catalyst_table.iterrows()
        },
        "threshold_sensitivity": threshold_results,
        "complete_runs": int(
            complete_runs
        ),
        "incomplete_runs": int(
            incomplete_runs
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("EXPERIMENTAL DESIGN AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"OUTPUT: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()