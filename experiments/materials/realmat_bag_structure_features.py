from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
DATA_DIR = DATA_ROOT / "data"
CIF_DIR = DATA_DIR / "cif_file"

PRETRAIN_FILE = DATA_DIR / "pretrain_data.json"
TRAIN_FILE = DATA_DIR / "fine_tune" / "train_data.json"
TEST_FILE = DATA_DIR / "fine_tune" / "test_data.json"

OUTPUT_DIR = DATA_ROOT / "features"

OUTPUT_FILE = OUTPUT_DIR / "realmat_bag_structure_features.parquet"


def load_targets(path, domain):
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


def composition_features(structure):
    composition = structure.composition
    elements = composition.elements

    atomic_numbers = np.array(
        [element.Z for element in elements],
        dtype=float,
    )

    atomic_masses = np.array(
        [element.atomic_mass for element in elements],
        dtype=float,
    )

    fractions = np.array(
        [
            float(composition.get_atomic_fraction(element))
            for element in elements
        ],
        dtype=float,
    )

    entropy = float(
        -np.sum(
            [
                fraction * np.log(fraction)
                for fraction in fractions
                if fraction > 0
            ]
        )
    )

    return {
        "n_elements": len(elements),
        "total_atoms": float(structure.num_sites),
        "mean_atomic_number": float(np.average(atomic_numbers, weights=fractions)),
        "min_atomic_number": float(np.min(atomic_numbers)),
        "max_atomic_number": float(np.max(atomic_numbers)),
        "mean_atomic_mass": float(np.average(atomic_masses, weights=fractions)),
        "min_atomic_mass": float(np.min(atomic_masses)),
        "max_atomic_mass": float(np.max(atomic_masses)),
        "composition_entropy": entropy,
    }


def structure_features(structure):
    lattice = structure.lattice

    lengths = np.array(
        lattice.abc,
        dtype=float,
    )

    angles = np.array(
        lattice.angles,
        dtype=float,
    )

    volume = float(lattice.volume)

    density = float(structure.density)

    comp = composition_features(structure)

    result = {
        "a": float(lengths[0]),
        "b": float(lengths[1]),
        "c": float(lengths[2]),
        "alpha": float(angles[0]),
        "beta": float(angles[1]),
        "gamma": float(angles[2]),
        "cell_volume": volume,
        "density": density,
        **comp,
    }

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    records.extend(
        load_targets(
            PRETRAIN_FILE,
            "computational",
        )
    )

    records.extend(
        load_targets(
            TRAIN_FILE,
            "experimental_train",
        )
    )

    records.extend(
        load_targets(
            TEST_FILE,
            "experimental_test",
        )
    )

    print("=" * 80)
    print("REALMAT-BAG STRUCTURE FEATURE EXTRACTION")
    print("=" * 80)
    print(f"TOTAL TARGET RECORDS: {len(records)}")
    print(f"CIF DIRECTORY: {CIF_DIR}")
    print(f"OUTPUT: {OUTPUT_FILE}")

    feature_rows = []
    failures = []

    for index, record in enumerate(records, start=1):
        mpid = record["mpid"]
        cif_path = CIF_DIR / f"{mpid}.cif"

        try:
            structure = Structure.from_file(cif_path)

            features = structure_features(structure)

            row = {
                "mpid": mpid,
                "bandgap": record["bandgap"],
                "domain": record["domain"],
                **features,
            }

            feature_rows.append(row)

        except Exception as exc:
            failures.append(
                {
                    "mpid": mpid,
                    "domain": record["domain"],
                    "error": str(exc),
                }
            )

        if index % 1000 == 0:
            print(f"PROCESSED: {index} / {len(records)}")

    dataframe = pd.DataFrame(feature_rows)

    dataframe.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    failure_file = OUTPUT_DIR / "structure_feature_failures.json"

    with failure_file.open("w", encoding="utf-8") as f:
        json.dump(
            failures,
            f,
            indent=2,
        )

    digest = hashlib.sha256()

    with OUTPUT_FILE.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    print()
    print("=" * 80)
    print("FEATURE EXTRACTION RESULT")
    print("=" * 80)

    print(f"SUCCESSFUL RECORDS: {len(dataframe)}")
    print(f"FAILED RECORDS: {len(failures)}")
    print(f"FEATURE COUNT: {len(dataframe.columns)}")

    if len(dataframe):
        print()
        print("COLUMNS:")
        for column in dataframe.columns:
            print(column)

        print()
        print("DOMAIN COUNTS:")
        print(dataframe["domain"].value_counts())

        print()
        print("MISSING VALUES:")
        print(dataframe.isna().sum())

    print()
    print(f"PARQUET SHA256: {digest.hexdigest()}")

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()