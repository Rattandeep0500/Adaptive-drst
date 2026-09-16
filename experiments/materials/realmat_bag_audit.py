from pathlib import Path
import json
import hashlib
from collections import Counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
PRETRAIN_FILE = DATA_ROOT / "data" / "pretrain_data.json"
TRAIN_FILE = DATA_ROOT / "data" / "fine_tune" / "train_data.json"
TEST_FILE = DATA_ROOT / "data" / "fine_tune" / "test_data.json"

CLASS_THRESHOLD_EV = 1.5


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_records(raw):
    records = []

    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                record = dict(value)
                record["_mpid"] = str(key)
                records.append(record)
            else:
                records.append(
                    {
                        "_mpid": str(key),
                        "bg": value,
                    }
                )
        return records

    if isinstance(raw, list):
        return raw

    raise TypeError(f"Unsupported JSON structure: {type(raw)}")


def extract_mpid(record):
    if not isinstance(record, dict):
        return None

    for key in (
        "_mpid",
        "mpid",
        "MPID",
        "material_id",
        "materialId",
        "materials_project_id",
        "materials_project",
    ):
        if key in record:
            return str(record[key])

    return None


def extract_bandgap(record):
    if not isinstance(record, dict):
        return None

    for key in (
        "bg",
        "bandgap",
        "band_gap",
        "Bandgap",
        "Band Gap",
        "target",
        "y",
        "value",
    ):
        if key in record:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                return None

    return None


def structure_signature(record):
    if not isinstance(record, dict):
        return None

    fields = [
        "formula",
        "pretty_formula",
        "composition",
        "structure",
        "cif",
        "cif_string",
    ]

    values = []

    for key in fields:
        if key in record:
            values.append(str(record[key]))

    if not values:
        return None

    raw = "||".join(values)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def inspect_dataset(name, path):
    print()
    print("=" * 80)
    print(name)
    print(f"FILE: {path}")

    if not path.exists():
        print("STATUS: MISSING")
        return []

    raw = load_json(path)
    records = normalize_records(raw)

    print(f"RAW TYPE: {type(raw).__name__}")
    print(f"SAMPLES: {len(records)}")
    print(f"SHA256: {sha256_file(path)}")

    if records:
        print("FIRST NORMALIZED RECORD:")
        print(records[0])

    mpids = []
    bandgaps = []
    signatures = []

    for record in records:
        mpid = extract_mpid(record)
        gap = extract_bandgap(record)
        sig = structure_signature(record)

        if mpid is not None:
            mpids.append(mpid)

        if gap is not None and np.isfinite(gap):
            bandgaps.append(gap)

        if sig is not None:
            signatures.append(sig)

    print(f"MPID PRESENT: {len(mpids)} / {len(records)}")
    print(f"BANDGAP PRESENT: {len(bandgaps)} / {len(records)}")
    print(f"STRUCTURE SIGNATURE PRESENT: {len(signatures)} / {len(records)}")

    if bandgaps:
        arr = np.asarray(bandgaps, dtype=float)

        print(f"BANDGAP MIN: {arr.min():.6f} eV")
        print(f"BANDGAP MAX: {arr.max():.6f} eV")
        print(f"BANDGAP MEAN: {arr.mean():.6f} eV")
        print(f"BANDGAP MEDIAN: {np.median(arr):.6f} eV")

        low = int(np.sum(arr < CLASS_THRESHOLD_EV))
        high = int(np.sum(arr >= CLASS_THRESHOLD_EV))

        print(
            f"CLASS THRESHOLD: {CLASS_THRESHOLD_EV:.3f} eV "
            f"(class 0: < threshold, class 1: >= threshold)"
        )
        print(f"CLASS 0: {low}")
        print(f"CLASS 1: {high}")
        print(f"CLASS 0 FRACTION: {low / len(arr):.6f}")
        print(f"CLASS 1 FRACTION: {high / len(arr):.6f}")

    if mpids:
        counts = Counter(mpids)

        duplicated = sum(
            1 for count in counts.values()
            if count > 1
        )

        repeated_rows = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        print(f"UNIQUE MPIDS: {len(counts)}")
        print(f"MPIDS WITH DUPLICATES: {duplicated}")
        print(f"DUPLICATE ROWS BY MPID: {repeated_rows}")

    if signatures:
        sig_counts = Counter(signatures)

        duplicated = sum(
            1 for count in sig_counts.values()
            if count > 1
        )

        repeated_rows = sum(
            count - 1
            for count in sig_counts.values()
            if count > 1
        )

        print(f"UNIQUE STRUCTURE SIGNATURES: {len(sig_counts)}")
        print(f"DUPLICATE STRUCTURE SIGNATURES: {duplicated}")
        print(f"DUPLICATE ROWS BY STRUCTURE: {repeated_rows}")

    return records


def main():
    print("=" * 80)
    print("ADAPTIVE-DRST REALMAT-BAG DATASET AUDIT")
    print("=" * 80)
    print(f"PROJECT ROOT: {ROOT}")
    print(f"DATA ROOT: {DATA_ROOT}")
    print(f"CLASSIFICATION THRESHOLD: {CLASS_THRESHOLD_EV:.3f} eV")

    pretrain = inspect_dataset(
        "COMPUTATIONAL / PRETRAIN",
        PRETRAIN_FILE,
    )

    train = inspect_dataset(
        "EXPERIMENTAL / TRAIN",
        TRAIN_FILE,
    )

    test = inspect_dataset(
        "EXPERIMENTAL / TEST",
        TEST_FILE,
    )

    print()
    print("=" * 80)
    print("CROSS-DOMAIN INTEGRITY")
    print("=" * 80)

    pretrain_ids = {
        extract_mpid(r)
        for r in pretrain
        if extract_mpid(r) is not None
    }

    train_ids = {
        extract_mpid(r)
        for r in train
        if extract_mpid(r) is not None
    }

    test_ids = {
        extract_mpid(r)
        for r in test
        if extract_mpid(r) is not None
    }

    train_test_overlap = train_ids & test_ids
    pretrain_train_overlap = pretrain_ids & train_ids
    pretrain_test_overlap = pretrain_ids & test_ids

    print(f"PRETRAIN UNIQUE MPIDS: {len(pretrain_ids)}")
    print(f"EXPERIMENTAL TRAIN UNIQUE MPIDS: {len(train_ids)}")
    print(f"EXPERIMENTAL TEST UNIQUE MPIDS: {len(test_ids)}")

    print(f"TRAIN / TEST MPID OVERLAP: {len(train_test_overlap)}")
    print(
        f"PRETRAIN / TRAIN MPID OVERLAP: "
        f"{len(pretrain_train_overlap)}"
    )
    print(
        f"PRETRAIN / TEST MPID OVERLAP: "
        f"{len(pretrain_test_overlap)}"
    )

    if train_test_overlap:
        print("WARNING: EXPERIMENTAL TRAIN/TEST OVERLAP DETECTED")
        print(sorted(train_test_overlap)[:20])
    else:
        print("TRAIN/TEST LEAKAGE CHECK: PASS")

    if pretrain_train_overlap:
        print("NOTE: COMPUTATIONAL/EXPERIMENTAL TRAIN MPID OVERLAP EXISTS")
        print(sorted(pretrain_train_overlap)[:20])

    if pretrain_test_overlap:
        print("NOTE: COMPUTATIONAL/EXPERIMENTAL TEST MPID OVERLAP EXISTS")
        print(sorted(pretrain_test_overlap)[:20])

    print()
    print("=" * 80)
    print("DATASET TOTALS")
    print("=" * 80)

    print(f"COMPUTATIONAL SAMPLES: {len(pretrain)}")
    print(f"EXPERIMENTAL TRAIN SAMPLES: {len(train)}")
    print(f"EXPERIMENTAL TEST SAMPLES: {len(test)}")
    print(f"EXPERIMENTAL TOTAL: {len(train) + len(test)}")

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()