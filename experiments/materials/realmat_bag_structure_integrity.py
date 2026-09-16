from pathlib import Path
import json
import zipfile
import re

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
DATA_DIR = DATA_ROOT / "data"

PRETRAIN_FILE = DATA_DIR / "pretrain_data.json"
TRAIN_FILE = DATA_DIR / "fine_tune" / "train_data.json"
TEST_FILE = DATA_DIR / "fine_tune" / "test_data.json"

CIF_ARCHIVE = DATA_ROOT / "cif_file.zip"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_mpid_from_filename(name):
    match = re.search(r"(mp-\d+)", Path(name).stem)
    if match:
        return match.group(1)
    return None


def main():
    print("=" * 80)
    print("ADAPTIVE-DRST REALMAT-BAG STRUCTURE INTEGRITY AUDIT")
    print("=" * 80)

    pretrain = load_json(PRETRAIN_FILE)
    train = load_json(TRAIN_FILE)
    test = load_json(TEST_FILE)

    pretrain_ids = {str(k) for k in pretrain.keys()}
    train_ids = {str(k) for k in train.keys()}
    test_ids = {str(k) for k in test.keys()}

    all_ids = pretrain_ids | train_ids | test_ids

    print()
    print("=" * 80)
    print("MISSING DATASET STRUCTURE")
    print("=" * 80)

    cif_names = []

    with zipfile.ZipFile(CIF_ARCHIVE, "r") as z:
        cif_names = [
            name
            for name in z.namelist()
            if name.lower().endswith(".cif")
        ]

    archive_mpids = set()
    unmatched = []

    for name in cif_names:
        mpid = extract_mpid_from_filename(name)

        if mpid is None:
            unmatched.append(name)
        else:
            archive_mpids.add(mpid)

    missing = sorted(all_ids - archive_mpids)

    print(f"TOTAL DATASET MPIDS: {len(all_ids)}")
    print(f"MPIDS WITH RECOGNIZED CIF: {len(all_ids & archive_mpids)}")
    print(f"MPIDS WITHOUT RECOGNIZED CIF: {len(missing)}")

    for mpid in missing:
        print(f"MISSING DATASET ID: {mpid}")

        if mpid in pretrain:
            print("DOMAIN: COMPUTATIONAL / PRETRAIN")
            print(f"VALUE: {pretrain[mpid]}")

        elif mpid in train:
            print("DOMAIN: EXPERIMENTAL / TRAIN")
            print(f"VALUE: {train[mpid]}")

        elif mpid in test:
            print("DOMAIN: EXPERIMENTAL / TEST")
            print(f"VALUE: {test[mpid]}")

    print()
    print("=" * 80)
    print("UNMATCHED CIF FILES")
    print("=" * 80)

    print(f"UNMATCHED CIF COUNT: {len(unmatched)}")

    for name in unmatched:
        print(f"UNMATCHED CIF: {name}")

        with zipfile.ZipFile(CIF_ARCHIVE, "r") as z:
            text = z.read(name).decode(
                "utf-8",
                errors="replace"
            )

        lines = text.splitlines()

        print(f"CIF LINES: {len(lines)}")

        for line in lines[:20]:
            print(line)

    print()
    print("=" * 80)
    print("CROSS-DOMAIN ANOMALY CHECK")
    print("=" * 80)

    print(
        f"COMPUTATIONAL / EXPERIMENTAL TRAIN OVERLAP: "
        f"{len(pretrain_ids & train_ids)}"
    )

    print(
        f"COMPUTATIONAL / EXPERIMENTAL TEST OVERLAP: "
        f"{len(pretrain_ids & test_ids)}"
    )

    print(
        f"EXPERIMENTAL TRAIN / TEST OVERLAP: "
        f"{len(train_ids & test_ids)}"
    )

    print()
    print("=" * 80)
    print("INTEGRITY RESULT")
    print("=" * 80)

    if len(missing) == 0 and len(unmatched) == 0:
        print("STATUS: CLEAN")
    else:
        print("STATUS: ANOMALIES REQUIRE REVIEW")

    print("=" * 80)


if __name__ == "__main__":
    main()