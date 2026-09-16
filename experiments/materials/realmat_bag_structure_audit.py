from pathlib import Path
import json
import zipfile
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "domains" / "materials" / "realmat_bag"
DATA_DIR = DATA_ROOT / "data"
CIF_ARCHIVE = DATA_ROOT / "cif_file.zip"

PRETRAIN_FILE = DATA_DIR / "pretrain_data.json"
TRAIN_FILE = DATA_DIR / "fine_tune" / "train_data.json"
TEST_FILE = DATA_DIR / "fine_tune" / "test_data.json"

CIF_DIR = DATA_ROOT / "data" / "cif_file"


def load_mpids(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dictionary in {path}")

    return {str(k) for k in data.keys()}


def find_cif_mpids(cif_dir):
    files = list(cif_dir.rglob("*.cif"))
    mpids = set()
    unmatched = []

    for path in files:
        name = path.stem

        match = re.search(r"(mp-\d+|mvc-\d+)", name)

        if match:
            mpids.add(match.group(1))
        else:
            unmatched.append(path.name)

    return files, mpids, unmatched


def main():
    print("=" * 80)
    print("ADAPTIVE-DRST REALMAT-BAG STRUCTURE AUDIT")
    print("=" * 80)
    print(f"PROJECT ROOT: {ROOT}")
    print(f"DATA ROOT: {DATA_ROOT}")
    print(f"ARCHIVE: {CIF_ARCHIVE}")
    print(f"CIF DIRECTORY: {CIF_DIR}")

    print()
    print("=" * 80)
    print("DATASET MPID COUNTS")
    print("=" * 80)

    pretrain_ids = load_mpids(PRETRAIN_FILE)
    train_ids = load_mpids(TRAIN_FILE)
    test_ids = load_mpids(TEST_FILE)

    print(f"PRETRAIN MPIDS: {len(pretrain_ids)}")
    print(f"EXPERIMENTAL TRAIN MPIDS: {len(train_ids)}")
    print(f"EXPERIMENTAL TEST MPIDS: {len(test_ids)}")

    all_ids = pretrain_ids | train_ids | test_ids

    print(f"TOTAL UNIQUE MPIDS ACROSS DATASETS: {len(all_ids)}")

    print()
    print("=" * 80)
    print("ARCHIVE CHECK")
    print("=" * 80)

    if not CIF_ARCHIVE.exists():
        print("CIF ARCHIVE STATUS: MISSING")
        print("Expected archive:")
        print(CIF_ARCHIVE)
        return

    print("CIF ARCHIVE STATUS: FOUND")
    print(f"ARCHIVE SIZE: {CIF_ARCHIVE.stat().st_size / (1024 ** 3):.3f} GB")

    with zipfile.ZipFile(CIF_ARCHIVE, "r") as z:
        members = z.namelist()

        cif_members = [
            name for name in members
            if name.lower().endswith(".cif")
        ]

        print(f"ZIP MEMBERS: {len(members)}")
        print(f"CIF FILES IN ZIP: {len(cif_members)}")

        archive_mpids = set()
        unmatched_archive_files = []

        for name in cif_members:
            match = re.search(r"(mp-\d+|mvc-\d+)", Path(name).stem)

            if match:
                archive_mpids.add(match.group(1))
            else:
                unmatched_archive_files.append(name)

        print(f"MPIDS IDENTIFIED IN ZIP: {len(archive_mpids)}")
        print(
            f"ZIP CIF FILES WITHOUT RECOGNIZED MPID: "
            f"{len(unmatched_archive_files)}"
        )

    print()
    print("=" * 80)
    print("MPID COVERAGE BEFORE EXTRACTION")
    print("=" * 80)

    pretrain_missing = pretrain_ids - archive_mpids
    train_missing = train_ids - archive_mpids
    test_missing = test_ids - archive_mpids

    print(
        f"PRETRAIN COVERAGE: "
        f"{len(pretrain_ids - pretrain_missing)} / {len(pretrain_ids)}"
    )

    print(
        f"EXPERIMENTAL TRAIN COVERAGE: "
        f"{len(train_ids - train_missing)} / {len(train_ids)}"
    )

    print(
        f"EXPERIMENTAL TEST COVERAGE: "
        f"{len(test_ids - test_missing)} / {len(test_ids)}"
    )

    print(f"PRETRAIN MISSING: {len(pretrain_missing)}")
    print(f"EXPERIMENTAL TRAIN MISSING: {len(train_missing)}")
    print(f"EXPERIMENTAL TEST MISSING: {len(test_missing)}")

    if pretrain_missing:
        print("FIRST PRETRAIN MISSING MPIDS:")
        print(sorted(pretrain_missing)[:20])

    if train_missing:
        print("FIRST EXPERIMENTAL TRAIN MISSING MPIDS:")
        print(sorted(train_missing)[:20])

    if test_missing:
        print("FIRST EXPERIMENTAL TEST MISSING MPIDS:")
        print(sorted(test_missing)[:20])

    print()
    print("=" * 80)
    print("EXTRACTION")
    print("=" * 80)

    CIF_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0

    with zipfile.ZipFile(CIF_ARCHIVE, "r") as z:
        for member in z.namelist():
            if not member.lower().endswith(".cif"):
                continue

            destination = CIF_DIR / Path(member).name

            if destination.exists():
                skipped += 1
                continue

            with z.open(member) as source, destination.open("wb") as target:
                target.write(source.read())

            extracted += 1

    print(f"NEW CIF FILES EXTRACTED: {extracted}")
    print(f"CIF FILES ALREADY PRESENT: {skipped}")

    print()
    print("=" * 80)
    print("LOCAL CIF AUDIT")
    print("=" * 80)

    cif_files, local_mpids, unmatched_local = find_cif_mpids(CIF_DIR)

    print(f"LOCAL CIF FILES: {len(cif_files)}")
    print(f"LOCAL MPIDS IDENTIFIED: {len(local_mpids)}")
    print(f"LOCAL FILES WITHOUT MPID: {len(unmatched_local)}")

    pretrain_local_missing = pretrain_ids - local_mpids
    train_local_missing = train_ids - local_mpids
    test_local_missing = test_ids - local_mpids

    print(
        f"PRETRAIN LOCAL COVERAGE: "
        f"{len(pretrain_ids - pretrain_local_missing)} / {len(pretrain_ids)}"
    )

    print(
        f"EXPERIMENTAL TRAIN LOCAL COVERAGE: "
        f"{len(train_ids - train_local_missing)} / {len(train_ids)}"
    )

    print(
        f"EXPERIMENTAL TEST LOCAL COVERAGE: "
        f"{len(test_ids - test_local_missing)} / {len(test_ids)}"
    )

    print()
    print("=" * 80)
    print("STRUCTURE DOMAIN OVERLAP")
    print("=" * 80)

    computational_experimental_train = pretrain_ids & train_ids
    computational_experimental_test = pretrain_ids & test_ids
    experimental_train_test = train_ids & test_ids

    print(
        f"COMPUTATIONAL / EXPERIMENTAL TRAIN: "
        f"{len(computational_experimental_train)}"
    )

    print(
        f"COMPUTATIONAL / EXPERIMENTAL TEST: "
        f"{len(computational_experimental_test)}"
    )

    print(
        f"EXPERIMENTAL TRAIN / TEST: "
        f"{len(experimental_train_test)}"
    )

    print()
    print("=" * 80)
    print("SAMPLE CIF CONTENT CHECK")
    print("=" * 80)

    sample_files = sorted(cif_files)[:5]

    for path in sample_files:
        print()
        print(f"FILE: {path.name}")

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            lines = text.splitlines()

            print(f"LINES: {len(lines)}")

            for line in lines[:8]:
                print(line)

        except Exception as exc:
            print(f"READ ERROR: {exc}")

    print()
    print("=" * 80)
    print("FINAL STRUCTURE COVERAGE SUMMARY")
    print("=" * 80)

    total_missing = all_ids - local_mpids

    print(f"ALL DATASET MPIDS: {len(all_ids)}")
    print(f"ALL DATASET MPIDS WITH CIF: {len(all_ids & local_mpids)}")
    print(f"ALL DATASET MPIDS WITHOUT CIF: {len(total_missing)}")

    if not total_missing:
        print("STRUCTURE COVERAGE STATUS: COMPLETE")
    else:
        print("STRUCTURE COVERAGE STATUS: INCOMPLETE")

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
