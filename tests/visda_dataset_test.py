import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.visda.dataset import VisDAClassificationDataset


def main():
    dataset = VisDAClassificationDataset(
        "data/visda/train"
    )

    print("Dataset size:", len(dataset))
    print("Classes:", dataset.classes)


if __name__ == "__main__":
    main()
