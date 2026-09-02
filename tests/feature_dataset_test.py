import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.feature_dataset import FeatureDataset


def main():
    features = torch.randn(10, 2048)
    labels = torch.randint(0, 12, (10,))

    dataset = FeatureDataset(
        features,
        labels,
    )

    print("Dataset size:", len(dataset))
    print("Feature shape:", dataset[0][0].shape)
    print("Label shape:", dataset[0][1].shape)

    assert len(dataset) == 10
    assert dataset[0][0].shape == (2048,)

    print("Feature dataset OK")


if __name__ == "__main__":
    main()
