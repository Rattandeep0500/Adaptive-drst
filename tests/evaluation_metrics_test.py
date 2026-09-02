import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.evaluation.metrics import (
    accuracy,
    brier_score,
    reliability_bins,
)


def main():
    probabilities = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.20, 0.20, 0.60],
            [0.70, 0.20, 0.10],
        ]
    )

    labels = torch.tensor([0, 1, 2, 1])

    acc = accuracy(
        probabilities,
        labels,
    )

    brier = brier_score(
        probabilities,
        labels,
    )

    bins = reliability_bins(
        probabilities,
        labels,
        num_bins=5,
    )

    print("Accuracy:", acc)
    print("Brier score:", brier)
    print("Reliability bins:", bins)
    print("Evaluation metrics OK")


if __name__ == "__main__":
    main()
