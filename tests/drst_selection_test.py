import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.drst.self_training import select_pseudo_labels


def main():
    probabilities = torch.tensor(
        [
            [0.40, 0.35, 0.25],
            [0.95, 0.03, 0.02],
            [0.20, 0.70, 0.10],
            [0.55, 0.30, 0.15],
            [0.10, 0.15, 0.75],
        ]
    )

    indices, labels, confidences = select_pseudo_labels(
        probabilities,
        portion=0.4,
    )

    print("Selected indices:", indices)
    print("Pseudo-labels:", labels)
    print("Confidences:", confidences)

    assert len(indices) == 2
    assert torch.all(confidences[:-1] >= confidences[1:])
    assert torch.equal(
        labels,
        torch.tensor([0, 2]),
    )

    print("DRST selection test OK")


if __name__ == "__main__":
    main()
