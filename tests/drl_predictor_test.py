import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.drl.predictor import drl_probabilities


def main():
    logits = torch.tensor([[3.0, 1.0, 0.0]])

    high_ratio = torch.tensor([2.0])
    low_ratio = torch.tensor([0.25])

    high_conf = drl_probabilities(logits, high_ratio)
    low_conf = drl_probabilities(logits, low_ratio)

    print("High-ratio probabilities:", high_conf)
    print("Low-ratio probabilities:", low_conf)

    assert high_conf[0, 0] > low_conf[0, 0]

    print("Conservativeness test OK")


if __name__ == "__main__":
    main()
