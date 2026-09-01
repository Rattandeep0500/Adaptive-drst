import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.drl.density_ratio import DensityRatioNetwork
from src.drl.predictor import drl_probabilities
from src.drst.drst import generate_pseudo_labels
from src.models.classifier import Classifier


def main():
    torch.manual_seed(42)

    target_x = torch.randn(20, 4)

    classifier = Classifier(
        input_dim=4,
        feature_dim=8,
        num_classes=3,
    )

    domain_network = DensityRatioNetwork(
        input_dim=4,
        hidden_dim=8,
    )

    indices, labels, confidences, probabilities = (
        generate_pseudo_labels(
            classifier,
            domain_network,
            target_x,
            portion=0.2,
        )
    )

    print("Selected samples:", len(indices))
    print("Pseudo-label shape:", labels.shape)
    print("Confidence shape:", confidences.shape)
    print("Probability shape:", probabilities.shape)

    assert len(indices) == 4
    assert labels.shape == (4,)
    assert confidences.shape == (4,)
    assert probabilities.shape == (20, 3)

    print("DRL + DRST integration OK")


if __name__ == "__main__":
    main()
