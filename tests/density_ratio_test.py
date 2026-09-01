import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.drl.density_ratio import DensityRatioNetwork


def main():
    model = DensityRatioNetwork(input_dim=4)
    x = torch.randn(8, 4)

    probs = model.domain_probabilities(x)
    ratios = model.density_ratio(x)

    assert probs.shape == (8, 2)
    assert ratios.shape == (8,)
    assert torch.allclose(
        probs.sum(dim=1),
        torch.ones(8),
        atol=1e-6,
    )
    assert torch.all(ratios > 0)

    print("Domain probabilities shape:", probs.shape)
    print("Density ratios shape:", ratios.shape)
    print("Density-ratio test OK")


if __name__ == "__main__":
    main()
