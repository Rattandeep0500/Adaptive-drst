import torch
import torch.nn as nn


class DensityRatioNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def domain_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)

    def density_ratio(
        self,
        x: torch.Tensor,
        source_prior: float = 0.5,
        target_prior: float = 0.5,
    ) -> torch.Tensor:
        probs = self.domain_probabilities(x)

        source_prob = probs[:, 0].clamp_min(1e-8)
        target_prob = probs[:, 1].clamp_min(1e-8)

        return (
            source_prob / target_prob
        ) * (target_prior / source_prior)
