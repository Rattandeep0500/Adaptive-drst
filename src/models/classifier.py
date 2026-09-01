import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Feature extractor + linear classification head.

    This follows the mathematical structure in the DRL paper:
        feature representation phi(x, alpha)
        linear head w * phi(x, alpha) + b
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
