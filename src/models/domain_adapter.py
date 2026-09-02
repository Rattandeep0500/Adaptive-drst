import torch
import torch.nn as nn

from src.drl.density_ratio import DensityRatioNetwork
from src.models.image_classifier import ImageClassifier


class ImageDRLModel(nn.Module):
    def __init__(self, num_classes: int = 12) -> None:
        super().__init__()

        self.classifier = ImageClassifier(
            num_classes=num_classes,
            pretrained=False,
        )

        self.domain_network = DensityRatioNetwork(
            input_dim=2048,
            hidden_dim=512,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.extract_features(x)

    def classification_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier.backbone.fc(features)

    def domain_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.domain_network(features)
