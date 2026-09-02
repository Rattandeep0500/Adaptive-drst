import torch
import torch.nn as nn
from torchvision.models import resnet50


class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        self.backbone = resnet50(weights=weights)

        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(
            feature_dim,
            num_classes,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.backbone.fc(features)
