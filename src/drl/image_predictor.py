import torch

from src.drl.predictor import drl_probabilities


def image_drl_probabilities(
    model,
    x: torch.Tensor,
) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        features = model.extract_features(x)
        logits = model.classifier.backbone.fc(features)
        ratio = model.domain_network.density_ratio(features)

        return drl_probabilities(
            logits,
            ratio,
        )
