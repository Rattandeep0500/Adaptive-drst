import torch

from src.drl.predictor import drl_probabilities
from src.drst.adaptive_selector import adaptive_select_pseudo_labels


def generate_adaptive_pseudo_labels(
    classifier,
    domain_network,
    target_x: torch.Tensor,
    min_threshold: float = 0.70,
    max_threshold: float = 0.95,
):
    classifier.eval()
    domain_network.eval()

    with torch.no_grad():
        features = classifier.extract_features(target_x)
        logits = classifier.classifier(features)

        domain_probs = domain_network.domain_probabilities(features)

        source_prob = domain_probs[:, 0].clamp_min(1e-8)
        target_prob = domain_probs[:, 1].clamp_min(1e-8)

        ratio = source_prob / target_prob

        probabilities = drl_probabilities(
            logits,
            ratio,
        )

    return adaptive_select_pseudo_labels(
        probabilities,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )