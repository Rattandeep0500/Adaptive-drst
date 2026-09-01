import torch

from src.drl.predictor import drl_probabilities
from src.drst.self_training import select_pseudo_labels


def generate_pseudo_labels(
    classifier,
    domain_network,
    target_x: torch.Tensor,
    portion: float = 0.2,
):
    """
    Generate DRL-based pseudo-labels for target data.

    Returns:
        selected_indices
        pseudo_labels
        confidences
        drl_probabilities
    """

    classifier.eval()
    domain_network.eval()

    with torch.no_grad():
        logits = classifier(target_x)

        domain_probs = domain_network.domain_probabilities(
            target_x
        )

        source_prob = domain_probs[:, 0].clamp_min(1e-8)
        target_prob = domain_probs[:, 1].clamp_min(1e-8)

        density_ratio = source_prob / target_prob

        probabilities = drl_probabilities(
            logits,
            density_ratio,
        )

    indices, labels, confidences = select_pseudo_labels(
        probabilities,
        portion=portion,
    )

    return (
        indices,
        labels,
        confidences,
        probabilities,
    )
