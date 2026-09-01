import torch


def select_pseudo_labels(
    probabilities: torch.Tensor,
    portion: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select the most confident target predictions.

    Args:
        probabilities:
            Tensor with shape [N, num_classes].
        portion:
            Fraction of target samples to pseudo-label.

    Returns:
        selected_indices:
            Indices of selected target samples.

        pseudo_labels:
            Predicted class for each selected sample.

        confidences:
            Confidence of each selected prediction.
    """

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape [N, num_classes]"
        )

    if not 0.0 < portion <= 1.0:
        raise ValueError(
            "portion must be in the interval (0, 1]"
        )

    num_samples = probabilities.shape[0]

    num_selected = max(
        1,
        int(num_samples * portion),
    )

    confidences, labels = probabilities.max(dim=1)

    _, ranking = torch.topk(
        confidences,
        k=num_selected,
        largest=True,
    )

    return (
        ranking,
        labels[ranking],
        confidences[ranking],
    )
