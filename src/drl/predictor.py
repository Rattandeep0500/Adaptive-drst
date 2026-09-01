import torch


def drl_probabilities(
    logits: torch.Tensor,
    density_ratio: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the DRL-adjusted class probabilities.

    The paper scales the class logits by the source/target
    density ratio before applying softmax.

    Args:
        logits: Tensor of shape [batch_size, num_classes].
        density_ratio: Tensor of shape [batch_size].

    Returns:
        Tensor of shape [batch_size, num_classes].
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes]")

    if density_ratio.ndim != 1:
        raise ValueError("density_ratio must have shape [batch_size]")

    if logits.shape[0] != density_ratio.shape[0]:
        raise ValueError("Batch sizes must match")

    ratio = density_ratio.clamp_min(1e-8).unsqueeze(1)

    scaled_logits = logits * ratio

    return torch.softmax(scaled_logits, dim=1)
