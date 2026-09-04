import torch


def domain_support_score(
    domain_probabilities: torch.Tensor,
) -> torch.Tensor:
    if domain_probabilities.ndim != 2:
        raise ValueError(
            "domain_probabilities must have shape [N, 2]"
        )

    if domain_probabilities.shape[1] != 2:
        raise ValueError(
            "domain_probabilities must contain source and target probabilities"
        )

    source_prob = domain_probabilities[:, 0]
    return source_prob.clamp(0.0, 1.0)


def dual_reliability_score(
    drl_probabilities: torch.Tensor,
    domain_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence = drl_probabilities.max(dim=1).values
    support = domain_support_score(domain_probabilities)

    score = confidence * support

    return score, confidence, support
