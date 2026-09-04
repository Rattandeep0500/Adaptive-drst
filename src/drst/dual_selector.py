import torch

from src.drst.dual_reliability import dual_reliability_score


def select_dual_reliable_pseudo_labels(
    drl_probabilities: torch.Tensor,
    domain_probabilities: torch.Tensor,
    threshold: float = 0.70,
    max_per_class: int | None = None,
):
    score, confidence, support = dual_reliability_score(
        drl_probabilities,
        domain_probabilities,
    )

    labels = drl_probabilities.argmax(dim=1)

    selected = []

    for cls in range(drl_probabilities.shape[1]):
        idx = torch.where(
            (labels == cls)
            & (score >= threshold)
        )[0]

        if max_per_class is not None:
            idx = idx[
                torch.argsort(
                    score[idx],
                    descending=True,
                )[:max_per_class]
            ]

        if idx.numel() > 0:
            selected.append(idx)

    if not selected:
        empty = torch.empty(
            0,
            dtype=torch.long,
            device=drl_probabilities.device,
        )

        return (
            empty,
            empty,
            confidence[:0],
            support[:0],
            score[:0],
        )

    indices = torch.cat(selected)

    return (
        indices,
        labels[indices],
        confidence[indices],
        support[indices],
        score[indices],
    )
