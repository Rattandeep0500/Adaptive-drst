import torch


def accuracy(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    predictions = probabilities.argmax(dim=1)
    return float(
        (predictions == labels).float().mean().item()
    )


def brier_score(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    num_classes = probabilities.shape[1]

    targets = torch.zeros_like(probabilities)
    targets.scatter_(
        1,
        labels.unsqueeze(1),
        1.0,
    )

    return float(
        ((probabilities - targets) ** 2).sum(dim=1).mean().item()
    )


def reliability_bins(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
):
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    correct = predictions.eq(labels)

    bins = []

    for i in range(num_bins):
        lower = i / num_bins
        upper = (i + 1) / num_bins

        if i == num_bins - 1:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if mask.any():
            bins.append(
                {
                    "confidence": float(
                        confidences[mask].mean().item()
                    ),
                    "accuracy": float(
                        correct[mask].float().mean().item()
                    ),
                    "count": int(mask.sum().item()),
                }
            )

    return bins
