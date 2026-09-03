import torch


def adaptive_select_pseudo_labels(
    probabilities: torch.Tensor,
    min_threshold: float = 0.70,
    max_threshold: float = 0.95,
    max_per_class: int | None = None,
):
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")

    confidence, labels = probabilities.max(dim=1)
    num_classes = probabilities.shape[1]

    thresholds = torch.full(
        (num_classes,),
        min_threshold,
        dtype=confidence.dtype,
        device=confidence.device,
    )

    selected = []

    for cls in range(num_classes):
        mask = labels == cls

        if not mask.any():
            continue

        cls_indices = torch.where(mask)[0]
        cls_conf = confidence[cls_indices]

        threshold = torch.quantile(
            cls_conf,
            0.5,
        ).clamp(
            min=min_threshold,
            max=max_threshold,
        )

        thresholds[cls] = threshold

        valid = cls_indices[
            cls_conf >= threshold
        ]

        if max_per_class is not None:
            valid = valid[
                torch.argsort(
                    confidence[valid],
                    descending=True,
                )[:max_per_class]
            ]

        if valid.numel() > 0:
            selected.append(valid)

    if not selected:
        empty = torch.empty(
            0,
            dtype=torch.long,
            device=probabilities.device,
        )
        return (
            empty,
            empty,
            confidence[:0],
            thresholds,
        )

    indices = torch.cat(selected)

    return (
        indices,
        labels[indices],
        confidence[indices],
        thresholds,
    )