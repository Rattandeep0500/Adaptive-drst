import torch


def target_loss_density_gradients(
    ds,
    dt,
    classifier_logits,
):
    ds = ds.clamp_min(1e-8)
    dt = dt.clamp_min(1e-8)

    ratio = ds / dt

    drl_logits = classifier_logits * ratio.unsqueeze(1)

    probabilities = torch.softmax(
        drl_logits,
        dim=1,
    )

    expected_score = (
        probabilities * classifier_logits
    ).sum(dim=1).mean()

    grad_ds = expected_score / dt

    grad_dt = (
        -(ds / (dt ** 2))
        * expected_score
    )

    return grad_ds, grad_dt, expected_score
