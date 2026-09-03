import torch


def density_ratio_gradients(
    ds: torch.Tensor,
    dt: torch.Tensor,
    expected_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Equation (9) from the DRL paper.

    ds = P(source | x)
    dt = P(target | x)

    expected_score represents the expectation of the
    classifier score term appearing in Equation (9).
    """

    dt_safe = dt.clamp_min(1e-8)

    grad_ds = expected_score / dt_safe
    grad_dt = -(ds / (dt_safe ** 2)) * expected_score

    return grad_ds, grad_dt
