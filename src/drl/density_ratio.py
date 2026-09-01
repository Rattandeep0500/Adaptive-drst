import torch
import torch.nn as nn


class DensityRatioNetwork(nn.Module):
    """
    Binary discriminator for source-vs-target domain prediction.

    Domain labels:
        0 = source
        1 = target

    The paper uses a discriminative network to estimate
    P(source | x) and P(target | x), which are then used
    to obtain the source/target density ratio via Bayes' rule.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns domain logits of shape [batch_size, 2].
        """
        return self.network(x)

    def domain_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            [P(source | x), P(target | x)]
        """
        return torch.softmax(self.forward(x), dim=-1)

    def density_ratio(
        self,
        x: torch.Tensor,
        source_prior: float = 0.5,
        target_prior: float = 0.5,
    ) -> torch.Tensor:
        """
        Estimate P_s(x) / P_t(x) using Bayes' rule.

        For equal source/target sampling priors:

            P_s(x) / P_t(x)
            = P(source | x) / P(target | x)

        More generally:

            P_s(x) / P_t(x)
            = [P(source | x) / P(target | x)]
              * [P(target) / P(source)]
        """
        probs = self.domain_probabilities(x)

        p_source_given_x = probs[:, 0].clamp_min(1e-8)
        p_target_given_x = probs[:, 1].clamp_min(1e-8)

        prior_correction = target_prior / source_prior

        return (
            p_source_given_x / p_target_given_x
        ) * prior_correction


def domain_loss(
    logits: torch.Tensor,
    domain_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Binary domain-classification loss.
    """
    return nn.functional.cross_entropy(logits, domain_labels)
