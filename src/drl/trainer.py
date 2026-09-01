import torch
import torch.nn as nn


class DRLTrainer:
    """
    DRL training implementation following the structure of
    Algorithm 1 in the paper.

    The implementation has three stages per batch:

    1. Train the source/target domain discriminator using Ld.
    2. Train the classifier using the DRL-weighted source loss.
    3. Update the density-ratio network using the target-loss
       gradients from Equation (9).

    This implementation uses our modular classifier and
    density-ratio network. The paper's exact image-backbone
    architecture is not specified in the main text.
    """

    def __init__(
        self,
        classifier: nn.Module,
        domain_network: nn.Module,
        classifier_optimizer: torch.optim.Optimizer,
        domain_optimizer: torch.optim.Optimizer,
    ) -> None:
        self.classifier = classifier
        self.domain_network = domain_network

        self.classifier_optimizer = classifier_optimizer
        self.domain_optimizer = domain_optimizer

        self.domain_loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def _freeze(model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _unfreeze(model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def domain_classification_step(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimize Ld, the binary source-vs-target classification loss.
        """

        self._freeze(self.classifier)
        self._unfreeze(self.domain_network)

        self.domain_optimizer.zero_grad(set_to_none=True)

        x = torch.cat(
            [source_x, target_x],
            dim=0,
        )

        labels = torch.cat(
            [
                torch.zeros(
                    source_x.shape[0],
                    dtype=torch.long,
                    device=source_x.device,
                ),
                torch.ones(
                    target_x.shape[0],
                    dtype=torch.long,
                    device=target_x.device,
                ),
            ],
            dim=0,
        )

        domain_logits = self.domain_network(x)

        loss = self.domain_loss_fn(
            domain_logits,
            labels,
        )

        loss.backward()
        self.domain_optimizer.step()

        return loss.detach()

    def classifier_step(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the classification network using the current
        density-ratio-adjusted source prediction.

        Equation (8) gives the standard softmax classification
        gradient structure, with the DRL probabilities used as
        the current prediction.
        """

        self._unfreeze(self.classifier)
        self._freeze(self.domain_network)

        self.classifier_optimizer.zero_grad(set_to_none=True)

        logits = self.classifier(source_x)

        with torch.no_grad():
            domain_probs = self.domain_network.domain_probabilities(
                source_x
            )

            source_prob = domain_probs[:, 0].clamp_min(1e-8)
            target_prob = domain_probs[:, 1].clamp_min(1e-8)

            density_ratio = source_prob / target_prob

        ratio = density_ratio.unsqueeze(1)

        # DRL temperature-like scaling from Ps(x) / Pt(x).
        drl_logits = logits * ratio

        loss = nn.functional.cross_entropy(
            drl_logits,
            source_y,
        )

        loss.backward()
        self.classifier_optimizer.step()

        return loss.detach()

    def density_ratio_target_step(
        self,
        target_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the density-ratio network using Equation (9).

        ds = P(source | x)
        dt = P(target  | x)

        Equation (9):

            grad_ds = E[score] / dt

            grad_dt = -(ds / dt^2) E[score]

        The classifier score is detached here because this stage
        is intended to supply gradients to the density estimator,
        not to update the classifier through Equation (9).
        """

        self._freeze(self.classifier)
        self._unfreeze(self.domain_network)

        self.domain_optimizer.zero_grad(set_to_none=True)

        domain_probs = self.domain_network.domain_probabilities(
            target_x
        )

        ds = domain_probs[:, 0].clamp_min(1e-8)
        dt = domain_probs[:, 1].clamp_min(1e-8)

        # The classifier score appearing in Equation (9):
        #
        # E_{P_hat(y|x)}
        #   [w · y phi_hat(x) + b]
        #
        # Our classifier logits represent:
        #   w · phi(x) + b
        #
        # Therefore the expected score is the probability-weighted
        # sum of the class logits.

        with torch.no_grad():
            classifier_logits = self.classifier(target_x)

            ratio = (ds / dt).detach()

            drl_logits = classifier_logits * ratio.unsqueeze(1)

            predictions = torch.softmax(
                drl_logits,
                dim=1,
            )

            expected_score = (
                predictions * classifier_logits
            ).sum(dim=1).mean()

        dt_safe = dt.clamp_min(1e-8)

        grad_ds = expected_score / dt_safe
        grad_dt = (
            -(ds / (dt_safe ** 2))
            * expected_score
        )

        torch.autograd.backward(
            tensors=[ds, dt],
            grad_tensors=[grad_ds, grad_dt],
        )

        self.domain_optimizer.step()

        return expected_score.detach()

    def train_step(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: torch.Tensor,
    ) -> dict[str, float]:
        """
        Perform one complete DRL training step.
        """

        domain_loss = self.domain_classification_step(
            source_x,
            target_x,
        )

        classification_loss = self.classifier_step(
            source_x,
            source_y,
        )

        target_score = self.density_ratio_target_step(
            target_x,
        )

        return {
            "domain_loss": float(domain_loss),
            "classification_loss": float(classification_loss),
            "target_score": float(target_score),
        }
