import torch
import torch.nn as nn


class DRLTrainer:
    """
    Faithful DRL training structure.

    Per minibatch:

    1. Accumulate the Eq. 9 task gradient into the domain network.
    2. Accumulate the binary source-vs-target domain-loss gradient.
    3. Perform ONE domain-network optimizer step.
    4. Recompute the DRL ratio using the updated domain network.
    5. Update the classifier using the fresh DRL ratio.

    The classifier and domain networks remain separate.
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

    def _domain_labels(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
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

    def _domain_batch(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ):
        x = torch.cat(
            [source_x, target_x],
            dim=0,
        )

        labels = self._domain_labels(
            source_x,
            target_x,
        )

        return x, labels

    def _expected_score(
        self,
        target_x: torch.Tensor,
        ds: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expected unscaled classifier score used by Eq. 9.

        Important:
        this is based on the classifier logits themselves,
        NOT the DRL-scaled logits and NOT post-softmax
        probabilities as the score.
        """

        with torch.no_grad():
            classifier_logits = self.classifier(
                target_x
            )

            ratio = (
                ds / dt.clamp_min(1e-8)
            )

            drl_logits = (
                classifier_logits
                * ratio.unsqueeze(1)
            )

            probabilities = torch.softmax(
                drl_logits,
                dim=1,
            )

            expected_score = (
                probabilities
                * classifier_logits
            ).sum(dim=1).mean()

        return expected_score

    def domain_task_gradient_step(
        self,
        target_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Accumulate Eq. 9 gradients without stepping.

        This method intentionally does NOT call optimizer.step().
        It exists so the Eq. 9 gradient and domain classification
        gradient can be accumulated before exactly one domain update.
        """

        self._freeze(self.classifier)
        self._unfreeze(self.domain_network)

        domain_probs = (
            self.domain_network.domain_probabilities(
                target_x
            )
        )

        ds = domain_probs[:, 0].clamp_min(
            1e-8
        )
        dt = domain_probs[:, 1].clamp_min(
            1e-8
        )

        expected_score = self._expected_score(
            target_x,
            ds,
            dt,
        )

        dt_safe = dt.clamp_min(1e-8)

        grad_ds = (
            expected_score / dt_safe
        )

        grad_dt = (
            -(
                ds
                / (dt_safe ** 2)
            )
            * expected_score
        )

        torch.autograd.backward(
            tensors=[ds, dt],
            grad_tensors=[grad_ds, grad_dt],
        )

        return expected_score.detach()

    def domain_classification_loss(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the domain classification loss.

        This method accumulates its gradient into the domain
        network but does not perform the optimizer step.
        """

        self._freeze(self.classifier)
        self._unfreeze(self.domain_network)

        x, labels = self._domain_batch(
            source_x,
            target_x,
        )

        domain_logits = self.domain_network(x)

        loss = self.domain_loss_fn(
            domain_logits,
            labels,
        )

        loss.backward()

        return loss.detach()

    def domain_update(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ):
        """
        Perform the complete discriminator-first phase.

        Eq. 9 gradient + domain classification gradient
        are accumulated before ONE domain optimizer step.
        """

        self._freeze(self.classifier)
        self._unfreeze(self.domain_network)

        self.domain_optimizer.zero_grad(
            set_to_none=True
        )

        target_score = (
            self.domain_task_gradient_step(
                target_x
            )
        )

        domain_loss = (
            self.domain_classification_loss(
                source_x,
                target_x,
            )
        )

        self.domain_optimizer.step()

        return (
            domain_loss,
            target_score,
        )

    def domain_classification_step(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility API.

        Unlike train_step(), this method performs only the
        standalone domain-classification optimization.
        """

        self._freeze(self.classifier)
        self._unfreeze(self.domain_network)

        self.domain_optimizer.zero_grad(
            set_to_none=True
        )

        x, labels = self._domain_batch(
            source_x,
            target_x,
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
        Compatibility API.

        Uses the CURRENT domain network and therefore a fresh
        density ratio at the time this method is called.
        """

        self._unfreeze(self.classifier)
        self._freeze(self.domain_network)

        self.classifier_optimizer.zero_grad(
            set_to_none=True
        )

        logits = self.classifier(
            source_x
        )

        with torch.no_grad():
            domain_probs = (
                self.domain_network.domain_probabilities(
                    source_x
                )
            )

            source_prob = (
                domain_probs[:, 0]
                .clamp_min(1e-8)
            )

            target_prob = (
                domain_probs[:, 1]
                .clamp_min(1e-8)
            )

            density_ratio = (
                source_prob
                / target_prob
            )

            drl_logits = (
                logits
                * density_ratio.unsqueeze(1)
            )

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
        Compatibility API for the standalone Eq. 9 step.

        The combined train_step() should be preferred because
        the faithful algorithm accumulates Eq. 9 with the domain
        classification gradient before one domain update.
        """

        self._freeze(self.classifier)
        self._unfreeze(self.domain_network)

        self.domain_optimizer.zero_grad(
            set_to_none=True
        )

        target_score = (
            self.domain_task_gradient_step(
                target_x
            )
        )

        self.domain_optimizer.step()

        return target_score

    def train_step(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: torch.Tensor,
    ) -> dict[str, float]:
        """
        One faithful DRL minibatch.

        Correct ordering:

            Eq. 9 gradient
                    +
            domain BCE gradient
                    |
                    v
            one domain update
                    |
                    v
            fresh ratio
                    |
                    v
            classifier update
        """

        if source_x.ndim < 2:
            raise ValueError(
                "source_x must contain a batch dimension"
            )

        if target_x.ndim < 2:
            raise ValueError(
                "target_x must contain a batch dimension"
            )

        if source_x.shape[0] == 0:
            raise ValueError(
                "source_x cannot be empty"
            )

        if target_x.shape[0] == 0:
            raise ValueError(
                "target_x cannot be empty"
            )

        if source_y.shape[0] != source_x.shape[0]:
            raise ValueError(
                "source_x and source_y batch sizes must match"
            )

        # --------------------------------------------------
        # 1-3. Domain network:
        #      Eq. 9 + domain classification, then ONE step.
        # --------------------------------------------------

        domain_loss, target_score = (
            self.domain_update(
                source_x=source_x,
                target_x=target_x,
            )
        )

        # --------------------------------------------------
        # 4-5. Recompute the DRL ratio using the UPDATED
        #      domain network, then update the classifier.
        # --------------------------------------------------

        self._unfreeze(self.classifier)
        self._freeze(self.domain_network)

        self.classifier_optimizer.zero_grad(
            set_to_none=True
        )

        source_logits = self.classifier(
            source_x
        )

        with torch.no_grad():
            updated_domain_probs = (
                self.domain_network.domain_probabilities(
                    source_x
                )
            )

            source_prob = (
                updated_domain_probs[:, 0]
                .clamp_min(1e-8)
            )

            target_prob = (
                updated_domain_probs[:, 1]
                .clamp_min(1e-8)
            )

            density_ratio = (
                source_prob
                / target_prob
            )

        drl_logits = (
            source_logits
            * density_ratio.unsqueeze(1)
        )

        classification_loss = (
            nn.functional.cross_entropy(
                drl_logits,
                source_y,
            )
        )

        classification_loss.backward()
        self.classifier_optimizer.step()

        return {
            "domain_loss": float(
                domain_loss
            ),
            "classification_loss": float(
                classification_loss.detach()
            ),
            "target_score": float(
                target_score
            ),
        }
